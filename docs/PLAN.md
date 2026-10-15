# RoentGen Reimplementation on CheXpert — Code Plan

## Context

We are reimplementing *A vision–language foundation model for the generation of realistic chest X-ray images* (Bluethgen et al., Nat BME 2025 — "RoentGen") for the IE643 project (Team NeuroSight). The paper domain-adapts Stable Diffusion 1.4 to chest X-rays using MIMIC-CXR image–report pairs on 64×A100s, then evaluates image fidelity, text–image alignment, and the value of synthetic CXRs as training data.

Our constraints differ from the paper in three ways that drive every design decision below:

1. **Dataset**: CheXpert-v1.0-small (public, no credentialing) instead of MIMIC-CXR. CheXpert ships **labels, not reports**, so free-text conditioning must be synthesised from labels via templates.
2. **Compute**: a dedicated **RTX 4070 desktop** (12 GB, 200 W, exclusive access, overnight runs permitted) as primary, with Kaggle as overflow and an RTX 4050 laptop for code development. Fine-tuning at the paper's 64×A100 scale is still impossible → **LoRA** on U-Net + text encoder as the primary model, plus a **full U-Net fine-tune at 256 px** as an ablation. See *Compute strategy* below.
3. **Scope**: five allotted pathologies — Pleural Other, Pneumonia, Lung Lesion, Fracture, Enlarged Cardiomediastinum.

Intended outcome: a reproducible pipeline that fine-tunes SD 1.4 on CheXpert, generates pathology-conditioned synthetic CXRs, evaluates them with the paper's metric families, and answers the project's research question — *does Real+Synthetic training beat Real-only for these five pathologies?*

**Confirmed decisions** (from clarification): templated label→text prompts; LoRA on U-Net + text encoder; core metrics + BiomedCLIP as the open substitute for the RRG/retrieval evaluations; downstream experiment limited to Real vs Synth vs Mixed at 1/10/100%.

---

## Compute strategy

**Primary machine: RTX 4070 desktop** (`sedrica`) — AD104, 12 GB GDDR6X, 504 GB/s, 200 W, sm_89 with bf16/fp8. Exclusive access, overnight runs permitted.

| | RTX 4070 (primary) | RTX 4050 laptop (dev) | Kaggle P100 | Kaggle T4×2 |
|---|---|---|---|---|
| VRAM | **12 GB** | 6 GB | 16 GB | 16 × 2 |
| Bandwidth | 504 GB/s | 192 GB/s | 732 GB/s | 320 GB/s |
| bf16 / fp8 | **yes** | yes | no | no |
| Rel. speed (SD training) | **~2.5×** | 1.0× | ~1.0× | ~1.0× each |

The 4070 is ~2.5–3× a single T4 and faster than the P100 (which has no tensor cores, so its bandwidth lead doesn't apply to this workload). **Run the whole project on it**; Kaggle becomes optional overflow, not the backbone. This also removes the 9–12 h session ceiling, the 30 GPU-h/week quota, and all checkpoint-resume-across-sessions machinery from the critical path.

**Division of labour**

| Stage | Where | Est. time |
|---|---|---|
| 1. Manifest, splits, latent/token cache | 4070 | ~1 h |
| 2. LoRA training (runs A, B) | 4070, overnight | ~3–4 h each |
| 2c. Full U-Net fine-tune @256 (run C) | 4070, overnight | ~6–8 h |
| 3. Generation (20k images) | 4070 | ~4–6 h |
| 4. Metrics | 4070 | ~2 h |
| 5. 27 downstream runs | 4070, unattended | ~11 h |
| Code dev, smoke tests, figures, writing | 4050 laptop | — |
| Overflow / parallel sweeps | Kaggle | as needed |

**VRAM budgets** (≈11.4 GB usable; the desktop session, Firefox and TeamViewer currently hold 650 MB — close them or drop to a TTY for the full-fine-tune run):

- *Run A — LoRA(U-Net + text encoder) @ 512, bf16, bs 4 + grad ckpt*: ~8–9 GB. Primary model.
- *Run B — LoRA(U-Net only) @ 512*: ~8 GB. Isolates the text encoder's contribution.
- *Run C — full U-Net fine-tune, text encoder frozen, @ 256, bf16 + 8-bit Adam + grad ckpt, bs 4*: U-Net bf16 1.7 + fp32 master 3.4 + 8-bit Adam states 1.7 + grads 1.7 + activations ~1.0 ≈ **9.5 GB**. Fits with margin.

Run C maps **exactly onto the paper's Table 1 "SD U-Net only, 60k" row** (FID_XRV 9.2), and A maps onto "SD U-Net + CLIP" (FID_XRV 3.6) — so the A/B/C table reproduces the paper's two central architectural claims (text-encoder adaptation matters; joint adaptation is best) rather than merely gesturing at them. Keeping the text encoder frozen in C is both what the paper did for that row and what keeps the run inside 12 GB.

**Latent caching** is now a speed optimisation rather than a necessity: precompute frozen-VAE latents at 512 (and 256 for run C) + CLIP token IDs once, costing ~1.3 GB per resolution, and training skips VAE encoding entirely for ~25% faster steps. Trade-off: fixed latents mean no random crop/flip — the paper used no augmentation either, so this costs nothing.

**Disk — confirmed ample**: 124 GB free on NVMe (`/dev/nvme1n1p1`), against a ~47 GB requirement (CheXpert-small 11 GB, synthetic PNGs ~8 GB, SD weights 5 GB, latent caches 3 GB, full-ft checkpoints ~10 GB, conda/torch ~10 GB). No external storage or symlink workaround needed. Keep ~20 GB headroom for regenerated synthetic corpora across the A/B/C checkpoints.

**RAM — ~126 GB, use it.** `/dev/shm` is 63 GB, so stage the hot data in RAM and remove disk I/O from the training loop entirely:
- Copy the 11 GB CheXpert image tree (or the ~3 GB latent caches) to `/dev/shm/cxr/` at session start and point `--data_root` there. At 512 px with bs 4 the dataloader is otherwise a real bottleneck on JPEG decode.
- Set `num_workers` to roughly `nproc - 2` with `persistent_workers=True` and `pin_memory=True`; with this much RAM there's no reason to be conservative.
- `/dev/shm` is volatile — it clears on reboot. Treat it as a cache populated by a one-line setup step, never as the source of truth.

**Hygiene for long runs**: torch cu124/cu128 build (driver 580 supports it); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; bf16 throughout (no GradScaler needed on Ada); 8-bit Adam via `bitsandbytes`; gradient checkpointing on; run under `tmux` so an SSH/TeamViewer drop doesn't kill the job; log `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` alongside the loss. At 200 W in a desktop chassis thermal throttling is unlikely, unlike the laptop.

---

## Why this setup over Kaggle-only

The plan was originally scoped for Kaggle alone. Moving to the 4070 changes enough that the comparison is worth recording — both to justify the choice and because Kaggle retains two genuine roles.

| Dimension | Kaggle-only | **4070 desktop (chosen)** |
|---|---|---|
| GPU | P100 16 GB / T4×2 | RTX 4070 12 GB |
| Relative training speed | 1.0× | **~2.5×** |
| Session limit | 9–12 h hard cutoff | **none** |
| Weekly quota | 30 GPU-h | **unlimited** |
| Project wall-clock | ~2–3 weeks, quota-rationed | **~4 overnight runs** |
| VRAM | **16 GB** | 12 GB |
| Disk | ~20 GB working | **124 GB** |
| RAM | ~30 GB | **126 GB** |
| bf16 | no (P100/T4 are fp16-only) | **yes** |
| Dataset | **mounted free, no download** | 11 GB one-time download |
| Resume engineering | **mandatory** — cross-session, via Dataset round-trip | nice-to-have |
| Bug-fix iteration | minutes (restart, remount, reinstall) | **seconds** |
| Multi-GPU | T4×2 available | single GPU |
| Shareable artifact | **yes — runnable notebook** | no |

**What the switch buys**

1. **~3× faster per hour *and* unlimited hours.** These compound. The binding constraint on Kaggle was never raw speed — it was the 30 GPU-h/week quota against a ~32 GPU-h project, i.e. a minimum of two weeks with zero room for the re-runs that mistakes inevitably force.
2. **Run C becomes affordable.** A full U-Net fine-tune would consume most of a week's Kaggle quota on its own; it's one overnight run here. This is the difference between *disclaiming* the LoRA approximation and *measuring* it.
3. **Deletes an entire category of engineering** — checkpoint/resume across sessions, optimiser-state round-trips through Kaggle Datasets, dataloader RNG offset restoration. Still worth implementing for crash recovery, but off the critical path.
4. **bf16** instead of fp16 + GradScaler: wider dynamic range, no loss-scale tuning, fewer silent NaNs on a long unattended run.
5. **Iteration speed.** Most of the real cost of this project is finding bugs, not running the final job.

**What Kaggle still wins, and the two roles it keeps**

- **16 GB > 12 GB.** If we ever wanted the full U-Net fine-tune at *512* px rather than 256, the P100's 16 GB is the only machine here that can host it. Keep this in reserve as a stretch ablation.
- **Reproducibility for graders.** A local run is not inspectable by anyone else. Ship a thin Kaggle notebook that reproduces Stage 1, runs a short training smoke test, and generates samples from the released LoRA weights — the course deliverable then includes something a marker can actually execute. This is the more important of the two roles.
- Also useful: T4×2 for running two hyperparameter configs in parallel while the 4070 is busy with the main run.

---

## Session handoff (moving to Claude Code on `sedrica`)

Claude Code sessions are stored per-machine and per-directory (`~/.claude/projects/<cwd>/`); `--continue` / `--resume` cannot see sessions from another machine. **A new session on `sedrica` starts with no memory of this conversation.** Carry the context over in files:

1. **Copy this plan into the repo** as `docs/PLAN.md` and commit it. It is written to be self-contained — stage-by-stage, with hyperparameters, VRAM budgets and verification steps — precisely so it survives the handoff.
2. **Add a `CLAUDE.md`** at the repo root recording what a fresh session cannot infer from code:
   - The four locked decisions: templated label→text prompts; LoRA(U-Net + text encoder) primary; BiomedCLIP as the open substitute for RRG/retrieval; downstream limited to Real/Synth/Mixed at 1/10/100%.
   - Hardware facts: 4070 12 GB, bf16, stage hot data in `/dev/shm`, per-run VRAM budgets.
   - The non-obvious traps: patient-level splits before stratification; `u = -1` → positive; Pleural Other is absent from TorchXRayVision.
3. Open the first session with *"read docs/PLAN.md and CLAUDE.md, then start Stage 1"*.

---

## Repository layout

Develop locally in `/home/avi/Desktop/IE643`, push to GitHub, and have each Kaggle notebook do `!git clone` — far easier to iterate than editing cells.

```
roentgen-chexpert/
  src/roentgen/
    config.py          # paths, label list, hyperparams (one place; env-switch local/Kaggle)
    chexpert.py        # manifest build, uncertainty policy, patient-level splits
    prompts.py         # label tuple -> templated medical prompt
    data.py            # DiffusionCXRDataset (512px, prompt) + ClassifierCXRDataset (224px, multilabel)
    train_lora.py      # Stage 2: LoRA fine-tune of SD1.4 U-Net + CLIP text encoder
    generate.py        # Stage 3: batched conditioned sampling + manifest of synthetic set
    filtering.py       # Stage 3b: quality/consistency filter for generated images
    eval_fidelity.py   # Stage 4a: FID (XRV + InceptionV3), MS-SSIM intra-prompt diversity
    eval_classify.py   # Stage 4b: pretrained-classifier AUROC on synthetic CXRs
    eval_clip.py       # Stage 4c: BiomedCLIP alignment + image-image / image-text retrieval
    train_classifier.py# Stage 5: DenseNet-121, real / synth / mixed x {1,10,100}%
    aggregate.py       # collect JSON results -> tables + figures
    utils.py           # seeding, checkpoint resume, logging, image IO
  notebooks/
    00_eda.ipynb              # already done (slides 17) — port in as-is
    01_prepare_manifest.ipynb
    02_train_lora.ipynb       # resumable, one session = one chunk of steps
    03_generate.ipynb
    04_evaluate_generation.ipynb
    05_downstream_classification.ipynb
  app/gradio_demo.py   # Stage 6 interface: prompt -> synthetic CXR
  configs/*.yaml
  results/             # JSON per experiment, committed so plots are reproducible
```

Every entry point takes `--smoke` to run a handful of steps/images for a fast sanity pass before burning a session.

---

## Stage 0 — Environment (`notebooks/*` header cell)

Pin versions; `diffusers` changes LoRA APIs often.

```
torch (Kaggle preinstalled) ; diffusers==0.31.0 ; transformers==4.44.2 ; peft==0.13.2
accelerate==1.0.1 ; safetensors ; torchmetrics==1.4.* ; torchxrayvision==1.2.*
open_clip_torch==2.24.* ; pytorch-fid or torchmetrics FID ; bitsandbytes (8-bit Adam)
```

Base weights: `CompVis/stable-diffusion-v1-4`. **Disable the safety checker** (`safety_checker=None`) — the paper notes it has a high false-positive rate on medical prompts. Add CheXpert-v1.0-small as a Kaggle input dataset (no download).

---

## Stage 1 — Dataset preparation (`chexpert.py`, `prompts.py`, `data.py`)

Paper §Methods "Datasets" — adapted.

**Manifest** (`build_manifest()` → `manifest.parquet`):
- Read `train.csv`, filter to **frontal (AP/PA)** views only.
- **Uncertainty policy**: follow the paper — `u = -1` → **positive**, blank → negative. This is the "positive + uncertain-as-positive" column from your EDA (Pneumonia 6,039 → 24,809). Make it a flag `--uncertain {positive,ignore,negative}` so the alternative is one run away.
- Keep the 5 target labels + `No Finding` (needed as the negative/normal pool for both generation diversity and the classifier).
- **Patient-level splits.** CheXpert paths encode `patient#####/study#/view#`; group by patient ID before splitting or the same patient leaks across train/test and inflates every number. Stratify by label combination (iterative stratification, `skmultilearn`, as in the paper) → 80/10/10 train/val/test.
- Keep the official CheXpert `valid.csv` (234 radiologist-labelled images) as a small clean secondary test set.
- **Class balance**: cap the dominant combinations; record per-label counts in the manifest metadata so the report can cite exact n.

**Prompts** (`prompts.py`):
- Map each label to natural radiological phrasing: `Pleural Other → "pleural abnormality"`, `Enlarged Cardiomediastinum → "enlarged cardiomediastinum"`, etc.
- Template bank, sampled at random per example for lexical variety:
  - `"chest x-ray with {findings}"`, `"frontal chest radiograph showing {findings}"`, `"{view} chest x-ray demonstrating {findings}"`, `"chest radiograph, no acute cardiopulmonary findings"` (for No Finding).
- Multi-label images get `"a and b"` / Oxford-comma joins → this is what buys the paper's **compositionality** result (Fig. 4).
- Hold back a small set of **modifier prompts** (`"small left-sided pleural abnormality"`, `"right lower lobe pneumonia"`) used only at evaluation, to reproduce the paper's Fig. 3 laterality/extent demonstration qualitatively.
- Assert every prompt is ≤ 77 CLIP tokens (ours always will be; the paper had to drop 14% of reports for this).

**Image pipeline** (`data.py`):
- Diffusion: grayscale → 3-channel, centre-crop to square, resize to **512×512**, `[-1,1]`. Note in the report: CheXpert-small is ~320 px on the short side, so 512 involves upsampling; we accept this because SD 1.4's VAE/U-Net are native-512 and off-resolution training degrades quality more than interpolation does. A 320-px ablation is a cheap optional run.
- Classifier: centre-crop + resize **224×224**, no augmentation (paper does exactly this), ImageNet normalisation.

---

## Stage 2 — LoRA domain adaptation (`train_lora.py`)

Paper §"Fine-tuning of a pre-trained LDM" + Table 1, adapted to one GPU.

- **Frozen**: VAE (the paper froze it too — ref. 48 showed it needs no CXR adaptation).
- **Trained**: LoRA adapters via `peft` on
  - U-Net attention: `to_q, to_k, to_v, to_out.0` (both self- and cross-attention), rank 16, alpha 16, dropout 0.0;
  - CLIP text encoder: `q_proj, k_proj, v_proj, out_proj`, rank 8.
  This mirrors the paper's headline finding that *jointly* adapting U-Net **and** text encoder gives the best fidelity (FID_XRV 3.6) and factual correctness.
- **Objective**: standard ε-prediction MSE (paper Eq. 1–3) — `noise_scheduler.add_noise`, U-Net predicts ε, MSE against sampled ε. `diffusers`' `train_text_to_image_lora.py` is the reference; we write our own loop so resume and logging are under our control.
- **Hyperparameters**: fp16 mixed precision, SDPA/xformers attention, gradient checkpointing, batch 4 + grad-accum 8 → effective 32. LR **1e-4** with constant schedule and 500 warm-up steps (the paper's 5e-5 is for *full* fine-tuning; LoRA needs ~10× higher). 10% prompt dropout (empty string) so classifier-free guidance works properly at sampling time.
- **Budget**: target ~15–20k optimiser steps at effective batch 32 (bs 4 × accum 8). On the 4070 this is a single overnight run of ~3–4 h.
- **Checkpointing**: save every 500 steps (LoRA weights ~20 MB; full-ft ~3.4 GB, so keep only the last 2 + best). `--resume_from latest` restores weights, optimiser state, LR scheduler, step counter and dataloader RNG offset. Still worth implementing even without session limits — it's what lets you recover from a crash 6 hours in.
- **Logging**: loss curve + a fixed grid of 6 validation prompts sampled every 1k steps, saved as PNG, so you can *see* the domain shift happening (noise → vaguely-thoracic → real CXR). This grid is the single best slide for the mid-term presentation.
**Ablation matrix** — three runs, mapping directly onto the paper's Table 1:

| Run | What's trained | Res | ≈ Paper row | VRAM | Time |
|---|---|---|---|---|---|
| **A** (primary) | LoRA: U-Net + text encoder | 512 | "SD U-Net + CLIP" (FID_XRV 3.6) | ~8–9 GB | 3–4 h |
| **B** | LoRA: U-Net only, CLIP frozen | 512 | "SD U-Net only" (FID_XRV 9.2) | ~8 GB | ~3 h |
| **C** | **Full** U-Net weights, CLIP frozen | 256 | full-weight counterpart of B | ~9.5 GB | 6–8 h |

A vs B reproduces the paper's claim that adapting the text encoder matters. B vs C isolates **LoRA vs full-weight** adaptation — the one axis our method deviates on — so the report can quantify what the LoRA approximation costs instead of merely disclaiming it. Run C uses bf16 + 8-bit Adam + gradient checkpointing and keeps CLIP frozen, which is both faithful to that paper row and what keeps it inside 12 GB. Compare A/B at 512 only; C is at 256, so compare it to a 256-px re-evaluation of B, never to A directly.

---

## Stage 3 — Conditioned generation (`generate.py`, `filtering.py`)

Paper §"Model architecture" sampling settings.

- Sampler: **PNDM, 75 inference steps, guidance scale 4.0** — the paper's exact settings. Also expose DPM-Solver++ @ 25 steps; at 512 px, 75 PNDM steps is roughly 3–4 s/image on a P100, so 20k images ≈ 20+ h ≈ two full sessions. Recommendation: generate the main synthetic corpus with DPM-Solver++ 25 steps (≈3× faster), and regenerate a 2k-image subsample with PNDM-75 to verify the fast sampler costs nothing on FID. Report both.
- Generation set: for each training-split real example, generate 1 synthetic image from its prompt (gives a label-matched 1:1 synthetic twin set, which is exactly what the "mixed" experiment needs). Plus an extra per-pathology balanced pool so rare classes (Pleural Other, n≈6k) can be over-sampled.
- Deterministic seeds recorded per image → every figure is regenerable.
- Write images as PNG + `synthetic_manifest.parquet` (path, prompt, label vector, seed, sampler, checkpoint step).
- **Filtering** (`filtering.py`, your slide-13 "filter low-quality samples"): drop images whose XRV classifier score for the prompted pathology falls below a threshold, and degenerate outputs (near-constant images / extreme intensity histograms). Report the rejection rate per pathology — a genuinely interesting result, since it quantifies how often conditioning fails, which is the paper's third stated limitation.

---

## Stage 4 — Evaluation of generated images

### 4a. Fidelity & diversity (`eval_fidelity.py`) — paper Table 1
- **FID_XRV**: features from `torchxrayvision` DenseNet-121 (`densenet121-res224-all`, 1024-d) — the paper's in-domain FID. This is the headline number.
- **FID_InceptionV3**: 2048-d pool3, for comparability with the generic literature.
- Compute against the held-out **real test split**, 5k vs 5k images.
- **MS-SSIM**: pairwise over **10 samples per prompt**, Gaussian kernel 11, σ 1.5, averaged — the paper's intra-prompt diversity metric. Lower = more diverse. Report as mean ± SD, and always jointly with FID (the paper warns that diverse-but-unrealistic images score well on MS-SSIM alone).
- Baselines to anchor the table: raw SD 1.4 (no fine-tuning) and the LoRA checkpoint at 1k steps — replicating the paper's "even 1k steps beats the baseline massively" result (FID_XRV 47.7 → ~6).

### 4b. Multi-label classification of synthetic CXRs (`eval_classify.py`) — paper Table 2
- Classify synthetic images with a classifier trained on **real** CXRs; prompt labels are the ground truth. Report AUROC per label + macro.
- **Caveat worth planning around**: TorchXRayVision covers Pneumonia, Lung Lesion, Fracture and Enlarged Cardiomediastinum, but **not Pleural Other**. So train our own DenseNet-121 on the real train split over all 5 labels (this is the same model Stage 5 needs as its Real-only baseline — build it once, reuse it) and report XRV numbers alongside for the 4 labels it supports.
- Upper bound row: the same classifier on **real** test images (paper's `Real CXR (D_Test)` row, macro 0.82).

### 4c. Vision–language correctness (`eval_clip.py`) — substitute for paper Tables 3–4
The paper's RRG / ConVIRT / CXR-RePaiR / RadGraph stack needs MIMIC-credentialed weights. Open substitute: **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, via `open_clip`).
- **Text–image alignment**: cosine similarity between the generating prompt and the synthetic image, vs. mismatched prompts → a CLIP-score and a zero-shot 5-way pathology accuracy.
- **Image–image retrieval** (paper Table 4 left): 200 synthetic queries, 400 real candidates, rank by BiomedCLIP image-embedding cosine similarity, score Precision@k for k ∈ {5,10,50} by label match. Real-query upper bound included.
- **Image–text retrieval**: query = synthetic image, candidates = the prompt bank, Precision@k by label match.
- State plainly in the report that this substitutes for, and is not equivalent to, RadGraph/F1-CheXbert.

### 4d. Qualitative (notebook cells)
Reproduce the paper's figures with our prompts: Fig. 2 (one panel per pathology), Fig. 3 (laterality/size modifiers — checks whether the model learned the radiological left/right convention), Fig. 4 (two findings composed in one image). Optional stretch: DAAM cross-attention heatmaps (paper ref. 60) for the attribution overlays — visually the strongest slide in the deck.

---

## Stage 5 — Downstream classification (`train_classifier.py`) — paper Extended Data Fig. 2

The project's actual research question.

- Model: **DenseNet-121, ImageNet-pretrained**, 224×224, 5 output logits, BCE-with-logits.
- Optimiser: AdamW, lr 1e-4, wd 0.05 (paper's exact settings). Paper runs 100 epochs; on Kaggle use early stopping on val macro-AUROC with patience 10 and a 30-epoch cap.
- **Matrix**: {1%, 10%, 100%} of the real train split × {Real-only, Synthetic-only, Mixed (equal real+synthetic)} × **3 seeds** (paper uses 5; drop to 3 if sessions get tight) = 27 runs. Select checkpoint by best val AUROC; evaluate on the held-out real test split.
- No data augmentation — matches the paper, and keeps synthetic-data gains uncontaminated by augmentation gains.
- Report macro-AUROC mean ± SD and **% change vs the Real-only baseline of the same training-set size**, which is how the paper tabulates it. Expect the paper's shape: synthetic-only slightly below real, mixed equal-or-better with the gain concentrated at 1% and 10%.
- Deferred by your decision, kept as documented stretch goals in the report's future-work section: synthetic surplus 2×/4×, SimSiam SSL pre-training, and an out-of-distribution test set.

---

## Stage 6 — Interface & reporting (`app/gradio_demo.py`, `aggregate.py`)

- Gradio app: prompt box + pathology checkboxes + guidance/steps sliders → generated CXR, with a side panel showing the classifier's prediction on the generated image (nice demonstration that the conditioning worked). Runs in a Kaggle notebook with `share=True`.
- `aggregate.py`: read `results/*.json` → the three paper-style tables (fidelity/diversity, classification of synthetic, downstream) + Extended-Data-Fig-2-style AUROC-vs-training-set-size curves with 95% CIs.

---

## Execution order & session budget

All GPU stages run on the 4070; code development and writing happen on the laptop in parallel.

| # | Stage | Est. time |
|---|-------|-----------|
| 1 | Manifest + prompts + splits + latent/token cache | ~1 h |
| 2 | Run A — LoRA(U-Net + text enc) @512 | 3–4 h |
| 2b| Run B — LoRA(U-Net only) @512 | ~3 h |
| 2c| Run C — full U-Net ft @256 | 6–8 h (overnight) |
| 3 | Generate + filter synthetic corpus (20k) | 4–6 h |
| 4 | FID / MS-SSIM / AUROC / BiomedCLIP | ~2 h |
| 5 | 27 downstream runs | ~11 h (unattended) |
| 6 | Demo + aggregation + report | — |

≈ 32 GPU-hours total, i.e. about four overnight runs. Run stage 2 to completion and sanity-check its sample grid before launching 2b/2c — a bug found at stage 2 costs 4 hours, the same bug found at stage 5 costs 30.

---

## Verification

Run each in order; every one is cheap and catches a different class of failure.

0. **GPU sanity (4070)**: install torch cu124+; confirm `torch.cuda.is_bf16_supported()` and `get_device_capability() == (8, 9)`. Then run one backward pass for each configuration and check `torch.cuda.max_memory_allocated()` against the budgets above — A ≈ 8–9 GB, C ≈ 9.5 GB. If A exceeds ~10 GB, gradient checkpointing isn't active or the VAE is still resident despite the latent cache. Do this *before* committing to an overnight run.
1. **Manifest**: `python -m roentgen.chexpert --build` → assert no patient ID appears in two splits; per-label positives match the EDA slide (Enlarged Cardiomediastinum 23,201 / Lung Lesion 10,674 / Fracture 9,682 / Pneumonia 24,809 / Pleural Other 6,176 under the uncertain-as-positive policy).
2. **Data/prompt smoke**: print 20 random (prompt, label-vector) pairs; confirm all ≤ 77 CLIP tokens; save a 4×4 image grid and eyeball orientation/contrast.
3. **Training smoke**: `train_lora.py --smoke` (20 steps, 32 images) → loss is finite and decreasing, a checkpoint directory appears, a resume from it reproduces the same loss at the same step.
4. **Baseline sanity**: generate 8 images from the *unmodified* SD 1.4 with a CXR prompt — they should look nothing like radiographs. This is the before/after evidence that domain adaptation did something.
5. **Metric sanity**: FID of real-test vs real-train ≈ small (< ~5 on XRV features); FID of real vs raw-SD output should land near the paper's 47.7 order of magnitude. If real-vs-real FID is large, the feature extractor or preprocessing is wrong, not the model.
6. **Classifier sanity**: Real-only 100% baseline should reach macro-AUROC in a plausible band (~0.70–0.80 on these five hard, low-prevalence labels — do not expect the paper's 0.82, which is on easier, more prevalent findings like effusion and cardiomegaly).
7. **End-to-end**: launch the Gradio app, prompt `"frontal chest radiograph showing a large right-sided pleural abnormality"`, confirm the output is a plausible CXR and the side-panel classifier fires on Pleural Other.

---

## Known deviations from the paper (state these explicitly in the report)

1. CheXpert labels + templates instead of MIMIC-CXR free-text impressions — weaker, lower-entropy conditioning signal.
2. LoRA instead of full U-Net + text-encoder fine-tuning — expect worse FID than the paper's 3.6. Run C quantifies this gap at 256 px rather than leaving it an untested assumption.
3. 512-px training on ~320-px source images (CheXpert-small) — real high-frequency detail is capped.
4. BiomedCLIP retrieval in place of ConVIRT / CXR-RePaiR / RadGraph / F1-CheXbert.
5. Five low-prevalence pathologies instead of the paper's 14 — absolute AUROCs will be lower across the board, so all comparisons must be *relative to our own Real-only baseline*, never to the paper's numbers.
