# Code map

What each code file does. Every module runs as `python -m roentgen.<name>` after `source env.sh`;
the module docstrings hold the full usage and options.

## Setup / helpers (repo root)

| File | What it does |
|---|---|
| `env.sh` | Source before any work. Keeps every cache (uv, HF, torch, pip, tmp, pycache) inside the project folder, puts `src/` on `PYTHONPATH`, activates `.venv`. |
| `jobstatus.sh` | Quick look at background jobs: running `roentgen.*` processes, tail of the Stage 3 generate log, PNG counts under `artifacts/synth/*/png`, GPU memory. |

## `src/roentgen/` — pipeline modules

### Shared
| File | What it does |
|---|---|
| `config.py` | Single source of truth: paths (`ROOT`, `ARTIFACTS`, manifest, cache dir), SD model id, the 5 target labels + "No Finding", seed 643, split fractions, per-combination cap (8000). |
| `prompts.py` | Turns a label vector into a templated radiology prompt (finding / normal / generic templates). Also holds the unseen `MODIFIER_PROMPTS` used for the Fig. 3 demo. |
| `data.py` | Image loading (grayscale → centre-crop → resize → RGB), `read_manifest()`, `DiffusionCXRDataset` (512 px, [-1,1], prompt) and `ClassifierCXRDataset` (224 px, multilabel, ImageNet norm). |

### Stage 1 — data
| File | What it does |
|---|---|
| `chexpert.py` | Builds `artifacts/manifest.parquet`: applies the uncertainty policy (−1 → pos/neg/ignore), makes patient-level stratified train/val/test splits, marks the capped diffusion-training subset. |
| `cache_latents.py` | Precomputes frozen-VAE latents and CLIP token IDs (several prompt variants per image) for the diffusion subset → `artifacts/cache/`. Training then never decodes images. |

### Stage 2 — diffusion fine-tuning
| File | What it does |
|---|---|
| `train_lora.py` | Fine-tunes SD 1.4 on the latent cache. Runs: **A** LoRA U-Net + LoRA CLIP @512, **B** LoRA U-Net @512, **C** full U-Net @256. Exact resume (all randomness seeded from the step), checkpoint pruning with milestones, periodic sample grids. Also exports `RUNS`, `VAL_PROMPTS`, `build_models` that other modules reuse. |

### Stage 3 — generation
| File | What it does |
|---|---|
| `generate.py` | Generates a label-matched synthetic twin for every real train row (default C@22.5k, g=4, DPM++ 25 steps; `--sampler pndm` for the paper sampler check). Per-row seeds; Stage 5's 1% / 10% subsets go first. Resumable; writes `synthetic_manifest.parquet`. |

### Stage 4 — evaluation
| File | What it does |
|---|---|
| `eval_fidelity.py` | 4a: samples 2000 images from the prompts of a fixed real test reference set, then computes FID (XRV + Inception) and intra-prompt MS-SSIM diversity. Also the `real` floor and the untrained `sd` baseline. The PNGs it writes are reused by 4b/4c. → `results/fidelity_*.json` |
| `eval_classify.py` | 4b: do the synthetic images show the pathology they were prompted for? Scores them with our DenseNet judge (primary) and XRV (reference only) and reports per-label and macro AUROC. → `results/classify_*.json` |
| `eval_clip.py` | 4c: BiomedCLIP stand-in for the paper's vision-language metrics: prompt-alignment margin, zero-shot 6-way accuracy, image-image and image-text retrieval precision. → `results/clip_*.json` |
| `qualitative.py` | 4d: paper-style figures (pathologies, modifiers, left/right laterality, composition, guidance sweep), each with a small numeric check by the judge at a 5% FPR threshold. → `artifacts/figures/qual_*`, `results/qual_*.json` |
| `compare_runs.py` | Side-by-side comparison of runs on the same prompts and noise. Cross-run and intra-run SSIM plus comparison sheets. |
| `readme_figures.py` | Builds the README summary charts (training loss, C checkpoint sweep, guidance trade-off) from `results/*.json` and the run logs. → `docs/img/` |
| `diag_c_vs_real.py` | One-off diagnostic for why C@15k scores above real test images on the judge: bootstrap CI, judge fit, label noise, score separation, nearest-neighbour memorisation check. → `results/diag_*.json` |

### Stage 5 — downstream classifier
| File | What it does |
|---|---|
| `train_classifier.py` | DenseNet-121 on the 5 targets (paper settings). `--data real|synth|mixed` × `--frac` × `--seed` for the Stage 5 grid; `--cache` pre-decodes splits to 224 px arrays. The real 100% seed-0 model doubles as the Stage 4b **judge** (`load_judge`, `predict`). → `artifacts/cls/<run>/` |

### Planned, not written yet
`filtering.py` (Stage 3b), `aggregate.py` (tables/figures from `results/*.json`), `utils.py`, `app/gradio_demo.py` (Stage 6). See `docs/PLAN.md`.

## `artifacts/logs/queue_*.sh` — job queues

Each one is a dated batch of commands run in the background. They record how each result was produced.

| File | What it ran |
|---|---|
| `queue_eval_1.sh` | 4a fidelity for B and A @15k. |
| `queue_eval_2.sh` | Trains the judge classifier, the SD / early-checkpoint baselines, then 4b for every set. |
| `queue_guidance.sh` | Guidance-scale sweep (1–7.5) for C@15k, plus A/B at g=7.5. |
| `queue_resume_ckpt.sh` | Finishes the A/B g=7.5 runs that hit OOM, then the C checkpoint sweep (5k–12.5k). |
| `queue_ckpt.sh` | Retry of the C checkpoint sweep at bs 4. |
| `queue_clip.sh` | 4c BiomedCLIP on CPU for the main sets. |
| `queue_extend_C.sh` | Extends C training 15k → 25k, then evaluates 17.5k–25k at g=4 and g=5.5. |
| `queue_generate.sh` | Stage 3: PNDM 2000-image check subset, then the full DPM++ corpus (re-run to resume). |
