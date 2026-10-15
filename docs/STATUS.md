# Status — 2026-10-09

## Done
- Stage 1 (manifest, splits, caches) and Stage 2 (runs A, B to 15k steps; C to 25k).
- **Primary for Stage 3: C @ 22.5k, guidance 4.0** (decided 2026-10-09; see extension table below).
- Fidelity @256px, n=2000: real-vs-real FID_XRV 0.11 / Inception 11.1; C@15k 0.69 / 31.2, MS-SSIM 0.22.
- Classifier image cache: artifacts/cache/cls224_{train,val,test}.npy.

## Findings so far
1. **A and B largely ignore the prompt**: on the compare sheets, each seed gives nearly the same image
   across all six prompts. C responds to the prompt. The text-encoder LoRA in A did train (lora_B norm 5.4),
   so it is not a loading bug.
2. All three loss curves are still falling at 15k steps, so none of the runs has converged.
3. The pretrained TorchXRayVision classifier can't judge our five labels: its macro AUROC on *real* test images
   is 0.55, vs 0.85–0.93 on effusion/cardiomegaly/edema with the same pipeline. The uncertainty policy
   (pos/ignore/neg) doesn't change this. -> Our own 5-label DenseNet is the judge (eval_classify "own").

## Stage 4 comparison table (256px, n=2000 test-row prompts, 25 steps, guidance 4.0)
| Set | FID_XRV ↓ | FID_Inception ↓ | MS-SSIM (lower = more diverse) | Label macro-AUROC, own judge ↑ | XRV macro-AUROC |
|---|---|---|---|---|---|
| real (test) | 0.11 | 11.1 | 0.34 (random real pairs) | 0.756 | 0.553 |
| SD 1.4 baseline | 19.35 | 193.1 | 0.088 | 0.514 | 0.476 |
| A @1k | 6.01 | 55.2 | 0.144 | 0.540 | 0.505 |
| A @15k | 3.16 | 46.3 | 0.128 | 0.661 | 0.528 |
| B @1k | 3.27 | 42.3 | 0.241 | 0.511 | 0.519 |
| B @15k | 3.90 | 41.5 | 0.139 | 0.652 | 0.512 |
| C @2.5k | 0.82 | 30.8 | 0.232 | 0.745 | 0.459 |
| **C @15k** | **0.69** | **31.2** | 0.220 | **0.847** | 0.529 |

Source: results/fidelity_*_256px.json, results/classify_*_256px.json. C is best on every axis; A/B condition
weakly. C@15k scores *above* real (0.85 vs 0.76) — see diagnosis below.

### Why C@15k scores above real (diagnosis 2026-10-06; results/diag_C_step15000_vs_real.json, src/roentgen/diag_c_vs_real.py)
- Not noise: paired bootstrap diff +0.092, 95% CI [0.067, 0.117].
- Not uncertain labels: the 2000 test rows have no -1 target labels.
- Not judge overfit alone: judge on 2000 real *train* images = 0.787 (test 0.756) — still below 0.85.
- Not memorisation: XRV cosine NN-to-train (20k train imgs) is *lower* for C (mean 0.880, max 0.978) than for
  real test (0.895); nearest pairs (artifacts/eval/C_step15000/nn_pairs.png) are clearly different patients.
- Cause: **label-image consistency**. A synthetic image is rendered from its label text only, so it contains the
  learned prototype of the label and nothing else. Real images carry label noise (labeler-missed / subtle
  findings, findings without visible signs) and unlabelled confounders. The gain is mostly *cleaner negatives*:
  mean judge prob on negatives drops for Pneumonia 0.117→0.072, Lung Lesion 0.063→0.045, Pleural Other
  0.044→0.035, while positives rise for Fracture 0.21→0.32 and Pleural Other 0.13→0.16. 63% of rows get the
  generic "chest x-ray" prompt, which C renders as images without target-like features.
- Implication: real AUROC is a reference, not a ceiling. Synthetic AUROC > real means "label signal is present
  and exaggerated", not "more realistic". Report both, and back it with BiomedCLIP (4c) + qualitative figures (4d).

### Guidance sweep, C@15k (2026-10-06; results/{fidelity,classify}_C_step15000[_g<g>]_256px.json)
| g | FID_XRV ↓ | FID_Inception ↓ | MS-SSIM | own-judge macro-AUROC ↑ | XRV macro-AUROC |
|---|---|---|---|---|---|
| 1.0 | 1.00 | 29.4 | 0.223 | 0.624 | 0.517 |
| 2.0 | 0.78 | 29.3 | 0.226 | 0.720 | 0.523 |
| 3.0 | 0.69 | 30.1 | 0.223 | 0.793 | 0.522 |
| 4.0 | **0.69** | 31.2 | 0.220 | 0.847 | 0.529 |
| 5.5 | 0.77 | 32.9 | 0.221 | 0.903 | 0.546 |
| 7.5 | 0.98 | 35.4 | 0.222 | **0.940** | 0.564 |

- Standard CFG trade-off: label AUROC rises steadily with g (0.62→0.94); FID_XRV is U-shaped with its minimum at g≈3–4,
  and Inception FID keeps getting worse as g goes up. Overall MS-SSIM is flat, but per prompt the pleural prompt
  collapses (0.29→0.35) and the fracture prompt spreads out (0.15→0.11) as g increases.
- Real-image AUROC (0.756) falls between g=2 and g=3. g=4 gives the best fidelity; g≈5.5 is a reasonable choice if
  label strength matters more for Stage 5 (FID_XRV +0.08). Pick it together with the checkpoint choice.

### Guidance 7.5 for A/B (2026-10-07): strong CFG doesn't rescue the LoRA runs
| Set | FID_XRV | FID_Inc | MS-SSIM | own AUROC |
|---|---|---|---|---|
| A@15k g4 → g7.5 | 3.16 → 3.08 | 46.3 → 47.8 | 0.128 → 0.127 | 0.661 → 0.738 |
| B@15k g4 → g7.5 | 3.90 → 4.09 | 41.5 → 43.6 | 0.139 → 0.149 | 0.652 → 0.710 |
Even at g=7.5, A and B stay below C@g2 on AUROC (0.72) and far behind on FID. **C is the primary run.**

### Extension C 15k → 25k (2026-10-08; artifacts/logs/queue_extend_C.sh, log artifacts/runs/C/train_15k-25k.log)
Resumed from ckpt-15000 with the same hparams (opt/sched/data order restored). 1.43 s/step, peak 10.1 GB. Ckpts at 17.5k/20k/22.5k/25k.
Source: results/{fidelity,classify}_C_step<s>[_g5.5]_256px.json.
| step | g | FID_XRV ↓ | FID_Inception ↓ | MS-SSIM | own-judge macro-AUROC ↑ |
|---|---|---|---|---|---|
| 15k | 4 | 0.69 | 31.2 | 0.220 | 0.847 |
| 15k | 5.5 | 0.77 | 32.9 | 0.221 | 0.903 |
| 17.5k | 4 | 0.94 | 33.9 | 0.211 | 0.871 |
| 17.5k | 5.5 | 1.15 | 34.3 | 0.218 | **0.922** |
| 20k | 4 | 0.88 | 31.3 | 0.177 | 0.867 |
| 20k | 5.5 | 0.92 | 32.4 | 0.181 | 0.913 |
| **22.5k** | **4** | **0.59** | **30.9** | 0.232 | 0.875 |
| 22.5k | 5.5 | 0.73 | 34.1 | 0.234 | 0.915 |
| 25k | 4 | 0.95 | 39.5 | 0.202 | 0.855 |
| 25k | 5.5 | 1.04 | 40.8 | 0.206 | 0.896 |
- **22.5k is the best checkpoint.** At g=4 it has the lowest FID on both feature sets, and its AUROC (0.875) is above 15k (0.847).
  25k regresses (Inception FID jumps to about 40), so we stopped there.
- FID_XRV moves by up to about 0.35 between neighbouring ckpts (single sampling seed), so the order of 17.5k vs 20k is noise. 22.5k
  agrees across both FIDs and AUROC.
- g=5.5 adds about 0.04 AUROC for +0.14 FID_XRV. **We chose g=4 (fidelity) for Stage 3.** Enlarged cardiomediastinum is still the weakest
  label (0.72 at g=4).

### Checkpoint sweep, C @ g=4 (2026-10-07; results/{fidelity,classify}_C_step<s>_256px.json)
| step | FID_XRV ↓ | FID_Inception ↓ | MS-SSIM | own-judge macro-AUROC ↑ | XRV macro-AUROC |
|---|---|---|---|---|---|
| 2.5k | 0.82 | **30.8** | 0.232 | 0.745 | 0.459 |
| 5k | 0.88 | 35.3 | 0.182 | 0.786 | 0.545 |
| 7.5k | 1.25 | 35.7 | 0.235 | 0.818 | 0.546 |
| 10k | 1.02 | 40.0 | 0.239 | **0.849** | 0.533 |
| 12.5k | 0.70 | 34.0 | 0.193 | 0.838 | 0.542 |
| 15k | **0.69** | 31.2 | 0.220 | 0.847 | 0.529 |
- AUROC rises to about 0.85 by 10k, then plateaus. Fidelity dips mid-training (worst at 7.5k–10k) and recovers by 12.5k–15k.
- **15k was the best checkpoint at this point** (superseded by 22.5k, see the extension table above). It has the best FID_XRV, near-best Inception FID, and AUROC level with the peak. The loss was still
  falling at 15k, so extending C past 15k is the one open improvement.

### 4c BiomedCLIP, real-image reference (results/clip_real_256px.json)
BiomedCLIP barely reads CXR labels, even on real images: zero-shot 6-class accuracy is 0.34 (chance 0.17),
image→image P@10 is 0.22 (chance 0.17), image→text P@1 is 0.32 (chance 0.16). Treat 4c as a weak secondary
signal. It can show "synthetic ≈ real" but can't separate runs finely. The own-judge AUROC remains the primary label metric.

| Set | prompt margin (own − mismatched) | own > mismatched | zero-shot acc | img→img P@10 | img→text P@1 |
|---|---|---|---|---|---|
| chance | 0 | 0.50 | 0.17 | 0.17 | 0.16 |
| real (test) | 0.018 | 0.64 | 0.34 | 0.22 | 0.32 |
| SD 1.4 | 0.001 | 0.49 | 0.22 | 0.17 | 0.19 |
| A@15k | 0.008 | 0.56 | 0.19 | 0.18 | 0.16 |
| B@15k | 0.011 | 0.60 | 0.22 | 0.20 | 0.18 |
| C@2.5k | 0.022 | 0.66 | 0.38 | 0.22 | 0.34 |
| C@15k | 0.028 | 0.72 | 0.42 | 0.27 | 0.37 |
| C@15k g7.5 | 0.041 | 0.81 | 0.55 | 0.32 | 0.53 |
Same ordering as the own judge (SD ≈ A < B < real ≲ C, and rising with g), so it independently confirms it. C again exceeds real
(cleaner label prototypes, as in the 4b diagnosis). The retrieval candidates are real images, so img→img above real means the label
features are on a real-image manifold.

### 4d qualitative, C@15k g=4 (2026-10-07; src/roentgen/qualitative.py -> artifacts/figures/qual_C_step15000/, results/qual_C_step15000.json)
16 seeds per prompt. A finding is "called" when the own-judge prob exceeds its 5%-FPR threshold on real test negatives.
- **Fig 2 (one prompt per pathology): works.** Called rate for synthetic vs real positives: pneumonia 0.13 vs 0.21, pleural 0.38 vs 0.29,
  lung lesion 0.50 vs 0.32, fracture 0.44 vs 0.33, enlarged CM 0.13 vs 0.16. The normal prompt has a 3.7% any-finding call rate (< 5% FPR).
  Enlarged cardiomediastinum is the weakest label, even with 11k training rows.
- **Fig 3a (unseen modifier prompts):** the finding still comes through (called 0.13–0.50), but size words have no visible effect.
- **Fig 3b (laterality): no effect.** Left vs right from the same noise gives near-identical images. The signed asymmetry S is within
  ±1 SE of 0 for all four findings, and 44–50% of seeds are on the "correct" side (chance level). This is expected: our templated prompts never contain
  laterality, unlike the paper's reports. It is a data limitation that more training can't fix. Report it as such.
- **Fig 4 (composition): weak.** "A and B" usually looks like a diluted mix. Both probs are above the normal prompt, but each is below
  its single-finding value (e.g. lesion+fracture 0.19/0.19 vs 0.30/0.29 single), and both findings are called in only 0–6% of seeds.
  11% (5452) of diffusion-train rows have ≥2 targets.
- Layout is set mostly by the seed and the prompt edits local detail. A few seeds (e.g. s1) give off-distribution, lateral-like films for every prompt.
- Caveat: n=16 per prompt, so each rate has SE ≈ 0.12. Treat these as qualitative support, not a metric.

## Before Stage 3 (decided: finish every comparison and improvement first)
- [x] Review queue 1+2: FID + MS-SSIM + label-AUROC table (above)
- [x] Guidance-scale sweep (C full; A/B @7.5)
- [x] Checkpoint sweep (FID/AUROC vs step): C@22.5k is best after the extension (tables above)
- [x] ~~Native-512 FID for A/B~~ skipped: judged moot, because A/B are out of the running on every 256px metric
- [x] 4c BiomedCLIP alignment/retrieval (table above)
- [x] 4d qualitative figures (above)
- [x] Training improvements: extended C to 25k (best at 22.5k). LoRA rank/TE lr not pursued, since C is primary
- [x] Choose the primary run: **C @ 22.5k, g=4**
- [~] Stage 3 generation (generate.py) with C ckpt-22500, guidance 4.0, launched 2026-10-14 10:56
  (artifacts/logs/queue_generate.sh -> generate.log; check with ./jobstatus.sh; re-run the queue to resume)
  - [x] PNDM-75 sampler check: 2000 rows -> artifacts/synth/C_step22500_g4.0_pndm75 (~30 min)
  - [ ] DPM++-25 twin corpus: all 152,539 train rows -> artifacts/synth/C_step22500_g4.0_dpm25 (~14 h @ ~3 img/s)
        Priority order: union of Stage 5 1% subsets (4,536 rows) -> 10% subsets -> rest, so small-data runs can start early
  - [ ] FID of PNDM-75 vs DPM-25 on the paired 2k (same prompts and seeds)
  - [ ] filtering.py (degenerate images + judge-score rejection rate per pathology)
  - [ ] Per-pathology balanced extra pool (deferred until the twin set is done)
- [x] train_classifier --data real|synth|mixed (2026-10-14): same seeded --frac draw; synth = twins of those rows,
      mixed = rows + twins; out dir artifacts/cls/<data>_f<frac>_s<seed>. Smoke-tested mixed 1% end-to-end.
