"""Stage 3 sampler check: PNDM-75 (paper) vs DPM++-25 (our corpus) on the same 2,000 train rows.

    python -m roentgen.compare_samplers [--size 256]

Both corpora were generated from identical prompts and seeds (generate.py, per-row seed), so the
pairs differ only in the sampler. Reported per sampler:
  * FID_XRV / FID_Inception vs the Stage 4 real-test reference (comparable with results/fidelity_*)
  * FID_XRV / FID_Inception vs the real images of the same 2,000 train rows (label-matched)
  * own-judge AUROC per target (labels of the source rows), as in eval_classify
and between samplers: FID(PNDM, DPM) and the mean paired MS-SSIM (same prompt + seed).

Writes results/sampler_check_<size>px.json.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from roentgen.config import ARTIFACTS, TARGETS
from roentgen.data import read_manifest
from roentgen.eval_fidelity import RESULTS, Extractors, frechet, load_real, real_features, resize
from roentgen.train_classifier import aurocs, load_judge, predict

SYNTH = ARTIFACTS / "synth"
SAMPLERS = {"pndm75": SYNTH / "C_step22500_g4.0_pndm75", "dpm25": SYNTH / "C_step22500_g4.0_dpm25"}


def load_pngs(d, idx):
    return np.stack([np.asarray(Image.open(d / "png" / f"{i:06d}.png").convert("L")) for i in idx])


def paired_msssim(a, b, bs=64):
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as ms
    vals = []
    for s in range(0, len(a), bs):
        x = torch.from_numpy(a[s:s + bs]).float().div(255).unsqueeze(1).cuda()
        y = torch.from_numpy(b[s:s + bs]).float().div(255).unsqueeze(1).cuda()
        vals.append(ms(x, y, data_range=1.0, kernel_size=11, sigma=1.5, reduction="none").cpu())
    return torch.cat(vals).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n_ref", type=int, default=2000, help="real-test reference size (Stage 4 used 2000)")
    a = ap.parse_args()

    idx = np.sort(pd.read_parquet(SAMPLERS["pndm75"] / "synthetic_manifest.parquet").train_idx.values)
    rows = read_manifest("train").iloc[idx].reset_index(drop=True)
    y = (rows[TARGETS].fillna(0).values == 1).astype(int)
    imgs = {k: resize(load_pngs(d, idx), a.size) for k, d in SAMPLERS.items()}

    ext = Extractors()
    ref_test = real_features(ext, a.size, a.n_ref)
    ref_rows = ext(load_real(rows, a.size))
    feats = {k: ext(v) for k, v in imgs.items()}
    del ext
    torch.cuda.empty_cache()

    judge = load_judge()
    res = {"args": vars(a), "n": len(idx), "positives": dict(zip(TARGETS, map(int, y.sum(0))))}
    for k, f in feats.items():
        per, macro = aurocs(y, predict(judge, imgs[k]))
        res[k] = {
            "fid_xrv_vs_test": frechet(f["xrv"], ref_test["xrv"]),
            "fid_inception_vs_test": frechet(f["inception"], ref_test["inception"]),
            "fid_xrv_vs_rows": frechet(f["xrv"], ref_rows["xrv"]),
            "fid_inception_vs_rows": frechet(f["inception"], ref_rows["inception"]),
            "own_auroc": per, "own_macro_auroc": macro,
        }
    per, macro = aurocs(y, predict(judge, resize(load_real(rows, 224), 224)))
    res["real_rows"] = {"own_auroc": per, "own_macro_auroc": macro,
                        "note": "judge was trained on these real rows, so this is optimistic"}
    p, d = feats["pndm75"], feats["dpm25"]
    ss = paired_msssim(imgs["pndm75"], imgs["dpm25"])
    res["between"] = {"fid_xrv": frechet(p["xrv"], d["xrv"]), "fid_inception": frechet(p["inception"], d["inception"]),
                      "paired_msssim_mean": float(ss.mean()), "paired_msssim_sd": float(ss.std())}

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sampler_check_{a.size}px.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
