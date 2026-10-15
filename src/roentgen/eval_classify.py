"""Stage 4b: do synthetic CXRs show the pathology their prompt asked for?

    python -m roentgen.eval_classify --tag C_step15000 [--n 2000 --size 256]
    python -m roentgen.eval_classify --tag real           # upper bound: same classifier on real test images
    (--device cpu runs it alongside a GPU job; ~2 min for 2000 images)

Reuses the FID set written by eval_fidelity (artifacts/eval/<tag>/fid/00000.png ...): image i
was generated from the prompt of reference row i, so that row's labels are the ground truth.
Two judges, both trained on real CXRs, score every image; we report AUROC per label and macro:
  * "own" (primary): our DenseNet-121 on the 5 targets (train_classifier, Real-only 100%, seed 0).
  * "xrv" (reference): TorchXRayVision densenet121-res224-all. Measured on *real* test images it is
    near chance on these five labels (macro ~0.55; vs 0.85-0.93 on effusion/cardiomegaly/edema), so
    it cannot judge conditioning here; kept only to document that. It has no "Pleural Other" head,
    so that label uses Pleural_Thickening as a proxy and is kept out of the xrv macro.

Writes results/classify_<tag>_<size>px.json.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from roentgen.config import LABELS, ROOT, TARGETS
from roentgen.eval_fidelity import EVAL_DIR, RESULTS, load_real, reference_rows, resize
from roentgen.train_classifier import JUDGE, load_judge, predict

XRV_HEAD = {
    "Pneumonia": "Pneumonia",
    "Lung Lesion": "Lung Lesion",
    "Fracture": "Fracture",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Pleural Other": "Pleural_Thickening",   # proxy only, excluded from macro
}
PROXY = {"Pleural Other"}


@torch.no_grad()
def xrv_scores(imgs, device="cuda", bs=64):
    import torchxrayvision as xrv
    m = xrv.models.DenseNet(weights="densenet121-res224-all", cache_dir=str(ROOT / ".cache" / "xrv")).to(device).eval()
    out = []
    for b in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[b:b + bs]).to(device).unsqueeze(1).float() / 255 * 2048 - 1024
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        out.append(m(x).float().cpu())
    p = torch.cat(out).numpy()
    return {t: p[:, m.pathologies.index(h)] for t, h in XRV_HEAD.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="eval dir name, e.g. C_step15000, sd, or 'real'")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = reference_rows("test", a.n)
    y = (rows[LABELS].fillna(0).values == 1).astype(int)
    if a.tag == "real":
        imgs = load_real(rows, a.size)
    else:
        d = EVAL_DIR / a.tag / "fid"
        paths = [d / f"{i:05d}.png" for i in range(a.n)]
        assert all(p.exists() for p in paths), f"incomplete FID set in {d}; run eval_fidelity first"
        imgs = resize(np.stack([np.asarray(Image.open(p)) for p in paths]), a.size)

    yt = y[:, [LABELS.index(t) for t in TARGETS]]
    res = {"args": vars(a), "positives": dict(zip(TARGETS, map(int, yt.sum(0))))}

    p = predict(load_judge(JUDGE, a.device), imgs, device=a.device)
    own = {t: float(roc_auc_score(yt[:, k], p[:, k])) for k, t in enumerate(TARGETS)}
    res["own"] = {"model": str(JUDGE.relative_to(ROOT)), "auroc": own, "macro_auroc": float(np.mean(list(own.values())))}

    scores = xrv_scores(imgs, a.device)
    xrv = {t: float(roc_auc_score(yt[:, k], scores[t])) for k, t in enumerate(TARGETS)}
    main_labels = [t for t in TARGETS if t not in PROXY]
    res["xrv"] = {"model": "densenet121-res224-all", "auroc": xrv, "proxy_labels": sorted(PROXY),
                  "macro_auroc": float(np.mean([xrv[t] for t in main_labels]))}

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"classify_{a.tag}_{a.size}px.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("own", "xrv", "positives")}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
