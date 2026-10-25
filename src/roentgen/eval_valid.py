"""Score every Stage 5 classifier on the official CheXpert valid set (radiologist labels).

    python -m roentgen.eval_valid

valid.csv is the small clean secondary test set kept aside in Stage 1 (split "chexpert_valid",
202 frontal images). It is tiny for our five labels: Enlarged Cardiomediastinum has 105 positives
and Pneumonia 8, while Pleural Other and Lung Lesion have 1 each and Fracture has none. Only labels
with >= MIN_POS positives enter the reported macro, so here it is a 2-label macro (EC + Pneumonia).

Reads artifacts/cls/<run>/best.pt and writes results/chexpert_valid.json (per run: per-label AUROC
and macro), plus mean ± std over seeds per (data, frac) in the same layout as stage5_table.
The real 1% checkpoints were lost in the 2026-10-24 folder deletion (only result.json survived), so
they were retrained with the same command into artifacts/cls_rerun/; a run found in cls/ wins.
"""
import json
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from roentgen.config import ARTIFACTS, TARGETS, find_chexpert_dir
from roentgen.data import read_manifest
from roentgen.eval_fidelity import RESULTS
from roentgen.train_classifier import CLS_DIR, SIZE, _decode, load_judge, predict

MIN_POS = 5


def main():
    rows = read_manifest("chexpert_valid")
    root = find_chexpert_dir()
    imgs = np.stack([_decode(root / p) for p in rows.rel_path])
    y = (rows[TARGETS].fillna(0).values == 1).astype(int)
    npos = dict(zip(TARGETS, map(int, y.sum(0))))
    used = [t for t in TARGETS if npos[t] >= MIN_POS]
    print(f"chexpert_valid: {len(rows)} images at {SIZE}px, positives {npos}; macro over {used}")

    runs, groups = {}, defaultdict(list)
    cks = {ck.parent.name: ck for d in (ARTIFACTS / "cls_rerun", CLS_DIR) for ck in d.glob("*/best.pt")}
    for name, ck in sorted(cks.items()):
        m = re.fullmatch(r"(real|synth|mixed)_f([\d.]+)_s(\d+)", name)
        if not m:
            continue
        p = predict(load_judge(ck), imgs)
        per = {t: float(roc_auc_score(y[:, k], p[:, k])) for k, t in enumerate(TARGETS) if t in used}
        runs[name] = {"checkpoint": str(ck.relative_to(ARTIFACTS)), "per_label": per, "macro": float(np.mean(list(per.values())))}
        groups[f"{m[1]}_f{m[2]}"].append(runs[name]["macro"])
        print(f"{name:18s} macro {runs[name]['macro']:.3f}  " + "  ".join(f"{t[:12]} {v:.3f}" for t, v in per.items()))

    cells = {k: {"n_seeds": len(v), "macro_mean": float(np.mean(v)),
                 "macro_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0} for k, v in groups.items()}
    out = {"n_images": len(rows), "positives": npos, "labels_in_macro": used, "min_pos": MIN_POS,
           "runs": runs, "cells": cells}
    (RESULTS / "chexpert_valid.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {RESULTS / 'chexpert_valid.json'}")


if __name__ == "__main__":
    main()
