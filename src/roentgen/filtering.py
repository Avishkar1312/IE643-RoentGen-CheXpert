"""Stage 3 filtering: flag degenerate and label-inconsistent synthetic twins.

    python -m roentgen.filtering [--synth artifacts/synth/C_step22500_g4.0_dpm25] [--recall 0.9]

Two checks per image, neither of which deletes anything:
  * degenerate: near-constant (pixel std < --min_std), mostly saturated (> --max_sat of pixels at 0 or 255).
  * label-inconsistent: the own judge (artifacts/cls/real_f1.0_s0) scores a *positive* target
    below the threshold that keeps --recall of real val positives for that target. Negatives are
    reported (judge above the matching real-val negative quantile) but never rejected, since the
    real labels count uncertain/unmentioned findings as 0.

Writes <synth>/filter.parquet (train_idx, judge probs, flags, keep) and results/filtering_<name>.json
with per-pathology rejection rates. train_classifier can drop rows with keep == False later.
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from roentgen.config import ROOT, TARGETS
from roentgen.train_classifier import SYNTH, _decode_synth, load_judge, load_split, predict

RESULTS = ROOT / "results"
CHUNK = 8192


def judge_corpus(judge, paths, min_std, max_sat):
    probs, std, sat = [], [], []
    with Pool(24) as pool:
        for s in range(0, len(paths), CHUNK):
            imgs = np.stack(pool.map(_decode_synth, paths[s:s + CHUNK], chunksize=128))
            probs.append(predict(judge, imgs))
            f = imgs.reshape(len(imgs), -1)
            std.append(f.std(1))
            sat.append(((f <= 2) | (f >= 253)).mean(1))
            print(f"  judged {min(s + CHUNK, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(probs), np.concatenate(std), np.concatenate(sat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=lambda s: __import__("pathlib").Path(s), default=SYNTH)
    ap.add_argument("--recall", type=float, default=0.9, help="real-val positive recall kept by each threshold")
    ap.add_argument("--min_std", type=float, default=10.0)
    ap.add_argument("--max_sat", type=float, default=0.5)
    a = ap.parse_args()

    judge = load_judge()
    xv, yv = load_split("val")
    pv = predict(judge, xv)
    lo = {t: float(np.quantile(pv[yv[:, k] == 1, k], 1 - a.recall)) for k, t in enumerate(TARGETS)}
    hi = {t: float(np.quantile(pv[yv[:, k] == 0, k], a.recall)) for k, t in enumerate(TARGETS)}

    man = pd.read_parquet(a.synth / "synthetic_manifest.parquet")
    y = (man[TARGETS].fillna(0).values == 1)
    p, std, sat = judge_corpus(judge, [a.synth / q for q in man.path], a.min_std, a.max_sat)

    out = pd.DataFrame({"train_idx": man.train_idx.values})
    for k, t in enumerate(TARGETS):
        out[f"p_{t}"] = p[:, k]
    out["degenerate"] = (std < a.min_std) | (sat > a.max_sat)
    miss = y & (p < np.array([lo[t] for t in TARGETS]))
    spur = ~y & (p > np.array([hi[t] for t in TARGETS]))
    out["label_miss"] = miss.any(1)
    out["keep"] = ~(out.degenerate | out.label_miss)
    out.to_parquet(a.synth / "filter.parquet", index=False)

    res = {"args": {k: str(v) for k, v in vars(a).items()}, "n": len(out),
           "thresholds_pos": lo, "thresholds_neg": hi,
           "degenerate": int(out.degenerate.sum()), "label_miss": int(out.label_miss.sum()),
           "kept": int(out.keep.sum()), "kept_frac": float(out.keep.mean()),
           "per_target": {t: {"positives": int(y[:, k].sum()),
                              "miss_rate": float(miss[y[:, k], k].mean()),
                              "spurious_rate_neg": float(spur[~y[:, k], k].mean())}
                          for k, t in enumerate(TARGETS)}}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"filtering_{a.synth.name}.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
