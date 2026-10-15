"""Diagnostic: why does the judge score C@15k synthetic images (macro AUROC 0.85) above real test
images (0.76)? Tests, on the same 2000 reference rows used by eval_classify:

  1. paired bootstrap CI of macro AUROC (C - real)
  2. judge fit level: AUROC on 2000 real *train* images (the judge and C share the train split)
  3. label noise: AUROC on rows without any uncertain (-1) target label
  4. score separation: mean judge prob on positives vs negatives
  5. memorisation: XRV-feature cosine nearest neighbour in 20k train images, C vs real test

    python -m roentgen.diag_c_vs_real [--tag C_step15000]
Writes results/diag_<tag>_vs_real.json and artifacts/eval/<tag>/nn_pairs.png.
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from roentgen.config import SEED, TARGETS
from roentgen.data import read_manifest
from roentgen.eval_fidelity import EVAL_DIR, RESULTS, Extractors, load_real, reference_rows, resize
from roentgen.train_classifier import JUDGE, load_judge, load_split, predict


def macro(y, p, rows=None):
    rows = np.arange(len(y)) if rows is None else rows
    per = {t: float(roc_auc_score(y[rows, k], p[rows, k])) for k, t in enumerate(TARGETS)
           if 0 < y[rows, k].sum() < len(rows)}
    return per, float(np.mean(list(per.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="C_step15000")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n_train_nn", type=int, default=20000)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    res = {"tag": a.tag}

    rows = reference_rows("test", a.n)
    raw = rows[TARGETS].values
    y = (np.nan_to_num(raw) == 1).astype(int)
    real = load_real(rows, 256)
    d = EVAL_DIR / a.tag / "fid"
    syn = resize(np.stack([np.asarray(Image.open(d / f"{i:05d}.png")) for i in range(a.n)]), 256)

    judge = load_judge(JUDGE)
    p_real, p_syn = predict(judge, real), predict(judge, syn)

    # 1. paired bootstrap
    _, m_real = macro(y, p_real)
    _, m_syn = macro(y, p_syn)
    diffs = []
    for _ in range(1000):
        b = rng.integers(0, a.n, a.n)
        try:
            diffs.append(macro(y, p_syn, b)[1] - macro(y, p_real, b)[1])
        except ValueError:
            pass
    res["bootstrap"] = {"real": m_real, "syn": m_syn, "diff": m_syn - m_real,
                        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}

    # 2. judge on real train images
    xtr, ytr_raw = load_split("train")
    idx = np.sort(rng.choice(len(xtr), a.n, replace=False))
    p_tr = predict(judge, np.asarray(xtr[idx]))
    res["judge_on_train"] = macro((ytr_raw[idx] == 1).astype(int), p_tr)

    # 3. label noise: rows with no uncertain target / single-finding rows
    certain = np.where(~(raw == -1).any(1))[0]
    res["no_uncertain"] = {"n": int(len(certain)), "real": macro(y, p_real, certain), "syn": macro(y, p_syn, certain)}
    res["n_rows_with_uncertain"] = int(a.n - len(certain))

    # 4. separation: mean prob on positives / negatives
    sep = {}
    for k, t in enumerate(TARGETS):
        pos, neg = y[:, k] == 1, y[:, k] == 0
        sep[t] = {"real_pos": float(p_real[pos, k].mean()), "real_neg": float(p_real[neg, k].mean()),
                  "syn_pos": float(p_syn[pos, k].mean()), "syn_neg": float(p_syn[neg, k].mean())}
    res["separation"] = sep
    # where does the gain come from? AUROC split by positives-vs-"normal"-prompt negatives vs generic negatives
    normal = (np.nan_to_num(rows["No Finding"].values) == 1)
    res["frac_rows_normal_prompt"] = float(normal.mean())
    res["frac_rows_generic_prompt"] = float(((y.sum(1) == 0) & ~normal).mean())

    # 5. memorisation via XRV features
    del judge
    torch.cuda.empty_cache()
    ext = Extractors()
    tr_idx = np.sort(rng.choice(len(xtr), a.n_train_nn, replace=False))
    f_tr = torch.from_numpy(ext(np.asarray(xtr[tr_idx]))["xrv"]).float().cuda()
    f_tr = torch.nn.functional.normalize(f_tr, dim=1)
    out = {}
    nn_idx = {}
    for name, im in (("real_test", real), ("syn", syn)):
        f = torch.nn.functional.normalize(torch.from_numpy(ext(im)["xrv"]).float().cuda(), dim=1)
        sim, j = (f @ f_tr.T).max(1)
        sim = sim.cpu().numpy()
        nn_idx[name] = (sim, j.cpu().numpy())
        out[name] = {"mean": float(sim.mean()), "p50": float(np.median(sim)), "p99": float(np.percentile(sim, 99)),
                     "frac_gt_0.95": float((sim > 0.95).mean()), "frac_gt_0.98": float((sim > 0.98).mean())}
    res["nn_cosine_to_train"] = out

    # figure: 8 most train-similar synthetic images next to their train NN
    sim, j = nn_idx["syn"]
    top = np.argsort(-sim)[:8]
    tiles = [np.concatenate([resize(syn[i:i + 1], 224)[0], np.asarray(xtr[tr_idx[j[i]]])], 0) for i in top]
    fig = Path_out = EVAL_DIR / a.tag / "nn_pairs.png"
    Image.fromarray(np.concatenate(tiles, 1)).save(fig)
    res["nn_figure"] = str(Path_out)
    res["nn_top_sims"] = [float(sim[i]) for i in top]

    RESULTS.mkdir(exist_ok=True)
    o = RESULTS / f"diag_{a.tag}_vs_real.json"
    o.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
