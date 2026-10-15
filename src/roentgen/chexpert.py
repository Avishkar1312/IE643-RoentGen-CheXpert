"""Stage 1: manifest build, uncertainty policy, patient-level stratified splits.

    python -m roentgen.chexpert --build [--uncertain positive|negative|ignore]
"""
import argparse
import json

import numpy as np
import pandas as pd

from roentgen.config import (
    ARTIFACTS, COMBO_CAP, LABELS, MANIFEST, MANIFEST_META, NORMAL, SEED,
    SPLIT_FRACTIONS, TARGETS, find_chexpert_dir,
)


def apply_uncertainty(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """CheXpert: 1 pos, 0 neg, -1 uncertain, blank unmentioned (-> 0).
    'ignore' keeps -1 as NaN so downstream losses can mask it."""
    out = df.copy()
    for c in LABELS:
        col = out[c].fillna(0)
        if policy == "positive":
            col = col.replace(-1, 1)
        elif policy == "negative":
            col = col.replace(-1, 0)
        elif policy == "ignore":
            col = col.replace(-1, np.nan)
        else:
            raise ValueError(policy)
        out[c] = col
    return out


def raw_counts(df: pd.DataFrame, policy: str) -> dict:
    lab = apply_uncertainty(df, policy)
    return {c: int((lab[c] == 1).sum()) for c in LABELS}


def load_frontal(root, csv: str, policy: str) -> pd.DataFrame:
    df = pd.read_csv(root / csv)
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    df = apply_uncertainty(df, policy)
    # Path: CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
    parts = df["Path"].str.split("/")
    df["patient_id"] = parts.str[2]
    df["study_id"] = parts.str[2] + "/" + parts.str[3]
    df["rel_path"] = parts.str[1:].str.join("/")       # relative to the CheXpert dir
    df["view"] = df["AP/PA"].fillna("AP").where(df["AP/PA"].isin(["AP", "PA"]), "AP")
    return df


def patient_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """Iterative stratification over *patients* (labels = any-positive across their
    frontal images), so no patient appears in two splits."""
    from skmultilearn.model_selection import IterativeStratification

    pat = (df[LABELS].fillna(0) == 1).groupby(df["patient_id"]).max().astype(int)
    np.random.seed(seed)
    order = np.random.permutation(len(pat))
    pat = pat.iloc[order]
    X = np.arange(len(pat)).reshape(-1, 1)
    y = pat.values
    fr = SPLIT_FRACTIONS
    strat = IterativeStratification(
        n_splits=3, order=2,
        sample_distribution_per_fold=[fr["test"], fr["val"], fr["train"]],
    )
    assign = np.empty(len(pat), dtype=object)
    for name, (_, fold_idx) in zip(["test", "val", "train"], strat.split(X, y)):
        assign[fold_idx] = name
    pat_split = pd.Series(assign, index=pat.index)
    return df["patient_id"].map(pat_split)


def diffusion_subset(df: pd.DataFrame, cap: int, seed: int) -> pd.Series:
    """Flag train-split images used for diffusion fine-tuning: cap each label combo."""
    tr = df[df["split"] == "train"]
    combo = (tr[LABELS].fillna(0) == 1).astype(int).astype(str).agg("".join, axis=1)
    keep = (tr.groupby(combo, group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), cap), random_state=seed)).index)
    return df.index.isin(keep)


def build(policy: str, cap: int, seed: int) -> pd.DataFrame:
    root = find_chexpert_dir()
    all_train = pd.read_csv(root / "train.csv")
    train = load_frontal(root, "train.csv", policy)
    train["split"] = patient_split(train, seed)
    train["diffusion"] = diffusion_subset(train, cap, seed)

    valid = load_frontal(root, "valid.csv", policy)   # 234 radiologist-labelled images
    valid["split"] = "chexpert_valid"
    valid["diffusion"] = False

    cols = ["rel_path", "patient_id", "study_id", "view", "Sex", "Age", "split", "diffusion"] + LABELS
    man = pd.concat([train, valid], ignore_index=True)[cols]

    # --- checks ---
    sets = {s: set(man.loc[man.split == s, "patient_id"]) for s in ["train", "val", "test"]}
    for a in sets:
        for b in sets:
            if a < b:
                assert not (sets[a] & sets[b]), f"patient leak {a}/{b}"

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    man.to_parquet(MANIFEST, index=False)

    pos = lambda d: {c: int((d[c] == 1).sum()) for c in LABELS}
    meta = {
        "uncertain_policy": policy, "seed": seed, "combo_cap": cap,
        "chexpert_dir": str(root),
        "raw_all_views_positives": raw_counts(all_train, policy),
        "n_images": {s: int((man.split == s).sum()) for s in man.split.unique()},
        "n_patients": {s: len(v) for s, v in sets.items()},
        "n_diffusion": int(man.diffusion.sum()),
        "positives": {s: pos(man[man.split == s]) for s in man.split.unique()},
        "positives_diffusion": pos(man[man.diffusion]),
        "no_target_not_normal": int((((man[TARGETS] == 1).sum(1) == 0) & (man[NORMAL] != 1)).sum()),
    }
    MANIFEST_META.write_text(json.dumps(meta, indent=2))
    return man, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--uncertain", default="positive", choices=["positive", "negative", "ignore"])
    ap.add_argument("--cap", type=int, default=COMBO_CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.build:
        _, meta = build(a.uncertain, a.cap, a.seed)
        print(json.dumps(meta, indent=2))
