"""Stage 5 classifier (and the Stage 4b judge): DenseNet-121 on the 5 target labels.

    python -m roentgen.train_classifier --cache                      # one-off: decode splits to 224px uint8 (CPU)
    python -m roentgen.train_classifier --frac 1.0 --seed 0          # Real-only 100% -> artifacts/cls/real_f1.0_s0/
    python -m roentgen.train_classifier --frac 0.1 --seed 1 --epochs 30 --patience 5
    python -m roentgen.train_classifier --data synth --frac 0.01 --seed 0  # -> artifacts/cls/synth_f0.01_s0/
    python -m roentgen.train_classifier --data mixed --frac 0.01 --seed 0  # real rows + their synthetic twins

Paper settings: ImageNet-pretrained DenseNet-121, 224 px, BCE-with-logits, AdamW lr 1e-4 wd 0.05,
no augmentation. Early stopping on val macro-AUROC; the best checkpoint is scored on the real test
split. Images are pre-decoded once to artifacts/cache/cls224_<split>.npy (uint8, N x 224 x 224) so
every one of the Stage 5 runs reads from RAM instead of decoding JPEGs.

--data picks the training set for the same seeded --frac draw of real train rows: real (the rows),
synth (their Stage 3 synthetic twins from --synth, same labels) or mixed (both, i.e. equal real +
synthetic). Twin PNGs are decoded to 224 px at start-up; a missing twin is an error.

Writes <out>/best.pt, <out>/log.jsonl and <out>/result.json (val/test AUROC per label + macro).
`load_judge()` + `predict()` are what eval_classify uses to score synthetic images.
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from roentgen.config import ARTIFACTS, CACHE_DIR, SEED, TARGETS, find_chexpert_dir
from roentgen.data import IMAGENET_MEAN, IMAGENET_STD, load_square, read_manifest

SIZE = 224
CLS_DIR = ARTIFACTS / "cls"
SYNTH = ARTIFACTS / "synth" / "C_step22500_g4.0_dpm25"
JUDGE = CLS_DIR / "real_f1.0_s0" / "best.pt"


# ----------------------------------------------------------------------------- cache
def _decode(path):
    return np.asarray(load_square(path, SIZE).convert("L"))


def cache_path(split):
    return CACHE_DIR / f"cls{SIZE}_{split}.npy"


def build_cache():
    root = find_chexpert_dir()
    for split in ("train", "val", "test"):
        out = cache_path(split)
        if out.exists():
            print("exists", out)
            continue
        df = read_manifest(split)
        t = time.time()
        with Pool(24) as pool:
            imgs = np.stack(pool.map(_decode, [root / p for p in df.rel_path], chunksize=256))
        np.save(out, imgs)
        print(f"{split}: {imgs.shape} in {time.time() - t:.0f}s -> {out}", flush=True)


def _decode_synth(path):
    return np.asarray(Image.open(path).convert("L").resize((SIZE, SIZE), Image.BICUBIC))


def load_twins(synth_dir, idx):
    """Synthetic twins of train rows idx -> (N,224,224) uint8, in idx order."""
    paths = [synth_dir / "png" / f"{i:06d}.png" for i in idx]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(paths)} synthetic twins missing under {synth_dir}, e.g. {missing[0]}")
    with Pool(24) as pool:
        return np.stack(pool.map(_decode_synth, paths, chunksize=256))


def load_split(split):
    return np.load(cache_path(split), mmap_mode="r"), read_manifest(split)[TARGETS].values.astype(np.float32)


# ----------------------------------------------------------------------------- model
def build_model(pretrained=True):
    from torchvision.models import DenseNet121_Weights, densenet121
    m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier = torch.nn.Linear(m.classifier.in_features, len(TARGETS))
    return m


def to_input(u8):
    """(B,H,W) uint8 tensor on device -> ImageNet-normalised (B,3,224,224) float."""
    x = u8.float().div(255).unsqueeze(1)
    if x.shape[-1] != SIZE:
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    x = x.expand(-1, 3, -1, -1)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def predict(model, imgs, bs=256, device="cuda"):
    """imgs: (N,H,W) uint8 numpy -> (N,5) sigmoid probabilities."""
    model.eval()
    out = []
    for b in range(0, len(imgs), bs):
        x = to_input(torch.from_numpy(np.ascontiguousarray(imgs[b:b + bs])).to(device))
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(torch.sigmoid(model(x.contiguous(memory_format=torch.channels_last)).float()).cpu())
    return torch.cat(out).numpy()


def load_judge(path=JUDGE, device="cuda"):
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).to(memory_format=torch.channels_last).eval()


def aurocs(y, p):
    per = {t: float(roc_auc_score(y[:, k], p[:, k])) for k, t in enumerate(TARGETS) if 0 < y[:, k].sum() < len(y)}
    return per, float(np.mean(list(per.values())))


# ----------------------------------------------------------------------------- train
def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(SEED + a.seed)
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    xva, yva = xva[:a.val_n], yva[:a.val_n]
    idx = np.arange(len(xtr)) if a.frac >= 1 else np.sort(rng.choice(len(xtr), int(a.frac * len(xtr)), replace=False))
    if a.data == "real":
        xtr, ytr = np.asarray(xtr[idx]), ytr[idx]                   # materialise subset in RAM
    elif a.data == "synth":
        xtr, ytr = load_twins(a.synth, idx), ytr[idx]               # twins share their real row's labels
    else:
        xtr, ytr = np.concatenate([xtr[idx], load_twins(a.synth, idx)]), np.concatenate([ytr[idx], ytr[idx]])
    out = a.out or CLS_DIR / f"{a.data}_f{a.frac}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    model = build_model().to(a.device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    # Small subsets get proportionally more epochs' worth of steps per "epoch" check: keep >= a.min_steps
    steps_per_epoch = max(len(xtr) // a.bs, a.min_steps)
    best, bad, log = -1.0, 0, open(out / "log.jsonl", "w")
    print(f"train n={len(xtr)} steps/epoch={steps_per_epoch} -> {out}", flush=True)
    for ep in range(a.epochs):
        model.train()
        t, tot = time.time(), 0.0
        order = np.concatenate([rng.permutation(len(xtr)) for _ in range(-(-steps_per_epoch * a.bs // len(xtr)))])
        for s in range(steps_per_epoch):
            b = np.sort(order[s * a.bs:(s + 1) * a.bs])
            x = to_input(torch.from_numpy(xtr[b]).to(a.device, non_blocking=True)).contiguous(memory_format=torch.channels_last)
            y = torch.from_numpy(ytr[b]).to(a.device)
            with torch.autocast(a.device, dtype=torch.bfloat16, enabled=a.device == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        per, macro = aurocs(yva, predict(model, xva, device=a.device))
        rec = {"epoch": ep, "loss": tot / steps_per_epoch, "val_macro": macro, "val": per, "s": time.time() - t}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"ep {ep:2d} loss {rec['loss']:.4f} val_macro {macro:.4f} ({rec['s']:.0f}s)", flush=True)
        if macro > best:
            best, bad = macro, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break

    model.load_state_dict(torch.load(out / "best.pt", map_location=a.device))
    xte, yte = load_split("test")
    xte, yte = xte[:a.val_n], yte[:a.val_n]
    per, macro = aurocs(yte, predict(model, xte, device=a.device))
    res = {"args": {k: str(v) for k, v in vars(a).items()}, "n_train": len(xtr), "best_val_macro": best,
           "test": per, "test_macro": macro}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"test": per, "test_macro": macro}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="build the 224px uint8 split caches and exit")
    ap.add_argument("--data", choices=["real", "synth", "mixed"], default="real")
    ap.add_argument("--synth", type=lambda s: __import__("pathlib").Path(s), default=SYNTH,
                    help="Stage 3 output folder holding png/<train_idx>.png")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--min_steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val_n", type=int, default=None, help="subsample val/test (smoke tests only)")
    ap.add_argument("--out", type=lambda s: __import__("pathlib").Path(s), default=None)
    a = ap.parse_args()
    if a.cache:
        build_cache()
    else:
        train(a)


if __name__ == "__main__":
    main()
