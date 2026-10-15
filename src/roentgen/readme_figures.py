"""Summary charts for the README, built from results/*.json and the training logs.
    python -m roentgen.readme_figures
Writes docs/img/{training_loss,checkpoint_sweep,guidance_tradeoff}.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roentgen.config import ARTIFACTS, ROOT

RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "img"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
LABEL = {"A": "A  LoRA U-Net + LoRA CLIP @512", "B": "B  LoRA U-Net @512", "C": "C  full U-Net @256"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def metrics(run, step, g=None):
    suffix = f"_g{g}" if g else ""
    f = json.load(open(RESULTS / f"fidelity_{run}_step{step}{suffix}_256px.json"))
    c = json.load(open(RESULTS / f"classify_{run}_step{step}{suffix}_256px.json"))
    return f["fid_xrv"], c["own"]["macro_auroc"]


def training_loss():
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for run in "ABC":
        rows = [json.loads(l) for l in open(ARTIFACTS / "runs" / run / "log.jsonl")]
        rows = [r for r in rows if "loss" in r]
        step = np.array([r["step"] for r in rows])
        loss = np.array([r["loss"] for r in rows])
        k = 50
        smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
        ax.plot(step[k - 1:], smooth, color=COLORS[run], lw=2, label=LABEL[run])
    ax.set_xlabel("optimizer step", color=MUTED)
    ax.set_ylabel("ε-prediction MSE (moving avg)", color=MUTED)
    ax.set_title("Stage 2 training loss (C is at 256 px, so its scale differs from A/B at 512 px)",
                 color=INK, loc="left", fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "training_loss.png")


def checkpoint_sweep():
    steps = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000]
    fid, auc = zip(*(metrics("C", s) for s in steps))
    real_auc = json.load(open(RESULTS / "classify_real_256px.json"))["own"]["macro_auroc"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    for ax, y, title in ((a1, fid, "FID (XRV features)  ↓"), (a2, auc, "label macro-AUROC, own judge  ↑")):
        ax.plot(steps, y, color=COLORS["C"], lw=2, marker="o", ms=6)
        i = int(np.argmin(y) if ax is a1 else steps.index(22500))
        ax.plot(steps[i], y[i], marker="o", ms=10, mfc="none", mec=INK, mew=1.5)
        ax.annotate("22.5k (chosen)", (steps[i], y[i]), xytext=(-95, 4) if ax is a1 else (-95, -16),
                    textcoords="offset points", fontsize=9, color=INK)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.set_xlabel("run C training step", color=MUTED)
        ax.set_xticks(steps[1::2], [f"{s // 1000}k" if s % 1000 == 0 else f"{s / 1000}k" for s in steps[1::2]])
        style(ax)
    a2.axhline(real_auc, color=MUTED, lw=1.2, ls="--")
    a2.text(steps[0], real_auc + 0.006, f"real test images ({real_auc:.2f})", fontsize=8.5, color=MUTED)
    fig.suptitle("Run C checkpoint sweep (256 px, n=2000, guidance 4)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "checkpoint_sweep.png")


def guidance_tradeoff():
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    gs = {"1": 1.0, "2": 2.0, "3": 3.0, None: 4.0, "5.5": 5.5, "7.5": 7.5}
    pts = [metrics("C", 15000, g) for g in gs]
    x, y = zip(*pts)
    ax.plot(x, y, color=COLORS["C"], lw=2, marker="o", ms=7, label=LABEL["C"] + "  @15k")
    for (xi, yi), g in zip(pts, gs.values()):
        ax.annotate(f"g={g:g}", (xi, yi), xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=MUTED)
    for run in "AB":
        p = [metrics(run, 15000), metrics(run, 15000, "7.5")]
        ax.plot(*zip(*p), color=COLORS[run], lw=2, marker="o", ms=7, label=LABEL[run] + "  @15k")
        for (xi, yi), g in zip(p, (4, 7.5)):
            ax.annotate(f"g={g:g}", (xi, yi), xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=MUTED)
    real_auc = json.load(open(RESULTS / "classify_real_256px.json"))["own"]["macro_auroc"]
    ax.axhline(real_auc, color=MUTED, lw=1.2, ls="--")
    ax.text(4.2, real_auc + 0.006, f"real test images ({real_auc:.2f})", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xlabel("FID (XRV features)  ↓  better to the left", color=MUTED)
    ax.set_ylabel("label macro-AUROC  ↑", color=MUTED)
    ax.set_title("Fidelity vs. prompt faithfulness across guidance scales", color=INK, loc="left", fontsize=12)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "guidance_tradeoff.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    training_loss()
    checkpoint_sweep()
    guidance_tradeoff()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
