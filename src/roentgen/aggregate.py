"""Stage 6 report: the paper-style tables and the AUROC-vs-training-set-size figure from results/*.json.

    python -m roentgen.aggregate

Tables (markdown, written to results/tables.md and printed):
  1. fidelity / diversity  (paper Table 1)      fidelity_*_256px.json
  2. classification of synthetic images         classify_*_256px.json + clip_*_256px.json
  3. downstream real / synth / mixed            stage5.json (+ chexpert_valid.json if present)
Figure: docs/img/stage5_auroc_vs_size.png (paper Ext. Data Fig. 2 style). Error bars are 95% CIs over
seeds (t-interval, n=3); the 100% point is a single seed and has none.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from roentgen.config import TARGETS
from roentgen.readme_figures import INK, MUTED, OUT, RESULTS, style

SETS = [  # (label, tag in results/*_<tag>_256px.json)
    ("real test images", "real"),
    ("SD 1.4, no fine-tuning", "sd"),
    ("A: LoRA U-Net + LoRA CLIP @512, 15k", "A_step15000"),
    ("B: LoRA U-Net @512, 15k", "B_step15000"),
    ("C: full U-Net @256, 2.5k", "C_step2500"),
    ("**C: full U-Net @256, 15k (primary)**", "C_step15000"),
    ("C: full U-Net @256, 22.5k", "C_step22500"),
]
DATA = {"real": "#52514e", "synth": "#2a78d6", "mixed": "#1baf7a"}
FRACS = ["0.01", "0.1", "1.0"]


def load(name):
    f = RESULTS / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def table_fidelity():
    rows = ["| Set | FID<sub>XRV</sub> ↓ | FID<sub>Inception</sub> ↓ | MS-SSIM (intra-prompt) |", "|---|---|---|---|"]
    for label, tag in SETS:
        f = load(f"fidelity_{tag}_256px")
        if f is None:
            continue
        ms = f"{f['msssim_real_random']:.3f} (random real pairs)" if tag == "real" else fmt(f["msssim"]["mean"])
        rows.append(f"| {label} | {fmt(f['fid_xrv'], 2)} | {fmt(f['fid_inception'], 1)} | {ms} |")
    return "\n".join(rows)


def table_classify():
    short = [t.split()[0] if t != "Enlarged Cardiomediastinum" else "Enl. CM" for t in TARGETS]
    rows = ["| Set | " + " | ".join(short) + " | macro AUROC (own judge) ↑ | BiomedCLIP zero-shot acc | img→text P@1 |",
            "|---|" + "---|" * (len(TARGETS) + 3)]
    for label, tag in SETS:
        c, clip = load(f"classify_{tag}_256px"), load(f"clip_{tag}_256px")
        if c is None:
            continue
        per = " | ".join(fmt(c["own"]["auroc"].get(t)) for t in TARGETS)
        zs = fmt(clip["alignment"]["zeroshot_acc"], 2) if clip else "–"
        p1 = fmt(clip["image_text"]["P@1"], 2) if clip else "–"
        rows.append(f"| {label} | {per} | {fmt(c['own']['macro_auroc'])} | {zs} | {p1} |")
    return "\n".join(rows)


def ci95(c):
    n = c["n_seeds"]
    return stats.t.ppf(0.975, n - 1) * c["macro_std"] / np.sqrt(n) if n > 1 else 0.0


def table_downstream(s5, valid):
    cells = s5["cells"]
    rows = ["| Real data | n real rows | Real | Synth | Mixed | Synth − Real | Mixed − Real |", "|---|---|---|---|---|---|---|"]
    for frac in FRACS:
        r = cells.get(f"real_f{frac}")
        if r is None:
            continue
        cell = lambda d: (f"{cells[d]['macro_mean']:.3f} ± {ci95(cells[d]):.3f}" if cells[d]["n_seeds"] > 1
                          else f"{cells[d]['macro_mean']:.3f} (1 seed)") if d in cells else "–"
        delta = lambda d: f"{cells[d]['delta_vs_real'] + 0.0:+.3f}".replace("-0.000", "+0.000") if "delta_vs_real" in cells.get(d, {}) else "–"
        rows.append(f"| {float(frac):.0%} | {r['n_train']:,} | {cell(f'real_f{frac}')} | {cell(f'synth_f{frac}')} | "
                    f"{cell(f'mixed_f{frac}')} | {delta(f'synth_f{frac}')} | {delta(f'mixed_f{frac}')} |")
    out = "Test split (18,932 images), macro AUROC over the 5 targets, mean ± 95% CI over 3 seeds:\n\n" + "\n".join(rows)
    if valid:
        vc = valid["cells"]
        rows = ["| Real data | Real | Synth | Mixed |", "|---|---|---|---|"]
        for frac in FRACS:
            if f"real_f{frac}" not in vc:
                continue
            rows.append(f"| {float(frac):.0%} | " + " | ".join(
                f"{vc[k]['macro_mean']:.3f} ± {vc[k]['macro_std']:.3f}" if k in vc else "–"
                for k in (f"{d}_f{frac}" for d in DATA)) + " |")
        out += (f"\n\nOfficial CheXpert valid ({valid['n_images']} frontal images, radiologist labels), macro AUROC over "
                f"{' + '.join(valid['labels_in_macro'])} only (the other labels have ≤1 positive), mean ± std:\n\n" + "\n".join(rows))
    return out


def figure(s5):
    cells = s5["cells"]
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    for d, col in DATA.items():
        pts = [(cells[k]["n_train"] if d == "real" else cells[f"real_f{f}"]["n_train"], cells[k]["macro_mean"], ci95(cells[k]))
               for f in FRACS if (k := f"{d}_f{f}") in cells]
        if not pts:
            continue
        x, y, e = map(np.array, zip(*pts))
        x = x * {"real": 1.0, "synth": 0.95, "mixed": 1.05}[d]     # dodge so the error bars don't overlap
        ax.errorbar(x, y, yerr=e, color=col, marker="o", ms=5, lw=2, capsize=3, label=d.capitalize())
    ax.set_xscale("log")
    ax.set_xlabel("real training rows (synth uses one twin per row)", color=INK, fontsize=9)
    ax.set_ylabel("test macro AUROC (5 targets)", color=INK, fontsize=9)
    ax.set_title("Stage 5: downstream classifier vs training-set size", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "stage5_auroc_vs_size.png")
    print("wrote", OUT / "stage5_auroc_vs_size.png")


def main():
    s5, valid = load("stage5"), load("chexpert_valid")
    md = ("## 1. Fidelity and diversity (256 px, n = 2000 test-row prompts, DPM++ 25, g = 4)\n\n" + table_fidelity() +
          "\n\n## 2. Classification of synthetic images (own DenseNet judge, per-label AUROC; BiomedCLIP)\n\n" + table_classify() +
          "\n\n## 3. Downstream: Real vs Synth vs Mixed (DenseNet-121, C@15k twins)\n\n" + table_downstream(s5, valid) + "\n")
    (RESULTS / "tables.md").write_text("# Result tables (generated by `python -m roentgen.aggregate`)\n\n" + md)
    print(md)
    figure(s5)
    print("wrote", RESULTS / "tables.md")


if __name__ == "__main__":
    main()
