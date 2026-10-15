"""Stage 4d: qualitative figures (paper Figs. 2-4) for one run, each backed by a small numeric check.

    python -m roentgen.qualitative [--run C --step 15000 --guidance 4.0 --seeds 16 --bs 4]

Every image is scored by the own judge (eval_classify "own"). A finding counts as "called" when
its probability exceeds the judge's 95th percentile on real test images *negative* for that label
(i.e. a 5% false-positive-rate threshold calibrated on real data).

  fig2_pathologies.png  one row per VAL_PROMPT: 2 real test positives | synthetic seeds.
  fig3_modifiers.png    the 8 unseen MODIFIER_PROMPTS.
  fig3_laterality.png   "left-sided X" vs "right-sided X" from the same noise, plus the mean
                        difference map (left - right). On a PA film the patient's left is on the
                        image *right*, so a model that learned the convention brightens image-right.
  fig4_composition.png  "A" | "B" | "A and B" from the same noise, with the judge's p(A), p(B).
  fig_guidance.png      one prompt per row across guidance scales, fixed seed.

Sampled PNGs are cached under artifacts/eval/<run>_step<step>_qual/ and reused on re-runs.
Writes the figures to artifacts/figures/qual_<run>_step<step>/ and results/qual_<run>_step<step>.json.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from roentgen.config import ARTIFACTS, LABELS, SEED, TARGETS
from roentgen.data import read_manifest
from roentgen.eval_fidelity import EVAL_DIR, RESULTS, load_real, reference_rows
from roentgen.prompts import MODIFIER_PROMPTS, PHRASES
from roentgen.train_classifier import load_judge, predict
from roentgen.train_lora import RUNS, VAL_PROMPTS

LATERALITY = {
    "Pleural Other": "chest x-ray with {side}-sided pleural abnormality",
    "Pneumonia": "chest x-ray with {side} lower lobe pneumonia",
    "Lung Lesion": "chest x-ray with {side} lung lesion",
    "Fracture": "chest x-ray with {side}-sided rib fracture",
}
PAIRS = [
    ("Pneumonia", "Pleural Other"),
    ("Enlarged Cardiomediastinum", "Pleural Other"),
    ("Pneumonia", "Enlarged Cardiomediastinum"),
    ("Lung Lesion", "Fracture"),
]
# VAL_PROMPTS[0] is the normal prompt; the rest are not in TARGETS order, so match by phrase
VAL_TARGET = [None] + [next(t for t in TARGETS if PHRASES[t] in p) for p in VAL_PROMPTS[1:]]
GUIDANCES = [1.0, 2.0, 3.0, 4.0, 5.5, 7.5]
GUIDANCE_PROMPTS = VAL_PROMPTS[1:5]


def single(t):
    return f"chest x-ray with {PHRASES[t]}"


def both(a, b):
    return f"chest x-ray with {PHRASES[a]} and {PHRASES[b]}"


# ----------------------------------------------------------------------------- sampling
class Sampler:
    """Caches one PNG per (prompt, seed, guidance); the pipeline is only built if something is missing."""

    def __init__(self, run, step, out_dir, steps, bs):
        self.run, self.step, self.dir, self.steps, self.bs = run, step, out_dir, steps, bs
        self.dir.mkdir(parents=True, exist_ok=True)
        self.size = RUNS[run]["size"]
        self.pipe = None

    def path(self, prompt, seed, g):
        return self.dir / f"g{g}_{prompt.replace(' ', '_')[:90]}_s{seed}.png"

    @torch.no_grad()
    def __call__(self, prompts, seeds, g):
        """-> uint8 array (len(prompts), len(seeds), H, W)."""
        jobs = [(p, s) for p in prompts for s in seeds]
        todo = [j for j in jobs if not self.path(*j, g).exists()]
        if todo and self.pipe is None:
            from roentgen.compare_runs import build_pipe
            self.pipe = build_pipe(self.run, self.step)
        for b in range(0, len(todo), self.bs):
            chunk = todo[b:b + self.bs]
            gens = [torch.Generator("cuda").manual_seed(SEED + s) for _, s in chunk]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ims = self.pipe([p for p, _ in chunk], height=self.size, width=self.size,
                                num_inference_steps=self.steps, guidance_scale=g, generator=gens,
                                output_type="np").images
            for (p, s), im in zip(chunk, ims):
                Image.fromarray((im.mean(-1) * 255).round().astype(np.uint8)).save(self.path(p, s, g))
        if todo:
            print(f"  sampled {len(todo)} images at g={g}", flush=True)
        imgs = np.stack([np.asarray(Image.open(self.path(p, s, g))) for p, s in jobs])
        return imgs.reshape(len(prompts), len(seeds), *imgs.shape[1:])


# ----------------------------------------------------------------------------- judge
class Judge:
    def __init__(self, size):
        self.model = load_judge()
        rows = reference_rows("test", 2000)
        y = (rows[TARGETS].fillna(0).values == 1)
        p = predict(self.model, load_real(rows, size))
        # 5%-FPR threshold per label, calibrated on real test negatives
        self.thr = np.array([np.quantile(p[~y[:, k], k], 0.95) for k in range(len(TARGETS))])
        self.real_pos_rate = {t: float((p[y[:, k], k] > self.thr[k]).mean()) for k, t in enumerate(TARGETS)}

    def __call__(self, imgs):
        """imgs (..., H, W) uint8 -> probs (..., 5)."""
        flat = imgs.reshape(-1, *imgs.shape[-2:])
        return predict(self.model, flat).reshape(*imgs.shape[:-2], len(TARGETS))

    def called(self, probs, t):
        k = TARGETS.index(t)
        return probs[..., k] > self.thr[k]


# ----------------------------------------------------------------------------- figures
def tile(ax, im, title=None, color="black"):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, color=color, pad=2)


def row_label(ax, text):
    ax.set_ylabel(text, fontsize=7, rotation=0, ha="right", va="center", labelpad=6)


def wrap(text, width=28):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\n".join(lines + [cur])


def fig2(sampler, judge, g, seeds, show, out, size):
    syn = sampler(VAL_PROMPTS, seeds, g)                                   # (6, S, H, W)
    probs = judge(syn)                                                     # (6, S, 5)
    test = read_manifest("test")
    lab = (test[LABELS].fillna(0).values == 1)
    rng = np.random.default_rng(SEED)
    real = []
    for t in VAL_TARGET:
        k = LABELS.index(t) if t else LABELS.index("No Finding")
        tk = [LABELS.index(x) for x in TARGETS]
        # prefer images with exactly this one target (or no target for normal)
        ok = lab[:, k] & (lab[:, tk].sum(1) == (1 if t else 0))
        real.append(test[ok].iloc[rng.choice(ok.sum(), 2, replace=False)])
    real_imgs = [load_real(r, size) for r in real]
    real_probs = [judge(r) for r in real_imgs]

    stats = {}
    fig, axs = plt.subplots(len(VAL_PROMPTS), 2 + show, figsize=(1.25 * (2 + show) + 1.6, 1.35 * len(VAL_PROMPTS)))
    for i, (p, t) in enumerate(zip(VAL_PROMPTS, VAL_TARGET)):
        k = TARGETS.index(t) if t else None
        for j in range(2):
            pr = real_probs[i][j]
            txt = f"real  p={pr[k]:.2f}" if t else f"real  max p={pr.max():.2f}"
            tile(axs[i, j], real_imgs[i][j], txt, "dimgray")
        for j in range(show):
            pr = probs[i, j]
            if t:
                c = "tab:green" if judge.called(pr, t) else "black"
                tile(axs[i, 2 + j], syn[i, j], f"p={pr[k]:.2f}", c)
            else:
                tile(axs[i, 2 + j], syn[i, j], f"max p={pr.max():.2f}")
        row_label(axs[i, 0], wrap(p))
        if t:
            stats[p] = {"target": t, "mean_p": float(probs[i, :, k].mean()),
                        "called_rate": float(judge.called(probs[i], t).mean()),
                        "real_called_rate": judge.real_pos_rate[t]}
        else:
            stats[p] = {"any_called_rate": float(np.mean([judge.called(probs[i], x) for x in TARGETS]).item()),
                        "mean_max_p": float(probs[i].max(-1).mean())}
    fig.suptitle("Fig. 2 - one prompt per pathology (left: real test positives; green = called at 5% real FPR)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig2_pathologies.png", dpi=200)
    plt.close(fig)
    return stats


def fig3_modifiers(sampler, judge, g, seeds, show, out):
    syn = sampler(MODIFIER_PROMPTS, seeds, g)
    probs = judge(syn)
    fig, axs = plt.subplots(len(MODIFIER_PROMPTS), show, figsize=(1.25 * show + 1.6, 1.35 * len(MODIFIER_PROMPTS)))
    stats = {}
    for i, p in enumerate(MODIFIER_PROMPTS):
        t = next(x for x in TARGETS if PHRASES[x] in p)
        k = TARGETS.index(t)
        for j in range(show):
            c = "tab:green" if judge.called(probs[i, j], t) else "black"
            tile(axs[i, j], syn[i, j], f"p={probs[i, j, k]:.2f}", c)
        row_label(axs[i, 0], wrap(p))
        stats[p] = {"target": t, "mean_p": float(probs[i, :, k].mean()),
                    "called_rate": float(judge.called(probs[i], t).mean())}
    fig.suptitle("Fig. 3a - unseen modifier prompts (green = target called at 5% real FPR)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig3_modifiers.png", dpi=200)
    plt.close(fig)
    return stats


def fig3_laterality(sampler, g, seeds, show, out):
    """Signed asymmetry S = mean(d | image-right half) - mean(d | image-left half), d = left-prompt - right-prompt.
    S > 0 means the 'left' prompt brightened the image right = patient's left (correct PA convention)."""
    fig, axs = plt.subplots(len(LATERALITY), 2 * show + 1, figsize=(1.25 * (2 * show + 1) + 1.6, 1.4 * len(LATERALITY)))
    stats = {}
    for i, (t, tpl) in enumerate(LATERALITY.items()):
        L, R = sampler([tpl.format(side="left"), tpl.format(side="right")], seeds, g)
        d = L.astype(np.float32) - R.astype(np.float32)                   # (S, H, W)
        half = d.shape[-1] // 2
        S = d[:, :, half:].mean((1, 2)) - d[:, :, :half].mean((1, 2))
        stats[t] = {"S_mean": float(S.mean()), "S_se": float(S.std(ddof=1) / np.sqrt(len(S))),
                    "frac_correct_side": float((S > 0).mean()), "n": len(S)}
        for j in range(show):
            tile(axs[i, 2 * j], L[j], f"left  s{seeds[j]}")
            tile(axs[i, 2 * j + 1], R[j], f"right s{seeds[j]}")
        m = d.mean(0)
        lim = max(np.abs(m).max(), 1e-6)
        ax = axs[i, -1]
        ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.axvline(half, color="k", lw=0.5, ls=":")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"mean(L-R)  S={S.mean():+.1f}\n{(S > 0).mean():.0%} correct", fontsize=7, pad=2)
        row_label(axs[i, 0], wrap(tpl.format(side="left/right")))
    fig.suptitle("Fig. 3b - laterality: same noise, 'left' vs 'right'. Red on image right = patient's left = correct.", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig3_laterality.png", dpi=200)
    plt.close(fig)
    return stats


def fig4(sampler, judge, g, seeds, show, out):
    normal = judge(sampler([VAL_PROMPTS[0]], seeds, g))[0]                # (S, 5)
    fig, axs = plt.subplots(len(PAIRS), 3 * show, figsize=(1.25 * 3 * show + 1.6, 1.45 * len(PAIRS)))
    stats = {}
    for i, (a, b) in enumerate(PAIRS):
        ka, kb = TARGETS.index(a), TARGETS.index(b)
        prompts = [single(a), single(b), both(a, b)]
        syn = sampler(prompts, seeds, g)                                  # (3, S, H, W)
        pr = judge(syn)                                                   # (3, S, 5)
        names = ["A only", "B only", "A + B"]
        rec = {}
        for c, n in enumerate(names):
            ca, cb = judge.called(pr[c], a), judge.called(pr[c], b)
            rec[n] = {"mean_pA": float(pr[c, :, ka].mean()), "mean_pB": float(pr[c, :, kb].mean()),
                      "called_A": float(ca.mean()), "called_B": float(cb.mean()), "called_both": float((ca & cb).mean())}
        rec["normal"] = {"mean_pA": float(normal[:, ka].mean()), "mean_pB": float(normal[:, kb].mean())}
        stats[f"A={a} | B={b}"] = rec
        for j in range(show):
            for c in range(3):
                ax = axs[i, 3 * j + c]
                ok = judge.called(pr[c, j], a) and judge.called(pr[c, j], b)
                tile(ax, syn[c, j], f"{names[c]}\npA={pr[c, j, ka]:.2f} pB={pr[c, j, kb]:.2f}",
                     "tab:green" if c == 2 and ok else "black")
        row_label(axs[i, 0], wrap(f"A: {PHRASES[a]}\nB: {PHRASES[b]}", 22))
    fig.suptitle("Fig. 4 - composition, same noise per triple (green = both findings called at 5% real FPR)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig4_composition.png", dpi=200)
    plt.close(fig)
    return stats


def fig_guidance(sampler, seed, out):
    cols = [sampler(GUIDANCE_PROMPTS, [seed], g)[:, 0] for g in GUIDANCES]   # each (P, H, W)
    fig, axs = plt.subplots(len(GUIDANCE_PROMPTS), len(GUIDANCES),
                            figsize=(1.25 * len(GUIDANCES) + 1.6, 1.35 * len(GUIDANCE_PROMPTS)))
    for i, p in enumerate(GUIDANCE_PROMPTS):
        for j, g in enumerate(GUIDANCES):
            tile(axs[i, j], cols[j][i], f"g={g}" if i == 0 else None)
        row_label(axs[i, 0], wrap(p))
    fig.suptitle(f"Guidance scale, fixed seed {seed}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig_guidance.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="C")
    ap.add_argument("--step", type=int, default=15000)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, default=16, help="samples per prompt for the numbers")
    ap.add_argument("--show", type=int, default=6, help="samples per prompt shown in figs 2/3a")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()

    tag = f"{a.run}_step{a.step}"
    out = ARTIFACTS / "figures" / f"qual_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(a.run, a.step, EVAL_DIR / f"{tag}_qual", a.steps, a.bs)
    seeds = list(range(a.seeds))
    size = RUNS[a.run]["size"]
    judge = Judge(size)

    res = {"args": vars(a), "thresholds_5pct_fpr": dict(zip(TARGETS, map(float, judge.thr))),
           "real_called_rate": judge.real_pos_rate}
    print("fig2 ...", flush=True)
    res["fig2"] = fig2(sampler, judge, a.guidance, seeds, a.show, out, size)
    print("fig3 ...", flush=True)
    res["fig3_modifiers"] = fig3_modifiers(sampler, judge, a.guidance, seeds, a.show, out)
    res["fig3_laterality"] = fig3_laterality(sampler, a.guidance, seeds, 2, out)
    print("fig4 ...", flush=True)
    res["fig4"] = fig4(sampler, judge, a.guidance, seeds, 2, out)
    print("guidance strip ...", flush=True)
    fig_guidance(sampler, seeds[0], out)

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"qual_{tag}.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=2))
    print("wrote", out, "and", path)


if __name__ == "__main__":
    main()
