"""Stage 4a: fidelity (FID_XRV, FID_Inception) and intra-prompt diversity (MS-SSIM).

    python -m roentgen.eval_fidelity --run C --step 15000 --size 256 [--n 2000]
    python -m roentgen.eval_fidelity --run real --size 256      # real-val vs real-test sanity floor
    python -m roentgen.eval_fidelity --run sd --size 256        # untrained SD 1.4 baseline (512 px)

Reference set: --n images drawn from the real test split (fixed seed). The synthetic set uses
the *same rows' labels* turned into prompts, so both sets share one label distribution and FID
measures image quality rather than label-mix mismatch. Each run samples at its native training
size and every image (real and synthetic) is brought to --size before feature extraction, which
makes runs trained at different resolutions comparable (B@512 vs C@256 at --size 256).

MS-SSIM: 10 samples per VAL_PROMPT, mean over the 45 pairs per prompt (Gaussian 11, sigma 1.5,
lower = more diverse); a random-pair real baseline is reported alongside for scale.

Generated PNGs are kept under artifacts/eval/ and reused on re-runs; real features are cached.
Writes results/fidelity_<run>_step<step>[_g<guidance>]_<size>px.json (suffix only when guidance != 4.0).
"""
import argparse
import itertools
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

from roentgen.config import ARTIFACTS, LABELS, ROOT, SEED, find_chexpert_dir
from roentgen.data import load_square, read_manifest
from roentgen.prompts import make_prompt
from roentgen.train_lora import RUNS, VAL_PROMPTS

EVAL_DIR = ARTIFACTS / "eval"
RESULTS = ROOT / "results"
MSSSIM_SAMPLES = 10


# ----------------------------------------------------------------------------- data
def reference_rows(split, n):
    df = read_manifest(split)
    return df.sample(n=n, random_state=SEED).reset_index(drop=True)


def load_real(df, size):
    root = find_chexpert_dir()
    return np.stack([np.asarray(load_square(root / p, size).convert("L")) for p in df.rel_path])


def prompts_for(df):
    rng = random.Random(SEED)
    labels = (df[LABELS].fillna(0).values == 1).astype(int)
    return [make_prompt(l, v, rng) for l, v in zip(labels, df.view)]


def resize(imgs, size):
    if imgs.shape[-1] == size:
        return imgs
    return np.stack([np.asarray(Image.fromarray(im).resize((size, size), Image.LANCZOS)) for im in imgs])


@torch.no_grad()
def generate(run, step, prompts, seeds, out_dir, steps, guidance, bs):
    """Native-size grayscale uint8 images, one PNG per (prompt, seed); existing PNGs are reused."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{i:05d}.png" for i in range(len(prompts))]
    todo = [i for i, p in enumerate(paths) if not p.exists()]
    if todo:
        from roentgen.compare_runs import build_pipe
        pipe = build_pipe(run, step)
        size = RUNS[run]["size"]
        for b in range(0, len(todo), bs):
            idx = todo[b:b + bs]
            gens = [torch.Generator("cuda").manual_seed(seeds[i]) for i in idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ims = pipe([prompts[i] for i in idx], height=size, width=size, num_inference_steps=steps,
                           guidance_scale=guidance, generator=gens, output_type="np").images
            for i, im in zip(idx, ims):
                Image.fromarray((im.mean(-1) * 255).round().astype(np.uint8)).save(paths[i])
            print(f"  {out_dir.name}: {min(b + bs, len(todo))}/{len(todo)}", flush=True)
        del pipe
        torch.cuda.empty_cache()
    return np.stack([np.asarray(Image.open(p)) for p in paths])


# ----------------------------------------------------------------------------- features
class Extractors:
    def __init__(self):
        import torchxrayvision as xrv
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.xrv = xrv.models.DenseNet(weights="densenet121-res224-all",
                                       cache_dir=str(ROOT / ".cache" / "xrv")).cuda().eval()
        self.inception = FrechetInceptionDistance(feature=2048).inception.cuda().eval()

    @torch.no_grad()
    def __call__(self, imgs, bs=64):
        """imgs: (N,H,W) uint8 -> {"xrv": (N,1024), "inception": (N,2048)} float64."""
        fx, fi = [], []
        for b in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[b:b + bs]).cuda().unsqueeze(1)                 # (B,1,H,W) uint8
            xr = x.float() / 255 * 2048 - 1024                                       # xrv's [-1024,1024]
            xr = F.interpolate(xr, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
            fx.append(self.xrv.features2(xr).double().cpu())
            fi.append(self.inception(x.expand(-1, 3, -1, -1).contiguous()).double().cpu())
        return {"xrv": torch.cat(fx).numpy(), "inception": torch.cat(fi).numpy()}


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def real_features(ext, size, n):
    path = EVAL_DIR / f"real_test_{size}px_n{n}.npz"
    if path.exists():
        return dict(np.load(path))
    feats = ext(load_real(reference_rows("test", n), size))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **feats)
    return feats


# ----------------------------------------------------------------------------- diversity
def msssim_pairs(groups):
    """groups: list of (K,H,W) uint8 arrays; mean MS-SSIM over all pairs within each group."""
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as ms
    vals = []
    for g in groups:
        t = torch.from_numpy(g).float().div(255).unsqueeze(1).cuda()
        i, j = zip(*itertools.combinations(range(len(g)), 2))
        v = ms(t[list(i)], t[list(j)], data_range=1.0, kernel_size=11, sigma=1.5, reduction="none")
        vals.append(float(v.mean()))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS) + ["real", "sd"])
    ap.add_argument("--step", type=int, default=15000)
    ap.add_argument("--size", type=int, default=256, help="common evaluation resolution")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=None)
    a = ap.parse_args()

    ext = Extractors()
    real = real_features(ext, a.size, a.n)
    res = {"args": vars(a)}

    if a.run == "real":
        # Sanity floor: an independent real sample (val split) against the test reference.
        fake = load_real(reference_rows("val", a.n), a.size)
        groups = None
        tag = f"real_{a.size}px"
    else:
        # "sd" = base SD 1.4 at 512 px: run B's module tree with no checkpoint loaded (step 0).
        run, step = ("B", 0) if a.run == "sd" else (a.run, a.step)
        bs = a.bs or (16 if RUNS[run]["size"] <= 256 else 8)
        rows = reference_rows("test", a.n)
        prompts = prompts_for(rows)
        tag = "sd" if a.run == "sd" else f"{a.run}_step{a.step}"
        if a.guidance != 4.0:   # 4.0 is the default; other scales get their own image dirs
            tag += f"_g{a.guidance:g}"
        print(f"generating {tag} (fid set, n={a.n})", flush=True)
        fake = resize(generate(run, step, prompts, [SEED + i for i in range(a.n)],
                               EVAL_DIR / tag / "fid", a.steps, a.guidance, bs), a.size)
        print(f"generating {tag} (ms-ssim set)", flush=True)
        div_prompts = [p for p in VAL_PROMPTS for _ in range(MSSSIM_SAMPLES)]
        div_seeds = [SEED + 10_000 + i for i in range(len(div_prompts))]
        div = resize(generate(run, step, div_prompts, div_seeds, EVAL_DIR / tag / "msssim",
                              a.steps, a.guidance, bs), a.size)
        groups = [div[k * MSSSIM_SAMPLES:(k + 1) * MSSSIM_SAMPLES] for k in range(len(VAL_PROMPTS))]
        tag = f"{tag}_{a.size}px"

    feats = ext(fake)
    res["fid_xrv"] = frechet(feats["xrv"], real["xrv"])
    res["fid_inception"] = frechet(feats["inception"], real["inception"])

    # Random real pairs give the MS-SSIM scale for "unrelated CXRs" (same pair count as one run).
    real_imgs = load_real(reference_rows("test", len(VAL_PROMPTS) * MSSSIM_SAMPLES), a.size)
    real_groups = [real_imgs[k * MSSSIM_SAMPLES:(k + 1) * MSSSIM_SAMPLES] for k in range(len(VAL_PROMPTS))]
    res["msssim_real_random"] = float(np.mean(msssim_pairs(real_groups)))
    if groups is not None:
        per = msssim_pairs(groups)
        res["msssim"] = {"mean": float(np.mean(per)), "sd": float(np.std(per)),
                         "per_prompt": dict(zip(VAL_PROMPTS, per))}

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"fidelity_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
