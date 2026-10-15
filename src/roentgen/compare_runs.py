"""Side-by-side comparison of trained runs on identical prompts and seeds.

    python -m roentgen.compare_runs --runs A B [--step 15000] [--seeds 4] [--size 256]

For every prompt (the 6 training-style VAL_PROMPTS + the 8 unseen MODIFIER_PROMPTS) and
seed, each run samples from the same initial noise, so differences come from the weights
alone. Each run samples at its native training size; --size downsamples every image to a
common resolution before metrics (needed to compare runs trained at different sizes, e.g. B vs C).
Writes per-image PNGs, one comparison sheet per prompt set, and metrics.json with
  * cross-run SSIM   — same prompt+seed, run X vs run Y (how much the weights change the image)
  * intra-run SSIM   — mean pairwise SSIM across seeds of one prompt (lower = more diverse)
"""
import argparse
import itertools
import json

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from roentgen.config import ARTIFACTS, SD_MODEL, SEED
from roentgen.prompts import MODIFIER_PROMPTS
from roentgen.train_lora import RUNS, VAL_PROMPTS, build_models, load_trainable


def build_pipe(run, step):
    """DPM-Solver++ StableDiffusionPipeline with run's trainable weights at `step` loaded
    (step 0 = untrained base SD 1.4, i.e. no checkpoint is loaded)."""
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline

    unet, te, tok, _ = build_models(run)
    if step:
        load_trainable(ARTIFACTS / "runs" / run / f"ckpt-{step}" / "trainable.safetensors", unet, te)
    unet.eval(); te.eval()
    vae = AutoencoderKL.from_pretrained(SD_MODEL, subfolder="vae", torch_dtype=torch.bfloat16).cuda()
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=te, tokenizer=tok, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(SD_MODEL, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def sample_run(run, step, prompts, seeds, steps, guidance):
    pipe = build_pipe(run, step)
    size = RUNS[run]["size"]
    out = {}
    for p in prompts:
        gens = [torch.Generator("cuda").manual_seed(SEED + s) for s in seeds]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ims = pipe([p] * len(seeds), height=size, width=size, num_inference_steps=steps,
                       guidance_scale=guidance, generator=gens, output_type="np").images
        out[p] = (ims.mean(-1) * 255).round().astype(np.uint8)          # grayscale uint8, (S,H,W)
    del pipe
    torch.cuda.empty_cache()
    return out


def ssim(a, b):
    return float(structural_similarity(a, b, data_range=255))


def sheet(imgs, runs, prompts, seeds, path, thumb=192, pad=6, label_h=22):
    """Rows = prompts; columns = run blocks of len(seeds) images, separated by a wider gap."""
    from PIL import ImageDraw
    gap = 4 * pad
    block = len(seeds) * (thumb + pad)
    W = 300 + len(runs) * block + (len(runs) - 1) * gap
    H = label_h + len(prompts) * (thumb + pad)
    canvas = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(canvas)
    for j, r in enumerate(runs):
        d.text((300 + j * (block + gap), 4), f"Run {r}  (seeds {seeds[0]}..{seeds[-1]})", fill=0)
    for i, p in enumerate(prompts):
        y = label_h + i * (thumb + pad)
        d.text((4, y + thumb // 2 - 6), p if len(p) < 46 else p[:43] + "...", fill=0)
        for j, r in enumerate(runs):
            for k in range(len(seeds)):
                x = 300 + j * (block + gap) + k * (thumb + pad)
                canvas.paste(Image.fromarray(imgs[r][p][k]).resize((thumb, thumb), Image.LANCZOS), (x, y))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["A", "B"])
    ap.add_argument("--step", type=int, default=15000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=None, help="downsample all images to this size")
    a = ap.parse_args()

    seeds = list(range(a.seeds))
    sets = {"val": VAL_PROMPTS, "modifier": MODIFIER_PROMPTS}
    prompts = VAL_PROMPTS + MODIFIER_PROMPTS
    out = ARTIFACTS / "figures" / (f"compare_{''.join(a.runs)}_step{a.step}" + (f"_{a.size}px" if a.size else ""))
    (out / "images").mkdir(parents=True, exist_ok=True)

    imgs = {}
    for r in a.runs:
        print(f"sampling run {r} ...", flush=True)
        imgs[r] = sample_run(r, a.step, prompts, seeds, a.steps, a.guidance)
        if a.size:
            imgs[r] = {p: np.stack([np.asarray(Image.fromarray(im).resize((a.size, a.size), Image.LANCZOS))
                                    for im in ims]) for p, ims in imgs[r].items()}
        for pi, p in enumerate(prompts):
            for s in seeds:
                Image.fromarray(imgs[r][p][s]).save(out / "images" / f"{r}_p{pi:02d}_s{s}.png")

    metrics = {"args": vars(a), "prompts": {}}
    for p in prompts:
        m = {f"intra_ssim_{r}": float(np.mean([ssim(imgs[r][p][i], imgs[r][p][j])
                                               for i, j in itertools.combinations(seeds, 2)]))
             for r in a.runs}
        for x, y in itertools.combinations(a.runs, 2):
            m[f"cross_ssim_{x}{y}"] = float(np.mean([ssim(imgs[x][p][s], imgs[y][p][s]) for s in seeds]))
        metrics["prompts"][p] = m
    for name, ps in sets.items():
        keys = metrics["prompts"][ps[0]].keys()
        metrics[f"mean_{name}"] = {k: float(np.mean([metrics["prompts"][p][k] for p in ps])) for k in keys}
        sheet(imgs, a.runs, ps, seeds, out / f"sheet_{name}.png")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k.startswith("mean_")}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
