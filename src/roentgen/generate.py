"""Stage 3: label-matched synthetic twin of the real train split.

    python -m roentgen.generate --smoke                                   # 16 images, sanity pass
    python -m roentgen.generate                                           # C@22.5k, g4, DPM++ 25 (main corpus)
    python -m roentgen.generate --sampler pndm --steps 75 --limit 2000    # paper sampler, sampler-check subset

One synthetic image per real train row: same label vector, a prompt templated from it with the
training templates, and a per-row seed, so image i is regenerable from (row, seed) alone and the
"mixed" Stage 5 experiment can pair every real row with its twin.

Rows are generated in priority order: the union of Stage 5's 1% subsets (seeds 0-2), then of the
10% subsets, then the rest. Those subsets are reproduced exactly as train_classifier.train draws
them, so the small-data experiments are covered early and the job can be stopped at any point.
Existing PNGs are skipped (resumable). Every --flush images, and at the end, writes
<out>/synthetic_manifest.parquet (train_idx, path, prompt, labels, seed, sampler, steps, ckpt step,
guidance) covering the PNGs that exist so far.
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from roentgen.config import ARTIFACTS, LABELS, SD_MODEL, SEED
from roentgen.data import read_manifest
from roentgen.prompts import make_prompt
from roentgen.train_lora import RUNS

SYNTH_DIR = ARTIFACTS / "synth"
FRACS = (0.01, 0.1)
CLS_SEEDS = (0, 1, 2)
SEED_BASE = 1_000_000          # disjoint from the eval seeds (SEED + i, SEED + 10_000 + i)


def priority_order(n):
    """Train-row indices: 1% subsets first, then 10% subsets, then the rest (as train_classifier draws them)."""
    order, seen = [], np.zeros(n, bool)
    for frac in FRACS:
        tier = set()
        for s in CLS_SEEDS:
            rng = np.random.default_rng(SEED + s)
            tier.update(rng.choice(n, int(frac * n), replace=False).tolist())
        tier = sorted(i for i in tier if not seen[i])
        seen[tier] = True
        order += tier
    return np.array(order + np.flatnonzero(~seen).tolist())


def build_rows():
    df = read_manifest("train")
    labels = (df[LABELS].fillna(0).values == 1).astype(int)
    df = df[["rel_path", "view"] + LABELS].copy()
    df.insert(0, "train_idx", np.arange(len(df)))
    df["prompt"] = [make_prompt(l, v, random.Random(SEED + i)) for i, (l, v) in enumerate(zip(labels, df.view))]
    df["seed"] = SEED_BASE + df.train_idx
    return df


def write_manifest(rows, out, a):
    done = rows[[(out / "png" / f"{i:06d}.png").exists() for i in rows.train_idx]].copy()
    done["path"] = [f"png/{i:06d}.png" for i in done.train_idx]
    done["run"], done["ckpt_step"], done["guidance"] = a.run, a.step, a.guidance
    done["sampler"], done["steps"] = a.sampler, a.steps
    done.sort_values("train_idx").to_parquet(out / "synthetic_manifest.parquet", index=False)
    return len(done)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="C")
    ap.add_argument("--step", type=int, default=22500)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--sampler", choices=["dpm", "pndm"], default="dpm")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows in priority order")
    ap.add_argument("--flush", type=int, default=2000, help="rewrite the manifest every N new images")
    ap.add_argument("--manifest_only", action="store_true", help="rebuild the manifest from existing PNGs and exit")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.limit = 16

    tag = f"{a.run}_step{a.step}_g{a.guidance}_{a.sampler}{a.steps}" + ("_smoke" if a.smoke else "")
    out = SYNTH_DIR / tag
    (out / "png").mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    order = priority_order(len(rows))[:a.limit]
    rows = rows.iloc[order].reset_index(drop=True)
    if a.manifest_only:
        print(f"{write_manifest(rows, out, a)} images in manifest -> {out}")
        return
    todo = rows[[not (out / "png" / f"{i:06d}.png").exists() for i in rows.train_idx]]
    print(f"{tag}: {len(rows)} planned, {len(todo)} to generate -> {out}", flush=True)

    if len(todo):
        from diffusers import PNDMScheduler
        from roentgen.compare_runs import build_pipe
        pipe = build_pipe(a.run, a.step)
        if a.sampler == "pndm":
            pipe.scheduler = PNDMScheduler.from_pretrained(SD_MODEL, subfolder="scheduler")
        size = RUNS[a.run]["size"]
        t0, since_flush = time.time(), 0
        for b in range(0, len(todo), a.bs):
            chunk = todo.iloc[b:b + a.bs]
            gens = [torch.Generator("cuda").manual_seed(int(s)) for s in chunk.seed]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ims = pipe(chunk.prompt.tolist(), height=size, width=size, num_inference_steps=a.steps,
                           guidance_scale=a.guidance, generator=gens, output_type="np").images
            for i, im in zip(chunk.train_idx, ims):
                Image.fromarray((im.mean(-1) * 255).round().astype(np.uint8)).save(out / "png" / f"{i:06d}.png")
            n = min(b + a.bs, len(todo))
            since_flush += len(chunk)
            if since_flush >= a.flush or n == len(todo):
                since_flush = 0
                rate = n / (time.time() - t0)
                print(f"{n}/{len(todo)}  {rate:.2f} img/s  eta {(len(todo) - n) / rate / 3600:.1f} h  "
                      f"manifest={write_manifest(rows, out, a)}", flush=True)
    print(f"GEN_DONE {write_manifest(rows, out, a)} images -> {out / 'synthetic_manifest.parquet'}", flush=True)


if __name__ == "__main__":
    main()
