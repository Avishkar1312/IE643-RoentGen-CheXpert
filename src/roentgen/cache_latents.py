"""Stage 1b: precompute frozen-VAE latents + CLIP token IDs for the diffusion subset.

    python -m roentgen.cache_latents --size 512 [--n_prompts 4] [--smoke]

Writes artifacts/cache/latents_{size}.npy (N,4,s/8,s/8 fp16), tokens.npy (N,K,77 int32),
prompts.json and index.parquet (row order of the cache).
"""
import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from roentgen.config import CACHE_DIR, LABELS, SD_MODEL, SEED
from roentgen.data import DiffusionCXRDataset, read_manifest
from roentgen.prompts import make_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n_prompts", type=int, default=4, help="template variants cached per image")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    from diffusers import AutoencoderKL
    from transformers import CLIPTokenizer

    df = read_manifest("train", diffusion_only=True)
    if a.smoke:
        df = df.sample(64, random_state=SEED).reset_index(drop=True)
    out = CACHE_DIR / ("smoke" if a.smoke else "")
    out.mkdir(parents=True, exist_ok=True)

    # Token IDs (resolution-independent): K template variants per image.
    tok = CLIPTokenizer.from_pretrained(SD_MODEL, subfolder="tokenizer")
    rng = random.Random(SEED)
    labels = (df[LABELS].fillna(0).values == 1)
    prompts = [[make_prompt(labels[i], df.view[i], rng) for _ in range(a.n_prompts)] for i in range(len(df))]
    flat = [p for ps in prompts for p in ps]
    lens = [len(x) for x in tok(flat, truncation=False).input_ids]
    assert max(lens) <= tok.model_max_length, f"prompt exceeds 77 tokens ({max(lens)})"
    ids = tok(flat, padding="max_length", max_length=tok.model_max_length, return_tensors="np").input_ids
    np.save(out / "tokens.npy", ids.reshape(len(df), a.n_prompts, -1).astype(np.int32))
    (out / "prompts.json").write_text(json.dumps(prompts))
    df.to_parquet(out / "index.parquet", index=False)
    print(f"tokens: {len(df)} x {a.n_prompts}, max len {max(lens)}")

    # Latents.
    vae = AutoencoderKL.from_pretrained(SD_MODEL, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    s = a.size // 8
    lat = np.lib.format.open_memmap(out / f"latents_{a.size}.npy", mode="w+", dtype=np.float16,
                                    shape=(len(df), 4, s, s))
    dl = DataLoader(DiffusionCXRDataset(df, a.size), batch_size=a.batch, num_workers=a.workers,
                    pin_memory=True, shuffle=False)
    with torch.no_grad():
        for b in tqdm(dl, desc=f"VAE {a.size}"):
            x = b["pixel_values"].cuda(non_blocking=True).to(torch.bfloat16)
            z = vae.encode(x).latent_dist.mean * vae.config.scaling_factor   # mean: deterministic cache
            lat[b["idx"].numpy()] = z.float().cpu().numpy().astype(np.float16)
    lat.flush()
    s = np.asarray(lat[:1024], dtype=np.float32)
    print(f"latents: {lat.shape}, finite {np.isfinite(s).all()}, mean {s.mean():.3f} std {s.std():.3f}")


if __name__ == "__main__":
    main()
