"""Stage 2: domain-adapt SD 1.4 on cached CheXpert latents.

    python -m roentgen.train_lora --run A [--steps N] [--resume_from latest] [--smoke]

Runs (paper Table 1 mapping):
    A  LoRA U-Net + LoRA text encoder @512   ~ "SD U-Net + CLIP"
    B  LoRA U-Net only, CLIP frozen  @512    ~ "SD U-Net only"
    C  full U-Net, CLIP frozen       @256    full-weight counterpart of B

Training reads the frozen-VAE latent cache only (no image decode, no VAE in the loop).
Every random draw (batch order, prompt variant, prompt dropout, noise, timestep) is seeded
from the micro-step index, so a resumed run reproduces the uninterrupted one.
"""
import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from roentgen.config import ARTIFACTS, CACHE_DIR, SD_MODEL, SEED

RUNS = {
    "A": dict(size=512, unet="lora", te="lora", lr=1e-4),
    "B": dict(size=512, unet="lora", te="frozen", lr=1e-4),
    "C": dict(size=256, unet="full", te="frozen", lr=5e-5),
}
VAL_PROMPTS = [
    "normal frontal chest x-ray",
    "chest x-ray with pneumonia",
    "PA chest radiograph with pleural abnormality",
    "chest x-ray with lung lesion",
    "frontal chest radiograph showing enlarged cardiomediastinum",
    "chest x-ray with fracture",
]
UNET_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
TE_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


# ----------------------------------------------------------------------------- models
def build_models(run: str, unet_rank=16, te_rank=8, device="cuda"):
    """Returns (unet, text_encoder, tokenizer, noise_scheduler) configured for `run`.
    Shared with generate.py so checkpoints load into an identical module tree."""
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    cfg = RUNS[run]
    tok = CLIPTokenizer.from_pretrained(SD_MODEL, subfolder="tokenizer")
    sched = DDPMScheduler.from_pretrained(SD_MODEL, subfolder="scheduler")
    full = cfg["unet"] == "full"
    unet = UNet2DConditionModel.from_pretrained(
        SD_MODEL, subfolder="unet", torch_dtype=torch.float32 if full else torch.bfloat16)
    te = CLIPTextModel.from_pretrained(SD_MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16)

    unet.requires_grad_(full)
    te.requires_grad_(False)
    if cfg["unet"] == "lora":
        unet.add_adapter(LoraConfig(r=unet_rank, lora_alpha=unet_rank, lora_dropout=0.0,
                                    init_lora_weights="gaussian", target_modules=UNET_LORA_TARGETS))
    if cfg["te"] == "lora":
        te.add_adapter(LoraConfig(r=te_rank, lora_alpha=te_rank, lora_dropout=0.0,
                                  init_lora_weights="gaussian", target_modules=TE_LORA_TARGETS))
    # Trainable params in fp32 (master weights); frozen base stays bf16; compute under bf16 autocast.
    for m in (unet, te):
        for p in m.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    return unet.to(device), te.to(device), tok, sched


def trainable_state(*mods):
    return {f"{tag}.{n}": p.detach().contiguous().cpu()
            for tag, m in zip(("unet", "te"), mods) for n, p in m.named_parameters() if p.requires_grad}


def load_trainable(path, unet, te):
    sd = load_file(str(path))
    for tag, m in (("unet", unet), ("te", te)):
        sub = {k[len(tag) + 1:]: v for k, v in sd.items() if k.startswith(tag + ".")}
        if sub:
            missing = set(sub) - set(dict(m.named_parameters()))
            assert not missing, f"unknown keys for {tag}: {list(missing)[:3]}"
            m.load_state_dict(sub, strict=False)


# ----------------------------------------------------------------------------- data
class LatentBatches:
    """Deterministic batches from the in-RAM latent/token cache, addressed by micro-step."""

    def __init__(self, size, smoke, bs, seed, p_drop, empty_ids):
        d = CACHE_DIR / "smoke" if smoke else CACHE_DIR
        self.lat = torch.from_numpy(np.load(d / f"latents_{size}.npy"))      # fully into RAM
        self.tok = torch.from_numpy(np.load(d / "tokens.npy").astype(np.int64))
        self.N, self.K = len(self.lat), self.tok.shape[1]
        self.bs, self.seed, self.p_drop = bs, seed, p_drop
        self.empty = torch.as_tensor(empty_ids, dtype=torch.int64)
        self._epoch, self._perm = -1, None

    def perm(self, epoch):
        if epoch != self._epoch:
            self._epoch = epoch
            self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.N)
        return self._perm

    def __call__(self, micro):
        pos = micro * self.bs + np.arange(self.bs)
        idx = np.array([self.perm(p // self.N)[p % self.N] for p in pos])
        rng = np.random.default_rng([self.seed, 7, micro])
        ids = self.tok[idx, rng.integers(0, self.K, self.bs)]
        drop = torch.from_numpy(rng.random(self.bs) < self.p_drop)
        ids[drop] = self.empty                                               # CFG prompt dropout
        return self.lat[idx].float(), ids


# ----------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample_grid(unet, te, tok, size, path, steps=25, guidance=4.0, seed=SEED):
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline
    from torchvision.utils import save_image

    vae = AutoencoderKL.from_pretrained(SD_MODEL, subfolder="vae", torch_dtype=torch.bfloat16).cuda()
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=te, tokenizer=tok, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(SD_MODEL, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    was_training = unet.training
    unet.eval(); te.eval()
    gens = [torch.Generator("cuda").manual_seed(seed + i) for i in range(len(VAL_PROMPTS))]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ims = pipe(VAL_PROMPTS, height=size, width=size, num_inference_steps=steps,
                   guidance_scale=guidance, generator=gens, output_type="pt").images
    save_image(ims.float(), path, nrow=len(VAL_PROMPTS))
    if was_training:
        unet.train(); te.train()
    del pipe, vae
    torch.cuda.empty_cache()


def gpu_stats():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        clk, temp, pw = [x.strip() for x in out.stdout.strip().split(",")]
        return {"sm_mhz": int(clk), "temp_c": int(temp), "power_w": float(pw)}
    except Exception:
        return {}


# ----------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS), default="A")
    ap.add_argument("--steps", type=int, default=6000, help="optimizer steps")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--p_drop", type=float, default=0.1)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--milestone_every", type=int, default=1000, help="ckpts never pruned (0 = none)")
    ap.add_argument("--resume_from", default=None, help="'latest' or a ckpt dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no_sample", action="store_true")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="faster, more VRAM")
    ap.add_argument("--smoke", action="store_true", help="64-image cache, 20 steps, accum 1")
    a = ap.parse_args()
    cfg = RUNS[a.run]
    lr = a.lr or cfg["lr"]
    if a.smoke:
        a.steps, a.accum, a.warmup = min(a.steps, 20), 1, 5
        a.ckpt_every, a.sample_every, a.log_every = 10, 10**9, 1
    out = Path(a.out or ARTIFACTS / "runs" / (f"{a.run}_smoke" if a.smoke else a.run))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({**vars(a), "cfg": cfg, "lr": lr}, indent=2))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    unet, te, tok, sched = build_models(a.run)
    if not a.no_grad_ckpt:
        unet.enable_gradient_checkpointing()
    params = [p for m in (unet, te) for p in m.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    if cfg["unet"] == "full":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-2)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))

    empty_ids = tok("", padding="max_length", max_length=tok.model_max_length).input_ids
    data = LatentBatches(cfg["size"], a.smoke, a.bs, a.seed, a.p_drop, empty_ids)
    alphas = sched.alphas_cumprod.cuda()

    step = 0
    if a.resume_from:
        ck = (sorted(out.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
              if a.resume_from == "latest" else Path(a.resume_from))
        load_trainable(ck / "trainable.safetensors", unet, te)
        st = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); lr_sched.load_state_dict(st["sched"]); step = st["step"]
        print(f"resumed from {ck} at step {step}")

    print(f"run {a.run}: {n_train/1e6:.2f}M trainable params, {data.N} images @ {cfg['size']}px, "
          f"bs {a.bs}x{a.accum}={a.bs*a.accum}, lr {lr}, {a.steps} steps "
          f"(~{a.steps*a.bs*a.accum/data.N:.1f} epochs)")
    if step == 0 and not a.no_sample:
        sample_grid(unet, te, tok, cfg["size"], out / "samples_step0000000.png")

    unet.train(); te.train()
    log = open(out / "log.jsonl", "a")
    torch.cuda.reset_peak_memory_stats()
    t0, losses = time.time(), []
    while step < a.steps:
        for k in range(a.accum):
            micro = step * a.accum + k
            lat, ids = data(micro)
            lat, ids = lat.cuda(non_blocking=True), ids.cuda(non_blocking=True)
            g = torch.Generator("cuda").manual_seed(a.seed * 1_000_003 + micro)
            noise = torch.randn(lat.shape, generator=g, device="cuda")
            t = torch.randint(0, sched.config.num_train_timesteps, (a.bs,), generator=g, device="cuda")
            ac = alphas[t].view(-1, 1, 1, 1)
            noisy = ac.sqrt() * lat + (1 - ac).sqrt() * noise                # q(x_t | x_0), Eq. 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if cfg["te"] == "lora":
                    enc = te(ids)[0]
                else:
                    with torch.no_grad():
                        enc = te(ids)[0]
                pred = unet(noisy, t, enc).sample
            loss = F.mse_loss(pred.float(), noise)                          # eps-prediction, Eq. 3
            (loss / a.accum).backward()
            losses.append(loss.item())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); lr_sched.step(); opt.zero_grad(set_to_none=True)
        step += 1

        if not math.isfinite(losses[-1]):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step % a.log_every == 0 or step == a.steps:
            dt = time.time() - t0
            rec = {"step": step, "loss": float(np.mean(losses)), "lr": lr_sched.get_last_lr()[0],
                   "grad_norm": float(gn), "s_per_step": dt / a.log_every,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, **gpu_stats()}
            log.write(json.dumps(rec) + "\n"); log.flush()
            eta = (a.steps - step) * rec["s_per_step"] / 3600
            print(f"step {step}/{a.steps} loss {rec['loss']:.4f} gn {rec['grad_norm']:.3f} "
                  f"{rec['s_per_step']:.2f}s/step peak {rec['peak_mem_gb']:.2f}GB eta {eta:.1f}h", flush=True)
            t0, losses = time.time(), []
        if step % a.ckpt_every == 0 or step == a.steps:
            ck = out / f"ckpt-{step}"
            ck.mkdir(exist_ok=True)
            save_file(trainable_state(unet, te), str(ck / "trainable.safetensors"))
            torch.save({"opt": opt.state_dict(), "sched": lr_sched.state_dict(), "step": step}, ck / "state.pt")
            for old in sorted(out.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1]))[:-a.keep]:
                if not (a.milestone_every and int(old.name.split("-")[1]) % a.milestone_every == 0):
                    shutil.rmtree(old)
        if not a.no_sample and (step % a.sample_every == 0 or step == a.steps):
            sample_grid(unet, te, tok, cfg["size"], out / f"samples_step{step:07d}.png")
            t0 = time.time()
    print("done")


if __name__ == "__main__":
    main()
