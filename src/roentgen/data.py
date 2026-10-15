"""Datasets: diffusion (512 px, [-1,1], prompt) and classifier (224 px, multilabel)."""
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from roentgen.config import LABELS, MANIFEST, TARGETS, find_chexpert_dir
from roentgen.prompts import make_prompt

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_square(path, size: int) -> Image.Image:
    """Grayscale -> centre-crop square -> resize -> 3-channel RGB."""
    im = Image.open(path).convert("L")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((size, size), Image.BICUBIC).convert("RGB")


def to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)


def read_manifest(split=None, diffusion_only=False) -> pd.DataFrame:
    df = pd.read_parquet(MANIFEST)
    if split is not None:
        df = df[df.split.isin([split] if isinstance(split, str) else split)]
    if diffusion_only:
        df = df[df.diffusion]
    return df.reset_index(drop=True)


class DiffusionCXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 512, seed: int = 0):
        self.df, self.size = df.reset_index(drop=True), size
        self.root = find_chexpert_dir()
        self.labels = (df[LABELS].fillna(0).values == 1).astype(np.float32)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = to_tensor(load_square(self.root / r.rel_path, self.size)) * 2 - 1
        return {"pixel_values": x, "prompt": make_prompt(self.labels[i], r.view, self.rng),
                "labels": torch.from_numpy(self.labels[i]), "idx": i}


class ClassifierCXRDataset(Dataset):
    """5 target labels; uncertain (NaN under policy 'ignore') returned as -1 for masking."""
    def __init__(self, df: pd.DataFrame, size: int = 224):
        self.df, self.size = df.reset_index(drop=True), size
        self.root = find_chexpert_dir()
        self.y = df[TARGETS].fillna(-1).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x = to_tensor(load_square(self.root / self.df.rel_path.iloc[i], self.size))
        return (x - IMAGENET_MEAN) / IMAGENET_STD, torch.from_numpy(self.y[i])
