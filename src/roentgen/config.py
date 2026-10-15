"""Single source of truth for paths, labels and Stage-1 settings."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("IE643_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = Path(os.environ.get("CXR_DATA_ROOT", ROOT / "data"))
ARTIFACTS = ROOT / "artifacts"          # manifests, caches, checkpoints (all inside the project folder)
MANIFEST = ARTIFACTS / "manifest.parquet"
MANIFEST_META = ARTIFACTS / "manifest_meta.json"
CACHE_DIR = ARTIFACTS / "cache"

SD_MODEL = "CompVis/stable-diffusion-v1-4"

TARGETS = [
    "Pleural Other",
    "Pneumonia",
    "Lung Lesion",
    "Fracture",
    "Enlarged Cardiomediastinum",
]
NORMAL = "No Finding"
LABELS = TARGETS + [NORMAL]   # column order of every label vector in the project

SEED = 643
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Diffusion training subset: cap each label combination so "No Finding" and the
# label-free background pool don't swamp the five rare pathologies.
COMBO_CAP = 8000


def find_chexpert_dir() -> Path:
    """Locate the folder holding train.csv / valid.csv (Kaggle unzips vary in nesting)."""
    for p in [DATA_ROOT / "CheXpert-v1.0-small", DATA_ROOT]:
        if (p / "train.csv").exists():
            return p
    hits = sorted(DATA_ROOT.rglob("train.csv"))
    if not hits:
        raise FileNotFoundError(f"train.csv not found under {DATA_ROOT}")
    return hits[0].parent
