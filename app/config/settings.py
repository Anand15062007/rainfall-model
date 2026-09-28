"""Configuration loading and reproducibility helpers."""
from pathlib import Path
import random
import numpy as np
import yaml


def load_config(path="configs/default.yaml"):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
