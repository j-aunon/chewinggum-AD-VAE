from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: Optional[str] = None):
    return torch.load(path, map_location=map_location)


def reconstruction_heatmap(input_tensor: torch.Tensor, recon_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        diff = torch.abs(input_tensor - recon_tensor).mean(dim=0, keepdim=True)  # 1 x H x W
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    return diff.squeeze().cpu().numpy()


def save_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, out_path: Path, cmap: str = "inferno"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(heatmap, cmap=cmap, alpha=0.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_json_stats(stats: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
