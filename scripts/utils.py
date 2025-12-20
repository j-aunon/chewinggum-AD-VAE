from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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


def normalize_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _gaussian_window(window_size: int, sigma: float, channel: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma**2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError("SSIM expects img1 and img2 with the same shape.")
    channel = img1.size(1)
    window = _gaussian_window(window_size, sigma, channel, img1.device, img1.dtype)
    padding = window_size // 2
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))
