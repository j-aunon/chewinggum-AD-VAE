from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from matplotlib import cm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_rocm_build() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _ensure_nvidia_cuda() -> None:
    if _is_rocm_build():
        raise RuntimeError(
            "ROCm/HIP build of PyTorch detected. This project assumes an NVIDIA CUDA PyTorch build."
        )


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        _ensure_nvidia_cuda()
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: torch.device | str | None = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_json_stats(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def normalize_path(path: Path | str, root: Path) -> str:
    path_obj = Path(path)
    root_obj = Path(root)
    try:
        rel = path_obj.resolve().relative_to(root_obj.resolve())
        return rel.as_posix()
    except Exception:
        return path_obj.as_posix()


def reconstruction_heatmap(original: torch.Tensor, reconstructed: torch.Tensor) -> np.ndarray:
    error = torch.abs(reconstructed - original).mean(dim=0)
    return error.detach().cpu().numpy()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError("SSIM expects img1 and img2 to have the same shape.")
    if img1.dim() != 4:
        raise ValueError("SSIM expects tensors with shape (N, C, H, W).")

    pad = window_size // 2
    mu1 = F.avg_pool2d(img1, window_size, 1, pad)
    mu2 = F.avg_pool2d(img2, window_size, 1, pad)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, pad) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, pad) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)

    per_channel = ssim_map.flatten(2).mean(dim=2)
    return per_channel.mean(dim=1)


def save_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, path: Path, alpha: float = 0.5) -> None:
    if image.max() > 1.0:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    heat = heatmap.astype(np.float32)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    colored = cm.get_cmap("jet")(heat)[..., :3]

    overlay = (1 - alpha) * img + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
