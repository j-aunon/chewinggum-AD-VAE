from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

from scripts.data import ChewingGumDataset
from scripts.model import ResNetVAE
from scripts import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained VAE and generate heatmaps.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved VAE checkpoint.")
    parser.add_argument("--csv", type=str, default="data/image_anno.csv")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick runs.")
    parser.add_argument("--output-dir", type=str, default="artifacts/inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold on heatmap for binary mask.")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--split-file", type=str, default=None, help="Optional split file; defaults to checkpoint metadata.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> Dict:
    ckpt = utils.load_checkpoint(ckpt_path, map_location=device)
    latent_dim = ckpt.get("args", {}).get("latent_dim", 256)
    model = ResNetVAE(in_channels=3, latent_dim=latent_dim, feature_size=8).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {"model": model, "ckpt": ckpt, "model_name": "resnet"}


def load_split_paths(args, ckpt: Dict, root: Path) -> set[str]:
    if not args.split:
        return set()
    split_file = args.split_file or ckpt.get("split_file")
    if not split_file:
        raise ValueError("Split requested but no split file found in args or checkpoint.")
    split_path = Path(split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    payload = json.loads(split_path.read_text())
    if args.split not in payload:
        raise KeyError(f"Split '{args.split}' not found in {split_path}")
    normalized = set()
    for entry in payload[args.split]:
        path = Path(entry)
        if path.is_absolute():
            normalized.add(utils.normalize_path(path, root))
        else:
            normalized.add(str(path))
    return normalized


def evaluate_masks(heatmap: np.ndarray, mask: torch.Tensor, threshold: float) -> Dict[str, float]:
    pred = (heatmap >= threshold).astype(np.uint8)
    target = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    iou = intersection / (union + 1e-8)
    return {"iou": float(iou), "intersection": int(intersection), "union": int(union)}


def run():
    args = parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device()
    loaded = load_model(Path(args.checkpoint), device)
    model = loaded["model"]
    split_paths = load_split_paths(args, loaded["ckpt"], Path(args.root))

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    dataset = ChewingGumDataset(csv_path=args.csv, root=args.root, transform=transform)
    if split_paths:
        dataset.entries = [
            entry for entry in dataset.entries if utils.normalize_path(entry["image"], Path(args.root)) in split_paths
        ]
    if args.limit:
        dataset.entries = dataset.entries[: args.limit]

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    out_dir = Path(args.output_dir)
    overlay_dir = out_dir / "overlays"
    recon_dir = out_dir / "reconstructions"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    stats: List[Dict] = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            recon, _, _ = model(imgs)
            for idx in range(imgs.size(0)):
                original = imgs[idx]
                reconstructed = recon[idx]
                heatmap = utils.reconstruction_heatmap(original, reconstructed)
                path = Path(batch["path"][idx])
                fname = path.stem

                recon_path = recon_dir / f"{fname}_recon.png"
                save_image(reconstructed, recon_path)

                np_image = np.transpose(original.cpu().numpy(), (1, 2, 0))
                utils.save_heatmap_overlay(np_image, heatmap, overlay_dir / f"{fname}_overlay.png")

                sample_stats = {
                    "image": str(path),
                    "mean_error": float(heatmap.mean()),
                    "max_error": float(heatmap.max()),
                    "label": batch["label"][idx],
                }
                if batch["has_mask"][idx]:
                    mask_stats = evaluate_masks(heatmap, batch["mask"][idx], args.threshold)
                    sample_stats.update(mask_stats)
                stats.append(sample_stats)

    utils.save_json_stats(stats, out_dir / "stats.json")
    print(f"Saved reconstructions to {recon_dir}")
    print(f"Saved heatmap overlays to {overlay_dir}")
    print(f"Saved stats to {out_dir / 'stats.json'}")


if __name__ == "__main__":
    run()
