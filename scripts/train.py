from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from scripts.data import ChewingGumDataset
from scripts.model import SimpleVAE
from scripts import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train a baseline VAE for chewing gum anomaly detection.")
    parser.add_argument("--csv", type=str, default="data/image_anno.csv", help="Path to annotation CSV.")
    parser.add_argument("--root", type=str, default=".", help="Project root containing data folder.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for KL term.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of training normals to use for validation.")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of normals to keep for test split.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="artifacts/checkpoints")
    parser.add_argument("--split-file", type=str, default="artifacts/splits.json", help="Where to save train/val/test splits.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloaders(args):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    dataset = ChewingGumDataset(csv_path=args.csv, root=args.root, transform=transform, only_normal=False)
    normal_indices = [i for i, entry in enumerate(dataset.entries) if entry["label"] == "normal"]
    defect_indices = [i for i, entry in enumerate(dataset.entries) if entry["label"] != "normal"]
    if len(normal_indices) < 3:
        raise ValueError("Not enough normal samples to create train/val/test splits.")
    rng = random.Random(args.seed)
    rng.shuffle(normal_indices)
    val_size = max(1, int(len(normal_indices) * args.val_fraction))
    test_size = max(1, int(len(normal_indices) * args.test_fraction))
    train_size = len(normal_indices) - val_size - test_size
    if train_size < 1:
        raise ValueError("Split sizes leave no samples for training; reduce val/test fractions.")
    train_indices = normal_indices[:train_size]
    val_indices = normal_indices[train_size : train_size + val_size]
    test_normal_indices = normal_indices[train_size + val_size :]
    test_indices = sorted(test_normal_indices + defect_indices)

    split_payload = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "train": [utils.normalize_path(dataset.entries[i]["image"], Path(args.root)) for i in train_indices],
        "val": [utils.normalize_path(dataset.entries[i]["image"], Path(args.root)) for i in val_indices],
        "test": [utils.normalize_path(dataset.entries[i]["image"], Path(args.root)) for i in test_indices],
        "test_normals": [utils.normalize_path(dataset.entries[i]["image"], Path(args.root)) for i in test_normal_indices],
        "test_defects": [utils.normalize_path(dataset.entries[i]["image"], Path(args.root)) for i in defect_indices],
    }
    utils.save_json_stats(split_payload, Path(args.split_file))

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader


def train():
    args = parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device()
    train_loader, val_loader = build_dataloaders(args)

    model = SimpleVAE(in_channels=3, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_recon = total_kld = total_ssim = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss, recon_loss, kld = model.loss_function(recon, imgs, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total_recon += recon_loss.item() * imgs.size(0)
            total_kld += kld.item() * imgs.size(0)
            total_ssim += utils.ssim(recon, imgs).mean().item() * imgs.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        train_recon = total_recon / len(train_loader.dataset)
        train_kld = total_kld / len(train_loader.dataset)
        train_ssim = total_ssim / len(train_loader.dataset)

        model.eval()
        val_loss = val_recon = val_kld = val_ssim = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                recon, mu, logvar = model(imgs)
                loss, r, k = model.loss_function(recon, imgs, mu, logvar, beta=args.beta)
                val_loss += loss.item() * imgs.size(0)
                val_recon += r.item() * imgs.size(0)
                val_kld += k.item() * imgs.size(0)
                val_ssim += utils.ssim(recon, imgs).mean().item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_recon /= len(val_loader.dataset)
        val_kld /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)

        print(
            f"Epoch {epoch:03d} | Train loss {train_loss:.4f} (recon {train_recon:.4f}, kld {train_kld:.4f}, ssim {train_ssim:.4f}) "
            f"| Val loss {val_loss:.4f} (recon {val_recon:.4f}, kld {val_kld:.4f}, ssim {val_ssim:.4f})"
        )

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "val_loss": val_loss,
                "split_file": str(Path(args.split_file)),
            }
            out_path = save_dir / f"vae_epoch{epoch:03d}_val{val_loss:.4f}.pt"
            utils.save_checkpoint(checkpoint, out_path)
            print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    train()
