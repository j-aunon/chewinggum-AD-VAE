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
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from scripts.data import ChewingGumDataset
from scripts.model import ImprovedVAE, VGGPerceptualLoss
from scripts import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train improved VAE for chewing gum anomaly detection.")
    parser.add_argument("--csv", type=str, default="data/image_anno.csv", help="Path to annotation CSV.")
    parser.add_argument("--root", type=str, default=".", help="Project root containing data folder.")
    parser.add_argument("--epochs", type=int, default=100, help="Increased from 20 to 100")
    parser.add_argument("--batch-size", type=int, default=16, help="Increased from 8 to 16")
    parser.add_argument("--lr", type=float, default=2e-4, help="Reduced learning rate")
    parser.add_argument("--latent-dim", type=int, default=512, help="Increased from 256 to 512")
    parser.add_argument("--feature-size", type=int, default=16, help="Increased from 8 to 16")
    parser.add_argument("--beta", type=float, default=0.05, help="REDUCED from 1.0 to 0.05 - critical fix")
    parser.add_argument("--beta-warmup", type=int, default=20, help="Epochs to warm up beta")
    parser.add_argument("--ssim-weight", type=float, default=0.2, help="Increased SSIM weight")
    parser.add_argument("--perceptual-weight", type=float, default=0.1, help="Add perceptual loss")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="artifacts/checkpoints_improved")
    parser.add_argument("--split-file", type=str, default="artifacts/splits_improved.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-dir", type=str, default="artifacts/tensorboard_improved")
    parser.add_argument("--log-images", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--use-augmentation", action="store_true", help="Use data augmentation")
    return parser.parse_args()


def get_transforms(augment: bool = False):
    """Get training transforms with optional augmentation"""
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform


def build_dataloaders(args):
    train_transform, val_transform = get_transforms(args.use_augmentation)
    
    # Training dataset with augmentation
    train_dataset = ChewingGumDataset(csv_path=args.csv, root=args.root, transform=train_transform, only_normal=False)
    # Validation dataset without augmentation
    val_dataset = ChewingGumDataset(csv_path=args.csv, root=args.root, transform=val_transform, only_normal=False)
    
    normal_indices = [i for i, entry in enumerate(train_dataset.entries) if entry["label"] == "normal"]
    defect_indices = [i for i, entry in enumerate(train_dataset.entries) if entry["label"] != "normal"]
    
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
        "train": [utils.normalize_path(train_dataset.entries[i]["image"], Path(args.root)) for i in train_indices],
        "val": [utils.normalize_path(val_dataset.entries[i]["image"], Path(args.root)) for i in val_indices],
        "test": [utils.normalize_path(train_dataset.entries[i]["image"], Path(args.root)) for i in test_indices],
        "test_normals": [utils.normalize_path(train_dataset.entries[i]["image"], Path(args.root)) for i in test_normal_indices],
        "test_defects": [utils.normalize_path(train_dataset.entries[i]["image"], Path(args.root)) for i in defect_indices],
    }
    utils.save_json_stats(split_payload, Path(args.split_file))

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader


def get_beta_for_epoch(epoch: int, final_beta: float, warmup_epochs: int) -> float:
    """Gradually increase beta during warmup"""
    if warmup_epochs == 0:
        return final_beta
    return min(final_beta, final_beta * epoch / warmup_epochs)


def train():
    args = parse_args()
    utils.set_seed(args.seed)
    prefer_cuda = args.device != "cpu"
    device = utils.get_device(prefer_cuda=prefer_cuda)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but not available. Use --device cpu or install CUDA.")
    
    print(f"Using device: {device}")
    print(f"Configuration:")
    print(f"  - Latent dim: {args.latent_dim} (was 256)")
    print(f"  - Feature size: {args.feature_size} (was 8)")
    print(f"  - Beta: {args.beta} (was 1.0) with {args.beta_warmup} epoch warmup")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - SSIM weight: {args.ssim_weight}")
    print(f"  - Perceptual weight: {args.perceptual_weight}")
    print(f"  - Data augmentation: {args.use_augmentation}")
    
    train_loader, val_loader = build_dataloaders(args)

    model = ImprovedVAE(in_channels=3, latent_dim=args.latent_dim, feature_size=args.feature_size).to(device)
    
    # Add perceptual loss if requested
    vgg_loss = None
    if args.perceptual_weight > 0:
        try:
            vgg_loss = VGGPerceptualLoss().to(device)
            print("  - Perceptual loss: enabled")
        except Exception as e:
            print(f"  - Perceptual loss: disabled (error: {e})")
            args.perceptual_weight = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(1, args.epochs + 1):
        # Beta warmup
        current_beta = get_beta_for_epoch(epoch, args.beta, args.beta_warmup)
        
        model.train()
        total_loss = total_recon = total_kld = total_ssim = total_ssim_loss = total_perceptual = 0.0
        
        for batch in train_loader:
            imgs = batch["image"].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            
            # Base loss
            loss, recon_loss, kld, perceptual_loss = model.loss_function(
                recon, imgs, mu, logvar, 
                beta=current_beta,
                perceptual_weight=args.perceptual_weight,
                vgg_features=vgg_loss
            )
            
            # SSIM loss
            ssim_val = utils.ssim(recon, imgs).mean()
            ssim_loss = 1.0 - ssim_val
            if args.ssim_weight > 0:
                loss = loss + args.ssim_weight * ssim_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            total_recon += recon_loss.item() * imgs.size(0)
            total_kld += kld.item() * imgs.size(0)
            total_ssim += ssim_val.item() * imgs.size(0)
            total_ssim_loss += ssim_loss.item() * imgs.size(0)
            total_perceptual += perceptual_loss.item() * imgs.size(0)
        
        train_loss = total_loss / len(train_loader.dataset)
        train_recon = total_recon / len(train_loader.dataset)
        train_kld = total_kld / len(train_loader.dataset)
        train_ssim = total_ssim / len(train_loader.dataset)
        train_ssim_loss = total_ssim_loss / len(train_loader.dataset)
        train_perceptual = total_perceptual / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = val_recon = val_kld = val_ssim = val_ssim_loss = val_perceptual = 0.0
        sample_batch = None
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                recon, mu, logvar = model(imgs)
                
                loss, r, k, p = model.loss_function(
                    recon, imgs, mu, logvar, 
                    beta=current_beta,
                    perceptual_weight=args.perceptual_weight,
                    vgg_features=vgg_loss
                )
                
                ssim_val = utils.ssim(recon, imgs).mean()
                ssim_loss = 1.0 - ssim_val
                if args.ssim_weight > 0:
                    loss = loss + args.ssim_weight * ssim_loss
                
                val_loss += loss.item() * imgs.size(0)
                val_recon += r.item() * imgs.size(0)
                val_kld += k.item() * imgs.size(0)
                val_ssim += ssim_val.item() * imgs.size(0)
                val_ssim_loss += ssim_loss.item() * imgs.size(0)
                val_perceptual += p.item() * imgs.size(0)
                
                if sample_batch is None:
                    sample_batch = (imgs.detach().cpu(), recon.detach().cpu())

        val_loss /= len(val_loader.dataset)
        val_recon /= len(val_loader.dataset)
        val_kld /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)
        val_ssim_loss /= len(val_loader.dataset)
        val_perceptual /= len(val_loader.dataset)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Tensorboard logging
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("recon/train", train_recon, epoch)
        writer.add_scalar("recon/val", val_recon, epoch)
        writer.add_scalar("kld/train", train_kld, epoch)
        writer.add_scalar("kld/val", val_kld, epoch)
        writer.add_scalar("ssim/train", train_ssim, epoch)
        writer.add_scalar("ssim/val", val_ssim, epoch)
        writer.add_scalar("ssim_loss/train", train_ssim_loss, epoch)
        writer.add_scalar("ssim_loss/val", val_ssim_loss, epoch)
        writer.add_scalar("perceptual/train", train_perceptual, epoch)
        writer.add_scalar("perceptual/val", val_perceptual, epoch)
        writer.add_scalar("beta", current_beta, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
        if sample_batch is not None and args.log_images > 0:
            imgs, recon = sample_batch
            count = min(args.log_images, imgs.size(0))
            grid = make_grid(
                torch.cat([imgs[:count], recon[:count]], dim=0),
                nrow=count,
                padding=2,
                pad_value=1.0,
            )
            writer.add_image("val/original_top_recon_bottom", grid, epoch)

        print(
            f"Epoch {epoch:03d} | β={current_beta:.4f} | Train loss {train_loss:.4f} "
            f"(recon {train_recon:.4f}, kld {train_kld:.4f}, ssim {train_ssim:.4f}) "
            f"| Val loss {val_loss:.4f} (recon {val_recon:.4f}, kld {val_kld:.4f}, ssim {val_ssim:.4f})"
        )

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": vars(args),
                "val_loss": val_loss,
                "val_ssim": val_ssim,
                "split_file": str(Path(args.split_file)),
            }
            out_path = save_dir / f"vae_best_epoch{epoch:03d}_ssim{val_ssim:.4f}.pt"
            utils.save_checkpoint(checkpoint, out_path)
            print(f"✓ Saved BEST checkpoint to {out_path}")
        
        # Save periodic checkpoints
        if epoch % args.save_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": vars(args),
                "val_loss": val_loss,
                "val_ssim": val_ssim,
                "split_file": str(Path(args.split_file)),
            }
            out_path = save_dir / f"vae_epoch{epoch:03d}.pt"
            utils.save_checkpoint(checkpoint, out_path)
    
    writer.close()
    print(f"\n✓ Training complete! Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    train()