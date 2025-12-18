from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _resolve_path(root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    candidate = root / rel
    if candidate.exists():
        return candidate
    # CSV entries use "chewinggum/..." while data lives in "data/...".
    parts = rel.parts
    if parts and parts[0] == "chewinggum":
        candidate = root / Path(*parts[1:])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve path for {rel_path} under {root}")


class ChewingGumDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root: str = ".",
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        only_normal: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        self.entries = []
        with self.csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if only_normal and row["label"].strip() != "normal":
                    continue
                if split == "train" and row["label"].strip() != "normal":
                    # Train on normals for unsupervised AD.
                    continue
                img_path = _resolve_path(self.root, row["image"])
                mask_path = row["mask"].strip()
                mask_resolved = _resolve_path(self.root, mask_path) if mask_path else None
                self.entries.append({"image": img_path, "mask": mask_resolved, "label": row["label"].strip()})

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry["image"]).convert("RGB")
        image = self.transform(image)
        mask_available = False
        if entry["mask"]:
            mask_img = Image.open(entry["mask"]).convert("L")
            mask_img = transforms.Resize((image.shape[1], image.shape[2]))(mask_img)
            mask = transforms.ToTensor()(mask_img)  # 1 x H x W in [0,1]
            mask_available = True
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2])
        return {
            "image": image,
            "mask": mask,
            "has_mask": mask_available,
            "label": entry["label"],
            "path": str(entry["image"]),
        }
