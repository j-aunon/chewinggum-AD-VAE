## ChewingGum AD VAE
A barebones PyTorch baseline for unsupervised anomaly detection on the chewing gum surface dataset using a simple VAE. Train on normal samples and flag defects via reconstruction error heatmaps.

### Quick start
- Install deps: `pip install torch torchvision pillow matplotlib numpy`
- Train (saves checkpoints under `artifacts/checkpoints`): `python scripts/train.py --csv data/image_anno.csv --root .`
- Infer (writes reconstructions, overlays, stats): `python scripts/infer.py --checkpoint <ckpt.pt> --csv data/image_anno.csv --root . --output-dir artifacts/inference`

### Data expectations
- `data/image_anno.csv` with columns `image, mask, label`; images/masks are resolved under the repo root (rows may start with `chewinggum/`).
- Masks are optional; when present they are used for IoU scoring during inference.

### What’s inside
- `scripts/train.py`: training loop with validation split on normals.
- `scripts/model.py`: lightweight conv encoder/decoder VAE.
- `scripts/infer.py`: loads a checkpoint, saves reconstructions + heatmap overlays, emits per-sample JSON stats.
