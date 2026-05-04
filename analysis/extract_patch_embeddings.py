"""
Phase 2.4-A.1: Patch embedding extraction for MIL.

- Uses frozen Phase 2.3 patch model
- Extracts 512-D embeddings
- CPU-safe
- Saves embeddings to disk
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# -------------------------------------------------------------------------
# Ensure project root on PYTHONPATH
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import PCamHDF5Dataset
from models import HistoMetPathModel
from training import HistoMetPathLightning


# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
PCAM_ROOT = os.path.join(PROJECT_ROOT, "data", "pcam")
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "logs",
    "phase_2_3",
    "checkpoints",
    "epoch=epoch=2.ckpt",
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "embeddings")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
USE_STAIN_NORM = True
IMAGE_SIZE = 96


# -------------------------------------------------------------------------
# Transforms (must match Phase 2.3)
# -------------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# -------------------------------------------------------------------------
# Load backbone model (feature extractor)
# -------------------------------------------------------------------------
backbone_model = HistoMetPathModel(
    backbone="resnet18",
    pretrained=False,
    freeze_backbone=False,
    hidden_dim=512,
    num_classes=1,
    dropout=0.3,
)

lightning_model = HistoMetPathLightning.load_from_checkpoint(
    CHECKPOINT_PATH,
    model=backbone_model,
)

# 🔒 Freeze everything
lightning_model.eval()
lightning_model.freeze()

# Remove classifier head → feature extractor
feature_extractor = lightning_model.model.backbone


# -------------------------------------------------------------------------
# Load dataset (example: TRAIN split)
# -------------------------------------------------------------------------
dataset = PCamHDF5Dataset(
    os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_train_x.h5"),
    os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_train_y.h5"),
    transform=transform,
    use_stain_norm=USE_STAIN_NORM,
)

# CPU-safe subset for demo; remove Subset for full extraction
dataset = Subset(dataset, range(20_000))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# -------------------------------------------------------------------------
# Extract embeddings
# -------------------------------------------------------------------------
all_embeddings = []
all_labels = []

with torch.no_grad():
    for i, (x, y) in enumerate(loader):
        feats = feature_extractor(x)
        feats = feats.view(feats.size(0), -1)  # flatten
        all_embeddings.append(feats.cpu().numpy())
        all_labels.append(y.numpy())

        if i % 50 == 0:
            print(f"Processed batch {i}/{len(loader)}")

all_embeddings = np.concatenate(all_embeddings)
all_labels = np.concatenate(all_labels)

print("Final embedding shape:", all_embeddings.shape)

# -------------------------------------------------------------------------
# Save
# -------------------------------------------------------------------------
np.save(os.path.join(OUTPUT_DIR, "train_embeddings.npy"), all_embeddings)
np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), all_labels)

print("✅ Embeddings saved to:", OUTPUT_DIR)