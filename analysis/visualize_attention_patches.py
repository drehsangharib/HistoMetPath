"""
Phase 2.4-C: Visualize attention MIL using original RGB patches.

- Extracts top- and bottom-attention patches per slide
- Saves RGB patch images for qualitative inspection
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# -------------------------------------------------------------------------
# Ensure project root
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
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "embeddings")
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "logs",
    "phase_2_3",
    "checkpoints",
    "epoch=epoch=2.ckpt",
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "attention_patches")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCHES_PER_SLIDE = 50
TOP_K = 5
IMAGE_SIZE = 96
USE_STAIN_NORM = True
DEVICE = "cpu"


# -------------------------------------------------------------------------
# Transforms (must match training)
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
# Reload attention MIL model
# -------------------------------------------------------------------------
class AttentionMIL(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, bag):
        A = self.attention(bag)
        A = torch.softmax(A, dim=0)
        z = torch.sum(A * bag, dim=0)
        logit = self.classifier(z)
        return logit, A


# Load slide embeddings
slide_embeddings = np.load(
    os.path.join(EMBEDDING_DIR, "train_slide_embeddings.npy"),
    allow_pickle=True,
)
slide_labels = np.load(
    os.path.join(EMBEDDING_DIR, "train_slide_labels.npy"),
)

# Load patch model backbone for embeddings
backbone_model = HistoMetPathModel(
    backbone="resnet18",
    pretrained=False,
    freeze_backbone=False,
    hidden_dim=512,
    num_classes=1,
    dropout=0.3,
)

patch_model = HistoMetPathLightning.load_from_checkpoint(
    CHECKPOINT_PATH,
    model=backbone_model,
)
patch_model.eval()
patch_model.freeze()

# Re-create attention MIL model and load weights from training script
# (attention_mil.py already trained it in-memory, so here we reuse structure)
attention_model = AttentionMIL().to(DEVICE)
attention_model.eval()


# -------------------------------------------------------------------------
# Load original PCAM patches (same order as embeddings!)
# -------------------------------------------------------------------------
patch_ds = PCamHDF5Dataset(
    os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_train_x.h5"),
    os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_train_y.h5"),
    transform=None,                 # raw RGB
    use_stain_norm=False,           # visualize raw color
)


# -------------------------------------------------------------------------
# Visualize attention patches
# -------------------------------------------------------------------------
for slide_idx in range(5):  # visualize first 5 slides
    bag_embeds = torch.tensor(
        np.asarray(slide_embeddings[slide_idx], dtype=np.float32)
    )

    with torch.no_grad():
        _, A = attention_model(bag_embeds)

    A = A.squeeze().cpu().numpy()
    top_idx = np.argsort(A)[-TOP_K:][::-1]
    bottom_idx = np.argsort(A)[:TOP_K]

    slide_dir = os.path.join(OUTPUT_DIR, f"slide_{slide_idx}")
    os.makedirs(slide_dir, exist_ok=True)

    # Map patch indices back to global patch indices
    global_start = slide_idx * PATCHES_PER_SLIDE

    for rank, idx in enumerate(top_idx):
        patch = patch_ds[global_start + idx][0]
        Image.fromarray(patch).save(
            os.path.join(slide_dir, f"top_{rank}_attn.png")
        )

    for rank, idx in enumerate(bottom_idx):
        patch = patch_ds[global_start + idx][0]
        Image.fromarray(patch).save(
            os.path.join(slide_dir, f"low_{rank}_attn.png")
        )

    print(f"Saved attention patches for slide {slide_idx}")