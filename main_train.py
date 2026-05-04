"""
Main Training Script for HistoMetPath (CPU-safe).

Phase 2.3:
- Stain normalization (Macenko, toggleable)
- Histopathology-aware modeling
- CPU-safe subset execution
- Resume-safe (logging + checkpoints)
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# -------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH (script mode)
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import PCamHDF5Dataset
from models import HistoMetPathModel
from training import HistoMetPathLightning, create_trainer


# -------------------------------------------------------------------------
# ✅ Phase 2.3 toggle
# -------------------------------------------------------------------------
USE_STAIN_NORM = True   # False = Phase 2.2 baseline, True = Phase 2.3


# -------------------------------------------------------------------------
# ✅ PCAM DATA ROOT (FIXED)
# -------------------------------------------------------------------------
# 🔹 CHANGE THIS if your data lives elsewhere
PCAM_ROOT = os.path.join(PROJECT_ROOT, "data", "pcam")


# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HistoMetPath model on PatchCamelyon (CPU-safe)"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


# -------------------------------------------------------------------------
# Reproducibility (CPU)
# -------------------------------------------------------------------------
def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------------------------------------------------------
# Transforms (PCAM-compatible)
# -------------------------------------------------------------------------
def get_transforms(image_size: int = 96):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 70)
    print("HistoMetPath — Phase 2.3 (Stain Normalization)")
    print("=" * 70)
    print(f"Arguments: {args}")
    print(f"Stain normalization: {USE_STAIN_NORM}")
    print(f"PCAM root: {PCAM_ROOT}")

    set_seed(args.seed)
    print(f"\nRandom seed set to: {args.seed}")

    transform = get_transforms()
    print("\nTransforms created (PCAM canonical)")

    # ---------------------------------------------------------------------
    # ✅ Verify PCAM files exist (FAIL FAST)
    # ---------------------------------------------------------------------
    required_files = [
        "camelyonpatch_level_2_split_train_x.h5",
        "camelyonpatch_level_2_split_train_y.h5",
        "camelyonpatch_level_2_split_valid_x.h5",
        "camelyonpatch_level_2_split_valid_y.h5",
        "camelyonpatch_level_2_split_test_x.h5",
        "camelyonpatch_level_2_split_test_y.h5",
    ]

    for fname in required_files:
        fpath = os.path.join(PCAM_ROOT, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"❌ Missing PCAM file: {fpath}")

    # ---------------------------------------------------------------------
    # Load PCAM datasets
    # ---------------------------------------------------------------------
    print("\nLoading PatchCamelyon datasets")

    train_ds = PCamHDF5Dataset(
        os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_train_x.h5"),
        os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_train_y.h5"),
        transform,
        use_stain_norm=USE_STAIN_NORM,
    )

    val_ds = PCamHDF5Dataset(
        os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_valid_x.h5"),
        os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_valid_y.h5"),
        transform,
        use_stain_norm=USE_STAIN_NORM,
    )

    test_ds = PCamHDF5Dataset(
        os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_test_x.h5"),
        os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_test_y.h5"),
        transform,
        use_stain_norm=USE_STAIN_NORM,
    )

    print(f"  Full train samples: {len(train_ds)}")
    print(f"  Full val samples:   {len(val_ds)}")
    print(f"  Full test samples:  {len(test_ds)}")

    # ---------------------------------------------------------------------
    # ✅ CPU-safe subset mode
    # ---------------------------------------------------------------------
    train_ds = Subset(train_ds, range(20_000))
    val_ds   = Subset(val_ds, range(5_000))
    test_ds  = Subset(test_ds, range(5_000))

    print("\nUsing CPU-safe subsets:")
    print(f"  Train subset: {len(train_ds)}")
    print(f"  Val subset:   {len(val_ds)}")
    print(f"  Test subset:  {len(test_ds)}")

    assert len(train_ds) > 0, "Train dataset is empty"
    assert len(val_ds) > 0, "Validation dataset is empty"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    model = HistoMetPathModel(
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        hidden_dim=512,
        num_classes=1,
        dropout=0.3,
    )

    lightning_model = HistoMetPathLightning(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=None,
    )

    # ---------------------------------------------------------------------
    # Trainer (with logging + resume)
    # ---------------------------------------------------------------------
    trainer = create_trainer(max_epochs=args.epochs)

    ckpt_path = os.path.join("logs", "phase_2_3", "checkpoints", "last.ckpt")
    resume_ckpt = ckpt_path if os.path.exists(ckpt_path) else None

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    print("\n🚀 Starting Phase 2.3 training...\n")
    trainer.fit(
        lightning_model,
        train_loader,
        val_loader,
        ckpt_path=resume_ckpt,
    )

    # ---------------------------------------------------------------------
    # Testing
    # ---------------------------------------------------------------------
    print("\n🔬 Running test evaluation...\n")
    trainer.test(lightning_model, test_loader, ckpt_path="best")

    print("\n✅ Phase 2.3 Training Complete")


if __name__ == "__main__":
    main()
