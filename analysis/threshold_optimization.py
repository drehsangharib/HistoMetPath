"""
Threshold optimization for HistoMetPath (Phase 2.3).

- Uses validation predictions
- Sweeps thresholds
- Reports sensitivity / specificity trade-offs
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, roc_auc_score
from torchvision import transforms

# -------------------------------------------------------------------------
# ✅ Ensure project root is on PYTHONPATH (CRITICAL)
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
BATCH_SIZE = 32
USE_STAIN_NORM = True
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "logs",
    "phase_2_3",
    "checkpoints",
    "epoch=epoch=2.ckpt",
)

IMAGE_SIZE = 96


# -------------------------------------------------------------------------
# Transforms (MUST MATCH TRAINING)
# -------------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # converts HWC -> CHW
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def compute_metrics(y_true, y_pred_bin):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return sensitivity, specificity, accuracy


# -------------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------------
model = HistoMetPathModel(
    backbone="resnet18",
    pretrained=False,
    freeze_backbone=False,
    hidden_dim=512,
    num_classes=1,
    dropout=0.3,
)

lightning_model = HistoMetPathLightning.load_from_checkpoint(
    CHECKPOINT_PATH,
    model=model,
)

lightning_model.eval()
lightning_model.freeze()


# -------------------------------------------------------------------------
# Load validation data
# -------------------------------------------------------------------------
val_ds = PCamHDF5Dataset(
    os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_valid_x.h5"),
    os.path.join(PCAM_ROOT, "camelyonpatch_level_2_split_valid_y.h5"),
    transform=transform,
    use_stain_norm=USE_STAIN_NORM,
)

# Same subset size used in training
val_ds = Subset(val_ds, range(5_000))

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# -------------------------------------------------------------------------
# Run inference
# -------------------------------------------------------------------------
all_probs = []
all_targets = []

with torch.no_grad():
    for x, y in val_loader:
        logits = lightning_model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.append(probs)
        all_targets.append(y.numpy().ravel())

all_probs = np.concatenate(all_probs)
all_targets = np.concatenate(all_targets)

print("Validation AUC:", roc_auc_score(all_targets, all_probs))


# -------------------------------------------------------------------------
# Threshold sweep
# -------------------------------------------------------------------------
print("\nThreshold  Sensitivity  Specificity  Accuracy")
print("-" * 55)

best_threshold = None

for t in np.linspace(0.05, 0.95, 19):
    preds = (all_probs >= t).astype(int)
    sens, spec, acc = compute_metrics(all_targets, preds)

    print(f"{t:0.2f}        {sens:0.3f}        {spec:0.3f}        {acc:0.3f}")

    if sens >= 0.80 and best_threshold is None:
        best_threshold = t

if best_threshold is not None:
    print(f"\n✅ Recommended threshold for sensitivity ≥ 0.80: {best_threshold:.2f}")
else:
    print("\n⚠️ No threshold achieved sensitivity ≥ 0.80")