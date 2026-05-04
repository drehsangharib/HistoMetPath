"""
Phase 2.4-B.1: Simple MIL baselines (mean / max pooling).

- Uses pseudo-slide embeddings
- Trains simple slide-level classifiers
- Reports slide-level metrics
"""

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

# -------------------------------------------------------------------------
# ✅ Ensure project root on PYTHONPATH
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "embeddings")
THRESHOLD = 0.15   # locked from Phase 2.3


# -------------------------------------------------------------------------
# Load pseudo-slides (ABSOLUTE PATHS)
# -------------------------------------------------------------------------
slide_embeddings_path = os.path.join(EMBEDDING_DIR, "train_slide_embeddings.npy")
slide_labels_path = os.path.join(EMBEDDING_DIR, "train_slide_labels.npy")

if not os.path.exists(slide_embeddings_path):
    raise FileNotFoundError(f"Missing file: {slide_embeddings_path}")
if not os.path.exists(slide_labels_path):
    raise FileNotFoundError(f"Missing file: {slide_labels_path}")

slide_embeddings = np.load(slide_embeddings_path, allow_pickle=True)
slide_labels = np.load(slide_labels_path)

num_slides, num_patches, emb_dim = slide_embeddings.shape
print(f"Slides: {num_slides}, Patches/slide: {num_patches}, Dim: {emb_dim}")


# -------------------------------------------------------------------------
# Pooling functions
# -------------------------------------------------------------------------
def mean_pool(bag):
    return bag.mean(axis=0)

def max_pool(bag):
    return bag.max(axis=0)


# -------------------------------------------------------------------------
# Build slide-level representations
# -------------------------------------------------------------------------
X_mean = np.stack([mean_pool(bag) for bag in slide_embeddings])
X_max  = np.stack([max_pool(bag)  for bag in slide_embeddings])
y = slide_labels


# -------------------------------------------------------------------------
# Train + evaluate
# -------------------------------------------------------------------------
def evaluate(X, y, name):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)

    preds = (probs >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n{name} pooling:")
    print(f"  AUC: {auc:.3f}")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")


# -------------------------------------------------------------------------
# Run baselines
# -------------------------------------------------------------------------
evaluate(X_mean, y, "Mean")
evaluate(X_max, y, "Max")