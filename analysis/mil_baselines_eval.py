"""
Phase 2.4-B.1-eval: Proper MIL baseline evaluation
with train/validation split.

- Mean pooling
- Max pooling
- Slide-level evaluation on held-out slides
"""

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------
# Ensure project root on PYTHONPATH
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "embeddings")
THRESHOLD = 0.15
TEST_SIZE = 0.30
RANDOM_STATE = 42


# -------------------------------------------------------------------------
# Load pseudo-slides
# -------------------------------------------------------------------------
slide_embeddings = np.load(
    os.path.join(EMBEDDING_DIR, "train_slide_embeddings.npy"),
    allow_pickle=True,
)
slide_labels = np.load(
    os.path.join(EMBEDDING_DIR, "train_slide_labels.npy"),
)

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
# Train / validation split (STRATIFIED)
# -------------------------------------------------------------------------
X_mean_tr, X_mean_val, y_tr, y_val = train_test_split(
    X_mean, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_max_tr, X_max_val, _, _ = train_test_split(
    X_max, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)


# -------------------------------------------------------------------------
# Evaluation function
# -------------------------------------------------------------------------
def evaluate(X_tr, y_tr, X_val, y_val, name):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)

    probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)

    preds = (probs >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n{name} pooling (validation):")
    print(f"  AUC: {auc:.3f}")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")


# -------------------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------------------
evaluate(X_mean_tr, y_tr, X_mean_val, y_val, "Mean")
evaluate(X_max_tr,  y_tr, X_max_val,  y_val, "Max")