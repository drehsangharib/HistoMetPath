import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, confusion_matrix


def compute_binary_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute medical-grade binary classification metrics.

    Args:
        targets: Ground truth labels (0 or 1), shape (N,)
        probs: Predicted probabilities for class 1, shape (N,)
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    preds = (probs >= threshold).astype(int)

    acc = (preds == targets).mean()

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    sensitivity = tp / (tp + fn + 1e-8)   # Recall
    specificity = tn / (tn + fp + 1e-8)

    # ROC-AUC (guard against single-class edge case)
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }