"""
Threshold calibration utilities.

Thresholds must be selected on validation data only.
"""

import numpy as np


def select_threshold(
    y_true,
    y_prob,
    metric="balanced_accuracy",
):
    """
    Select an optimal decision threshold on validation data.

    Args:
        y_true (array-like): Ground truth labels (0/1)
        y_prob (array-like): Predicted probabilities
        metric (str): Optimization metric

    Returns:
        float: Selected threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thresholds = np.linspace(0.01, 0.99, 99)

    best_score = -np.inf
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "balanced_accuracy":
            tpr = tp / (tp + fn + 1e-8)
            tnr = tn / (tn + fp + 1e-8)
            score = 0.5 * (tpr + tnr)
        else:
            score = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        if score > best_score:
            best_score = score
            best_threshold = t

    return float(best_threshold)
