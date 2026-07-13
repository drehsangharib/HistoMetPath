import numpy as np

from analysis.run_camelyon16_real_wsi_mil_pilot import (
    calculate_metrics,
    select_threshold,
)


def test_validation_threshold_selection_is_deterministic():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    first = select_threshold(labels, probabilities)
    second = select_threshold(labels, probabilities)
    assert first == second
    assert first["validation_balanced_accuracy"] == 1.0
    assert 0.0 <= first["threshold"] <= 1.0


def test_small_test_metrics_schema():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.3, 0.7, 0.8])
    metrics = calculate_metrics(labels, probabilities, threshold=0.5)
    assert metrics["sample_count"] == 4
    assert metrics["auroc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
