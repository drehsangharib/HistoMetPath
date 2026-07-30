import numpy as np

from analysis.run_camelyon16_training_only_stability import (
    mean_pairwise_cosine,
    metrics,
)


def test_metrics_use_fixed_threshold():
    result = metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9]),
        0.5,
    )
    assert result["balanced_accuracy"] == 1.0
    assert result["accuracy"] == 1.0


def test_coefficient_cosine_stability():
    vectors = [np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])]
    assert mean_pairwise_cosine(vectors) == 1.0
