import numpy as np
from analysis.threshold_calibration import select_threshold

def test_threshold_calibration():
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    t = select_threshold(probs, labels)
    assert 0.0 <= t <= 1.0
