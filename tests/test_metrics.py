import numpy as np
from analysis.mil_models import mean_pool

def test_metrics_smoke():
    bag = np.random.randn(10, 5)
    pooled = mean_pool(bag)
    assert pooled.shape == (5,)
