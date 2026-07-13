import numpy as np
from analysis.mil_models import mean_pool, max_pool

def test_mean_max_pool():
    bag = np.array([[1,2],[3,4]])
    assert mean_pool(bag).tolist() == [2,3]
    assert max_pool(bag).tolist() == [3,4]
