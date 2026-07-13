import numpy as np
import torch

from analysis.attention_mil_v2 import AttentionMIL
from core.wsi.generate_real_slide_attention_maps import (
    attention_entropy,
    score_slide_bag,
)


def test_real_slide_attention_alignment():
    torch.manual_seed(7)
    model = AttentionMIL(in_dim=8, hidden_dim=4)
    embeddings = np.random.default_rng(7).normal(size=(13, 8)).astype(np.float32)
    probability, weights = score_slide_bag(model, embeddings, torch.device("cpu"))
    assert 0.0 <= probability <= 1.0
    assert weights.shape == (13,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    assert 0.0 <= attention_entropy(weights) <= np.log(13)
