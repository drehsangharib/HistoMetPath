import torch
from analysis.attention_mil_v2 import AttentionMIL

def test_attention_model_smoke():
    model = AttentionMIL(
        in_dim=512,
        hidden_dim=128,
    )

    bag = torch.rand(50, 512)

    logit, attention = model(bag)

    assert attention.shape[0] == 50
    assert torch.isfinite(logit)
