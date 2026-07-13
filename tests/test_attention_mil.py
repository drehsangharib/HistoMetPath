import torch
from analysis.attention_mil_v2 import AttentionMIL

def test_attention_forward():
    model = AttentionMIL(in_dim=4, hidden_dim=2)
    bag = torch.randn(10, 4)
    logit, attn = model(bag)
    assert attn.shape[0] == 10
