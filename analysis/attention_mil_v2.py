"""
Reusable Attention MIL implementation (Ilse et al., 2018).
No side effects at import time.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, bag):
        # bag: [N, D]
        A = self.attention(bag)           # [N, 1]
        A = torch.softmax(A, dim=0)
        z = torch.sum(A * bag, dim=0)
        logit = self.classifier(z)
        return logit.squeeze(), A.squeeze()
