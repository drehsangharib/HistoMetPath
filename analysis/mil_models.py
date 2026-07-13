"""
Unified MIL models: mean, max, attention.
"""
from __future__ import annotations
import numpy as np
import torch
from .attention_mil_v2 import AttentionMIL

def mean_pool(bag: np.ndarray) -> np.ndarray:
    return bag.mean(axis=0)

def max_pool(bag: np.ndarray) -> np.ndarray:
    return bag.max(axis=0)

class MeanMIL:
    def __call__(self, bag: np.ndarray) -> np.ndarray:
        return mean_pool(bag)

class MaxMIL:
    def __call__(self, bag: np.ndarray) -> np.ndarray:
        return max_pool(bag)

class AttentionMILWrapper:
    def __init__(self, in_dim: int = 512):
        self.model = AttentionMIL(in_dim=in_dim)

    def forward(self, bag_tensor: torch.Tensor):
        return self.model(bag_tensor)
