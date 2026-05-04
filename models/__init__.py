"""
HistoMetPath Models Module.

This module provides neural network architectures for breast cancer metastasis
classification from histopathology images.

Design Decisions:
- Modular architecture with configurable backbones
- Support for both CNN and Vision Transformer architectures
- Clear separation between feature extraction and classification head
- Built-in support for mixed precision and gradient checkpointing

Scientific Rationale:
- Transfer learning from ImageNet is standard for medical imaging
- Multiple architectures allow for ablation studies and comparison
- Modular design supports easy swapping of components

Note: This is a placeholder. Model implementations will be added in Phase 2.
"""

from .backbone import Backbone
from .classifier import Classifier
from .histometpath_model import HistoMetPathModel

__all__ = ["Backbone", "Classifier", "HistoMetPathModel"]