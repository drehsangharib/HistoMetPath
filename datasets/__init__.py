"""
HistoMetPath Dataset Module.

This module provides dataset classes for loading and preprocessing histopathology
image patches. It supports multiple dataset formats and includes stain
normalization utilities specific to H&E images.

Design Decisions:
- PyTorch Dataset classes for seamless integration with DataLoader
- Lazy loading to handle large medical image datasets efficiently
- Stain normalization as a preprocessing step (not augmentation)
- Clear separation between dataset loading and augmentation

Scientific Rationale:
- H&E (Hematoxylin and Eosin) staining varies significantly between labs
- Stain normalization reduces domain shift between training and test data
- Patch-based approaches are standard for histopathology (WSIs too large for direct processing)
"""

from .histopathology_dataset import HistoPathDataset
from .pcam_dataset import PCamHDF5Dataset
from .stain_normalizer import StainNormalizer

__all__ = [
    "HistoPathDataset",
    "PCamHDF5Dataset",
    "StainNormalizer",
]