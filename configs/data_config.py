"""
Data Configuration for HistoMetPath.

Defines all data-related hyperparameters including dataset paths, preprocessing
parameters, and augmentation strategies specific to histopathology images.

Scientific Rationale:
- Histopathology images are typically 256x256 or 512x512 pixel patches
- H&E staining requires specific color normalization (Macenko or Vahadane)
- Data augmentation must preserve medical validity (no horizontal flips for
  oriented tissues, color jitter within realistic H&E ranges)

Note: This is a placeholder. Actual implementation will depend on the chosen
dataset (e.g., CAMELYON17, BreakHis, or custom dataset).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """
    Configuration for dataset loading and preprocessing.

    Attributes:
        dataset_name: Name of the dataset (e.g., "CAMELYON17", "BreakHis")
        data_root: Root directory containing the dataset
        patch_size: Size of image patches in pixels (assumes square)
        color_normalization: Stain normalization method ("macenko", "vahadane", "none")
        train_split: Fraction of data for training (0.0-1.0)
        val_split: Fraction of data for validation (0.0-1.0)
        test_split: Fraction of data for testing (0.0-1.0)
        batch_size: Number of samples per batch
        num_workers: DataLoader worker processes
        pin_memory: Enable pinned memory for faster GPU transfer
        augmentation: List of augmentation strategies to apply
        seed: Random seed for reproducibility
    """

    dataset_name: str = "CAMELYON17"
    data_root: Path = Path("datasets/camelyon17")
    patch_size: int = 256
    color_normalization: str = "macenko"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    augmentation: list = field(default_factory=lambda: ["random_flip", "color_jitter"])
    seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.train_split + self.val_split + self.test_split != 1.0:
            raise ValueError("Train/val/test splits must sum to 1.0")
        if self.patch_size not in [128, 256, 512, 1024]:
            raise ValueError("Patch size should be a power of 2 (128, 256, 512, 1024)")
        if self.color_normalization not in ["macenko", "vahadane", "none"]:
            raise ValueError("Color normalization must be 'macenko', 'vahadane', or 'none'")