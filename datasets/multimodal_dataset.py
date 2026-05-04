"""
HistoPathDataset: Minimal Dataset for Breast Cancer Metastasis Classification.

This module provides a PyTorch Dataset implementation for loading histopathology
image patches for binary classification (primary vs. metastatic breast cancer).

Design Decisions:
- Lazy loading: Images are loaded on-demand to handle large medical image datasets
- Path-based structure: Expected directory format enables easy dataset organization
- Binary labels: Simple 0/1 encoding for primary/metastatic classification
- No augmentation in dataset: Augmentation should be applied via transforms
- No stain normalization: Will be added as optional transform in Phase 2

Scientific Rationale:
- Patch-based approach: Whole slide images (WSIs) are too large for direct processing
- Binary classification: Foundation for more complex metastasis detection tasks
- Lazy loading: Medical datasets can contain thousands of high-resolution images
- Clear label semantics: 0=primary tumor, 1=metastatic tumor (lymph node)

Dataset Directory Structure Expected:
    data_root/
    ├── train/
    │   ├── primary/          # Class 0: Primary breast tumor
    │   │   ├── patient_001_patch_001.png
    │   │   ├── patient_001_patch_002.png
    │   │   └── ...
    │   └── metastatic/       # Class 1: Metastatic tumor in lymph node
    │       ├── patient_001_patch_001.png
    │       └── ...
    ├── val/
    │   └── (same structure)
    └── test/
        └── (same structure)

Note: This implementation assumes:
- Image patches are RGB (3 channels)
- Images are stored as PNG, JPG, or TIFF files
- Patch size is consistent across dataset (typically 256x256 or 512x512)
- Labels are inferred from directory structure (primary/, metastatic/)

Future Extensions (Phase 2+):
- Stain normalization as optional transform
- Multiple magnification levels
- WSI-level loading with coordinate-based sampling
- Multi-class (primary, metastatic, normal)
- Genomic/clinical data integration (multimodal)
"""

from typing import Optional, Callable, Tuple, List
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class HistoPathDataset(Dataset):
    """
    PyTorch Dataset for breast cancer histology patch classification.

    This dataset loads image patches and their corresponding binary labels
    for training, validation, or testing of metastasis detection models.

    Attributes:
        image_paths: List of paths to image patches
        labels: List of binary labels (0=primary, 1=metastatic)
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to labels

    Example:
        >>> dataset = HistoPathDataset(
        ...     root_dir=Path("datasets/train"),
        ...     transform=transforms.Compose([...])
        ... )
        >>> image, label = dataset[0]
        >>> print(f"Image shape: {image.shape}, Label: {label}")
        Image shape: torch.Size([3, 256, 256]), Label: 0
    """

    # Supported image extensions
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # Class mapping: directory name -> integer label
    CLASS_MAPPING = {
        "primary": 0,      # Primary breast tumor
        "metastatic": 1,  # Metastatic tumor (typically in lymph node)
    }

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[set] = None,
        recursive: bool = True,
    ):
        """
        Initialize the HistoPathDataset.

        Args:
            root_dir: Root directory containing class subdirectories (primary/, metastatic/)
            transform: Optional transform to apply to images (e.g., normalization, augmentation)
            target_transform: Optional transform to apply to labels
            extensions: Set of valid file extensions (default: SUPPORTED_EXTENSIONS)
            recursive: Whether to search subdirectories recursively

        Raises:
            FileNotFoundError: If root_dir doesn't exist or has no valid class directories
            ValueError: If no valid images are found in the dataset

        Example:
            >>> from torchvision import transforms
            >>> transform = transforms.Compose([
            ...     transforms.ToTensor(),
            ...     transforms.Normalize(mean=[0.485, 0.456, 0.406],
            ...                          std=[0.229, 0.224, 0.225])
            ... ])
            >>> dataset = HistoPathDataset(
            ...     root_dir=Path("data/train"),
            ...     transform=transform
            ... )
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = extensions or self.SUPPORTED_EXTENSIONS

        # Validate root directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        # Scan for images and labels
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        self._scan_directory(recursive=recursive)

        # Validate dataset is not empty
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No valid images found in {self.root_dir}. "
                f"Expected directories: {list(self.CLASS_MAPPING.keys())}"
            )

        # Log dataset statistics
        n_primary = sum(1 for l in self.labels if l == 0)
        n_metastatic = sum(1 for l in self.labels if l == 1)
        print(f"Loaded {len(self.image_paths)} images: "
              f"{n_primary} primary, {n_metastatic} metastatic")

    def _scan_directory(self, recursive: bool = True):
        """
        Scan directory for images and corresponding labels.

        Args:
            recursive: Whether to search subdirectories recursively
        """
        for class_name, label in self.CLASS_MAPPING.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                # Skip missing class directories (e.g., if one class is empty)
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            # Find all images in the class directory
            if recursive:
                # Recursive search (useful for nested patient directories)
                pattern = "**/*"
            else:
                # Non-recursive (flat directory)
                pattern = "*"

            for ext in self.extensions:
                for image_path in class_dir.glob(f"{pattern}{ext}"):
                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of image patches in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample (0 to len(dataset)-1)

        Returns:
            Tuple of (image, label) where:
                - image: Tensor of shape (C, H, W) with values in [0, 1]
                - label: Integer (0=primary, 1=metastatic)

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Load image
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image using PIL (supports most common formats)
        # Note: Using PIL for broad format support; could use cv2 or imageio for specific needs
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image {image_path}: {e}")

        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)

        # Apply target transform if provided
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary with class counts: {"primary": n, "metastatic": n}
        """
        n_primary = sum(1 for l in self.labels if l == 0)
        n_metastatic = sum(1 for l in self.labels if l == 1)
        return {"primary": n_primary, "metastatic": n_metastatic}

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets.

        This is useful for weighted loss functions in training.

        Returns:
            Tensor of class weights [weight_primary, weight_metastatic]
            where weights are inversely proportional to class frequencies
        """
        n_samples = len(self)
        n_primary = sum(1 for l in self.labels if l == 0)
        n_metastatic = sum(1 for l in self.labels if l == 1)

        # Avoid division by zero
        weight_primary = n_samples / (2 * max(n_primary, 1))
        weight_metastatic = n_samples / (2 * max(n_metastatic, 1))

        return torch.tensor([weight_primary, weight_metastatic])

    def get_sample_info(self, idx: int) -> dict:
        """
        Get detailed information about a specific sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with sample metadata
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = "primary" if label == 0 else "metastatic"

        # Try to get image dimensions without loading
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = None, None

        return {
            "index": idx,
            "path": str(image_path),
            "class": class_name,
            "label": label,
            "width": width,
            "height": height,
        }


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    # This example demonstrates how to use the HistoPathDataset

    from torchvision import transforms

    # Example: Create a dataset with ImageNet normalization
    # (Note: For histopathology, you may want to compute dataset-specific statistics)

    print("=" * 60)
    print("HistoPathDataset Usage Example")
    print("=" * 60)

    # Define transforms (standard ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure consistent size
        transforms.ToTensor(),           # Convert PIL to tensor [0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])

    # Note: In practice, you would point to your actual data directory
    # dataset = HistoPathDataset(
    #     root_dir=Path("datasets/train"),
    #     transform=transform
    # )

    # For demonstration, we'll show what would happen
    print("""
    # To use this dataset:

    from pathlib import Path
    from torch.utils.data import DataLoader

    # Create dataset
    train_dataset = HistoPathDataset(
        root_dir=Path("data/train"),
        transform=transform,
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Iterate through batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")   # [B, 3, 256, 256]
        print(f"  Labels shape: {labels.shape}")   # [B]
        print(f"  Labels: {labels}")               # 0=primary, 1=metastatic
        break

    # Get class weights for loss function
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    """)

    # Demonstrate class mapping
    print("\nClass Mapping:")
    for class_name, label in HistoPathDataset.CLASS_MAPPING.items():
        print(f"  {class_name} -> {label}")

    print("\n" + "=" * 60)
    print("Note: Replace 'data/train' with your actual dataset path")
    print("=" * 60)