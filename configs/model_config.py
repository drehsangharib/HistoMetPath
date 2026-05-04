"""
Model Configuration for HistoMetPath.

Defines architecture hyperparameters and pretrained model settings.
Supports multiple backbone architectures for transfer learning.

Scientific Rationale:
- ResNet50/101: Well-validated for medical imaging, good balance of speed/accuracy
- EfficientNet: State-of-the-art efficiency, good for limited compute resources
- Vision Transformer: Emerging research shows promise for histopathology
- All backbones pretrained on ImageNet require adaptation for H&E images

Note: This is a placeholder. Model implementation will be added in Phase 2.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for model architecture and pretrained weights.

    Attributes:
        backbone: Backbone architecture ("resnet50", "resnet101", "efficientnet_b0", "vit")
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Freeze backbone weights during initial training
        num_classes: Number of output classes (2 for binary classification)
        dropout: Dropout rate for regularization
        hidden_dim: Hidden dimension for classification head
        activation: Activation function ("relu", "gelu", "silu")
    """

    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 2
    dropout: float = 0.3
    hidden_dim: Optional[int] = None  # Defaults to backbone output dim
    activation: str = "relu"

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_backbones = ["resnet50", "resnet101", "efficientnet_b0", "vit"]
        if self.backbone not in valid_backbones:
            raise ValueError(f"Backbone must be one of {valid_backbones}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Dropout must be between 0.0 and 1.0")
        if self.num_classes < 1:
            raise ValueError("num_classes must be at least 1")