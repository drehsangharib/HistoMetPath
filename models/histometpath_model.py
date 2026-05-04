"""
HistoMetPath Phase 1 Model.

End-to-end CNN model for binary breast cancer metastasis classification,
designed for PatchCamelyon (PCam) and Lightning-based training.

Phase 1 goals:
- Simple, reliable baseline
- Transfer learning with ImageNet-pretrained backbones
- Single-logit output for BCEWithLogitsLoss

Phase 2 will introduce:
- Modular backbone / classifier objects
- WSI-specific architectures
- Attention and interpretability mechanisms
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


class HistoMetPathModel(nn.Module):
    """
    End-to-end model for binary metastasis classification.

    Uses a CNN backbone (ResNet family) followed by a small classification head.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 512,
        num_classes: int = 1,
        dropout: float = 0.3,
    ):
        """
        Args:
            backbone: Backbone architecture name (resnet18/34/50/101)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone parameters
            hidden_dim: Hidden dimension in classifier head
            num_classes: Output classes (1 for binary logit)
            dropout: Dropout probability
        """
        super().__init__()

        # ---------------------------------------------------------------------
        # Backbone
        # ---------------------------------------------------------------------
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the ImageNet classification head
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---------------------------------------------------------------------
        # Classifier head
        # ---------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, 3, H, W)

        Returns:
            Logits of shape (B, 1)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def get_total_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
