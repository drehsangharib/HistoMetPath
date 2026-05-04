"""
Pathology Backbone Module.

Provides a clean, minimal backbone for histopathology image classification.
This implementation uses ResNet50 as the default backbone due to its proven
effectiveness in medical imaging tasks.

Design Decisions:
- ResNet50: Best-validated CNN for medical imaging; good balance of depth/speed
- Pretrained on ImageNet: Transfer learning accelerates training on limited medical data
- Modular design: Easy to swap backbones (EfficientNet, ViT) for comparison
- Feature extraction mode: Can output embeddings for downstream tasks

Scientific Rationale:
- ResNet architecture has been extensively validated on histopathology
- Skip connections help with fine-grained texture classification
- ImageNet pretraining provides useful low-level features (edges, textures)
- 2048-dimensional features are sufficient for metastasis detection

Note: This is a minimal Phase 1 implementation. More sophisticated architectures
and pretrained pathology-specific models can be added in later phases.
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class PathologyBackbone(nn.Module):
    """
    Backbone network for histopathology image feature extraction.

    This class wraps pretrained CNN architectures to provide a unified interface
    for feature extraction from H&E stained histology images.

    Attributes:
        name: Backbone architecture name
        pretrained: Whether to use ImageNet pretrained weights
        freeze: Whether to freeze backbone weights during training
        feature_dim: Dimension of output features

    Example:
        >>> backbone = PathologyBackbone(name="resnet50", pretrained=True)
        >>> features = backbone(images)  # [B, 2048]
    """

    # Supported backbone architectures
    SUPPORTED_BACKBONES = {
        "resnet18": {"feature_dim": 512, "depth": 18},
        "resnet34": {"feature_dim": 512, "depth": 34},
        "resnet50": {"feature_dim": 2048, "depth": 50},
        "resnet101": {"feature_dim": 2048, "depth": 101},
    }

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        freeze: bool = False,
    ):
        """
        Initialize the backbone network.

        Args:
            name: Backbone architecture name (resnet18/34/50/101)
            pretrained: Use ImageNet pretrained weights
            freeze: Freeze backbone weights (useful for transfer learning)

        Raises:
            ValueError: If backbone name is not supported

        Example:
            >>> # Use pretrained ResNet50
            >>> backbone = PathologyBackbone(pretrained=True)
            >>>
            >>> # Use ResNet18 (faster, less memory)
            >>> backbone = PathologyBackbone(name="resnet18", pretrained=True)
            >>>
            >>> # Freeze backbone for feature extraction
            >>> backbone = PathologyBackbone(freeze=True)
        """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.freeze = freeze

        # Validate backbone name
        if name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {name}. "
                f"Supported: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        self.feature_dim = self.SUPPORTED_BACKBONES[name]["feature_dim"]

        # Load pretrained model
        self._load_backbone()

        # Optionally freeze weights
        if self.freeze:
            self.freeze_weights()

    def _load_backbone(self):
        """Load the backbone architecture with optional pretrained weights."""
        # Map backbone names to torchvision model constructors
        model_constructors = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
        }

        # Load model with optional pretrained weights
        constructor = model_constructors[self.name]
        weights = "IMAGENET1K_V1" if self.pretrained else None
        self.model = constructor(weights=weights)

        # Remove the final classification layer
        # Original ResNet: [conv1, bn1, relu, maxpool, layer1-4, avgpool, fc]
        # Feature extraction: [conv1, bn1, relu, maxpool, layer1-4, avgpool]
        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Input images (B, 3, H, W) - typically (B, 3, 256, 256)

        Returns:
            Feature embeddings (B, feature_dim)
            - ResNet50: (B, 2048)
            - ResNet18/34: (B, 512)
        """
        return self.model(x)

    def freeze_weights(self):
        """Freeze all backbone weights (except final avgpool)."""
        for name, param in self.model.named_parameters():
            # Keep BatchNorm layers trainable for domain adaptation
            if "bn" not in name:
                param.requires_grad = False

    def unfreeze_weights(self):
        """Unfreeze all backbone weights."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self):
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())


class ClassificationHead(nn.Module):
    """
    Simple classification head for binary metastasis detection.

    Design Decisions:
    - Single output logit (not two) for BCEWithLogitsLoss compatibility
    - Configurable hidden dimension
    - No sigmoid/softmax applied (BCEWithLogitsLoss expects raw logits)

    Scientific Rationale:
    - Dropout prevents overfitting on limited medical datasets
    - Single logit is standard for binary classification with BCE loss
    - Raw logits allow BCEWithLogitsLoss to apply sigmoid numerically stable
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: Optional[int] = 512,
        num_classes: int = 1,  # Changed: 1 for binary classification
        dropout: float = 0.3,
    ):
        """
        Initialize the classification head.

        Args:
            input_dim: Dimension of input features (from backbone)
            hidden_dim: Hidden layer dimension (None = direct projection)
            num_classes: Number of output classes
            dropout: Dropout rate for regularization

        Example:
            >>> head = ClassificationHead(input_dim=2048, hidden_dim=512)
            >>> logits = head(features)  # [B, 2]
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # Should be 1 for binary

        # Build layers
        layers = []

        if hidden_dim is not None:
            # Hidden layer with dropout
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            output_dim = hidden_dim
        else:
            output_dim = input_dim

        # Output layer: single logit for binary classification
        # BCEWithLogitsLoss expects raw logits (no softmax)
        layers.append(nn.Linear(output_dim, 1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Logits (B, 1) - raw logit for binary classification
            Use torch.sigmoid() to convert to probability
        """
        return self.classifier(x).squeeze(-1)  # (B,)


class HistoMetPathModel(nn.Module):
    """
    Complete model for breast cancer metastasis classification.

    Combines a backbone network with a classification head for end-to-end
    binary classification of primary vs. metastatic breast cancer.

    Design Decisions:
    - Single output logit for BCEWithLogitsLoss compatibility
    - Modular: Backbone and head can be swapped independently
    - No sigmoid/softmax in forward (raw logits for BCEWithLogitsLoss)
    - Feature extraction mode available for visualization/debugging

    Scientific Rationale:
    - Binary classification is the foundational Phase 1 task
    - BCEWithLogitsLoss expects raw logits (numerically stable)
    - Single logit: probability of metastatic class (class 1)

    Attributes:
        backbone: Feature extraction backbone
        classifier: Classification head

    Example:
        >>> model = HistoMetPathModel(backbone_name="resnet50")
        >>> logits = model(images)  # [B,] - raw logit for metastatic
        >>> probs = torch.sigmoid(logits)  # probability of metastatic
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 512,
        num_classes: int = 1,  # Binary: single logit
        dropout: float = 0.3,
    ):
        """
        Initialize the complete model.

        Args:
            backbone_name: Backbone architecture name
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone during training
            hidden_dim: Hidden dimension for classification head
            num_classes: Number of output classes
            dropout: Dropout rate for regularization

        Example:
            >>> # Default: ResNet50 with pretrained weights
            >>> model = HistoMetPathModel()
            >>>
            >>> # Smaller model for limited resources
            >>> model = HistoMetPathModel(backbone_name="resnet18")
            >>>
            >>> # Freeze backbone for transfer learning
            >>> model = HistoMetPathModel(freeze_backbone=True)
        """
        super().__init__()

        # Initialize backbone
        self.backbone = PathologyBackbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )

        # Initialize classification head (single output for binary)
        self.classifier = ClassificationHead(
            input_dim=self.backbone.feature_dim,
            hidden_dim=hidden_dim,
            num_classes=1,  # Binary: single logit
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input images (B, 3, H, W) - typically (B, 3, 256, 256)

        Returns:
            Logits (B,) - raw logit for metastatic class (class 1)
            Use torch.sigmoid() to convert to probability [0, 1]
            - Positive logit -> high probability of metastatic
            - Negative logit -> high probability of primary
        """
        features = self.backbone(x)
        logits = self.classifier(features)  # (B,)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.

        Useful for:
        - Visualization of learned features
        - Debugging and analysis
        - Transfer learning (feature extraction mode)

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Feature embeddings (B, feature_dim)
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions using sigmoid.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Probabilities (B,) - probability of metastatic class
            Value in [0, 1]: close to 1 = metastatic, close to 0 = primary
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)  # (B,)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (0=primary, 1=metastatic).

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Predicted class indices (B,): 0 for primary, 1 for metastatic
        """
        probs = self.predict_proba(x)
        return (probs > 0.5).long()  # Binary threshold at 0.5


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HistoMetPathModel Usage Example")
    print("=" * 60)

    # Create model
    model = HistoMetPathModel(
        backbone_name="resnet50",
        pretrained=True,
        freeze_backbone=False,
        hidden_dim=512,
        num_classes=1,  # Binary: single logit
        dropout=0.3,
    )

    # Print model summary
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Backbone: {model.backbone.name}")
    print(f"Feature dim: {model.backbone.feature_dim}")
    print(f"Total parameters: {model.get_total_parameters():,}")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)           # [4] - single logit per sample
        features = model.extract_features(dummy_input)
        probs = model.predict_proba(dummy_input)  # [4] - sigmoid applied
        preds = model.predict(dummy_input)    # [4] - binary 0/1

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")      # [4]
    print(f"Features shape: {features.shape}")  # [4, 2048]
    print(f"Probs shape: {probs.shape}")        # [4]
    print(f"Predictions: {preds}")              # class indices

    print(f"\nSample predictions:")
    for i in range(batch_size):
        metastatic_prob = probs[i].item()
        pred_class = "metastatic" if preds[i] == 1 else "primary"
        print(f"  Sample {i}: prob={metastatic_prob:.3f} -> {pred_class}")

    print("\n" + "=" * 60)
    print("Design Notes:")
    print("=" * 60)
    print("""
    1. Backbone Choice (ResNet50):
       - Most validated CNN for medical imaging
       - Good balance of depth (50 layers) and speed
       - 2048-dim features sufficient for classification

    2. Transfer Learning:
       - ImageNet pretraining provides useful low-level features
       - Freeze backbone for limited data scenarios
       - Fine-tune entire model for best performance

    3. Extension Points:
       - Swap backbone: resnet18, resnet101, efficientnet, vit
       - Add attention mechanism for interpretability
       - Multi-scale input for different patch sizes

    4. Training (Phase 2):
       - Use BCEWithLogitsLoss for binary classification
       - Or use CrossEntropyLoss with one-hot labels
       - Apply class weights for imbalanced data
    """)

    print("=" * 60)