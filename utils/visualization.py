"""
Visualization Utilities for HistoMetPath.

Provides visualization tools for model interpretability and results analysis,
including attention maps, confusion matrices, and training curves.

Scientific Rationale:
- Attention visualization helps understand model focus regions
- Confusion matrices reveal error patterns
- Training curves show learning dynamics

Note: This is a placeholder. Implementation will be added in Phase 2.
"""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt


def plot_attention(
    image: np.ndarray,
    attention: np.ndarray,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay attention map on original image.

    Args:
        image: Original image (H, W, 3)
        attention: Attention map (H, W)
        alpha: Overlay transparency
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # TODO: Implement attention visualization
    raise NotImplementedError("Visualization implementation pending Phase 2")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["Primary", "Metastatic"],
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix with annotations.

    Args:
        cm: Confusion matrix (2x2)
        class_names: Class labels
        normalize: Whether to normalize values
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # TODO: Implement confusion matrix visualization
    raise NotImplementedError("Visualization implementation pending Phase 2")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[float],
    val_metrics: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Training metrics per epoch
        val_metrics: Validation metrics per epoch
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # TODO: Implement training curves visualization
    raise NotImplementedError("Visualization implementation pending Phase 2")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # TODO: Implement ROC curve visualization
    raise NotImplementedError("Visualization implementation pending Phase 2")