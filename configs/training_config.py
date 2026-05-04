"""
Training Configuration for HistoMetPath.

Defines all training hyperparameters including optimizer settings, learning
rate schedules, regularization, and checkpointing strategies.

Scientific Rationale:
- Learning rate: 1e-4 is standard for fine-tuning pretrained models
- Cosine annealing: Smooth LR decay prevents catastrophic forgetting
- Early stopping: Prevents overfitting on limited medical datasets
- Gradient clipping: Stabilizes training with deep networks
- Mixed precision: Enables larger batch sizes on modern GPUs

Note: This is a placeholder. Training loop implementation will be added in Phase 2.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration for training loop and optimization.

    Attributes:
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        optimizer: Optimizer name ("adam", "adamw", "sgd")
        scheduler: Learning rate scheduler ("cosine", "step", "plateau")
        warmup_epochs: Number of warmup epochs for learning rate
        early_stopping_patience: Epochs to wait before early stopping
        gradient_clip_norm: Maximum gradient norm for clipping
        mixed_precision: Enable automatic mixed precision training
        save_best_only: Only save checkpoints when metric improves
        checkpoint_dir: Directory to save model checkpoints
        log_interval: Log training metrics every N batches
        seed: Random seed for reproducibility
    """

    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    save_best_only: bool = True
    checkpoint_dir: str = "experiments/checkpoints"
    log_interval: int = 10
    seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer not in ["adam", "adamw", "sgd"]:
            raise ValueError("optimizer must be 'adam', 'adamw', or 'sgd'")
        if self.scheduler not in ["cosine", "step", "plateau"]:
            raise ValueError("scheduler must be 'cosine', 'step', or 'plateau'")