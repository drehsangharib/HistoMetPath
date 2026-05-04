"""
HistoMetPath Utilities Module.

This module provides utility functions and helper classes for the project,
including logging, visualization, and general helper functions.

Design Decisions:
- Centralized utilities for consistency across the project
- Logging configured for research reproducibility
- Visualization tools for model interpretability

Note: This is a placeholder. Implementation will be added in Phase 2.
"""

from .logger import setup_logger
from .visualization import plot_attention, plot_confusion_matrix
from .seed import set_seed

__all__ = ["setup_logger", "plot_attention", "plot_confusion_matrix", "set_seed"]