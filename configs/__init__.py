"""
HistoMetPath Configuration Module.

This module provides centralized configuration management for the entire project.
It uses dataclasses for type-safe, validated configuration objects that can be
easily serialized to/from YAML/JSON for experiment reproducibility.

Design Decisions:
- Dataclasses provide IDE autocomplete, type checking, and default values
- Separation of concerns: data, model, and training configs are independent
- All paths are relative to project root for portability
- Hyperparameters are grouped logically to facilitate grid search

Future Extensions:
- Add config validation (e.g., image size must be power of 2)
- Support config inheritance for ablation studies
- Add automatic config versioning
"""

from .data_config import DataConfig
from .model_config import ModelConfig
from .training_config import TrainingConfig

__all__ = ["DataConfig", "ModelConfig", "TrainingConfig"]