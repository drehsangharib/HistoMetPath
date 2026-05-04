"""
HistoMetPath Training Module.

Exports:
- HistoMetPathLightning: LightningModule (model + training logic)
- create_trainer: Trainer factory (logging, checkpointing, resume)
"""

from .lightning_trainer import HistoMetPathLightning
from .trainer import create_trainer

__all__ = [
    "HistoMetPathLightning",
    "create_trainer",
]