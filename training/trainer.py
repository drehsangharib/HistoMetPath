"""
Trainer factory for HistoMetPath using PyTorch Lightning.

- CPU-safe (auto-detected)
- Resume-safe
- Colab-robust
- No accelerator/device ambiguity
"""

import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def create_trainer(
    max_epochs: int = 5,
    log_dir: str = "logs/phase_2_3",
    log_every_n_steps: int = 50,
) -> pl.Trainer:
    """
    Create a PyTorch Lightning Trainer with logging and checkpointing.
    CPU is auto-selected to avoid Lightning 2.x device parsing bugs.
    """

    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = CSVLogger(
        save_dir=log_dir,
        name="metrics",
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch={epoch}",
        save_last=True,
        save_top_k=-1,      # save all epochs
        every_n_epochs=1,
    )

    # ✅ DO NOT pass accelerator or devices
    # ✅ Lightning will safely select CPU
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    return trainer