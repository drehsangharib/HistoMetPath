"""
LightningModule for HistoMetPath.

Defines:
- Model forward pass
- Training / validation / test steps
- Medical-grade metrics (accuracy, AUC, sensitivity, specificity)

Phase 2.3-C:
- Loss rebalancing via positive-class weighting
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from training.metrics import compute_binary_metrics


class HistoMetPathLightning(pl.LightningModule):
    """
    PyTorch Lightning module for breast cancer metastasis classification.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # --------------------------------------------------
        # Phase 2.3-C: loss rebalancing for sensitivity
        # --------------------------------------------------
        if class_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        else:
            # Emphasize positive (tumor) class
            pos_weight = torch.tensor(1.5)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Epoch-level buffers
        self.train_probs = []
        self.train_targets = []
        self.val_probs = []
        self.val_targets = []
        self.test_probs = []
        self.test_targets = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.train_probs.append(
            torch.sigmoid(logits).detach().cpu().numpy().ravel()
        )
        self.train_targets.append(
            y.detach().cpu().numpy().ravel()
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        metrics = compute_binary_metrics(
            np.concatenate(self.train_targets),
            np.concatenate(self.train_probs),
        )
        for k, v in metrics.items():
            self.log(f"train_{k}", v)
        self.train_probs.clear()
        self.train_targets.clear()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_probs.append(
            torch.sigmoid(logits).detach().cpu().numpy().ravel()
        )
        self.val_targets.append(
            y.detach().cpu().numpy().ravel()
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        metrics = compute_binary_metrics(
            np.concatenate(self.val_targets),
            np.concatenate(self.val_probs),
        )
        for k, v in metrics.items():
            self.log(f"val_{k}", v, prog_bar=True)
        self.val_probs.clear()
        self.val_targets.clear()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)

        logits = self(x)

        self.test_probs.append(
            torch.sigmoid(logits).detach().cpu().numpy().ravel()
        )
        self.test_targets.append(
            y.detach().cpu().numpy().ravel()
        )

    def on_test_epoch_end(self):
        metrics = compute_binary_metrics(
            np.concatenate(self.test_targets),
            np.concatenate(self.test_probs),
        )

        print("\n📊 Test Metrics (Phase 2.3-C)")
        for k, v in metrics.items():
            print(f"{k:>15}: {v:.4f}")

        self.test_probs.clear()
        self.test_targets.clear()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )