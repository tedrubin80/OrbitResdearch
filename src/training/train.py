"""PyTorch training loop for orbit prediction models.

Supports LSTM and Transformer models with AdamW optimizer,
cosine LR schedule, early stopping, and checkpoint saving.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class Trainer:
    """Training loop with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config_path: str = "config.yaml",
        device: str = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.config = load_config(config_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tc = self.config["training"]
        self.optimizer = AdamW(
            model.parameters(),
            lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=tc["epochs"])
        self.criterion = nn.MSELoss()
        self.epochs = tc["epochs"]
        self.patience = tc["patience"]

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "model",
    ) -> dict:
        """Run full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name for checkpoint files

        Returns:
            Training history dict
        """
        print(f"\nTraining {model_name} on {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Epochs: {self.epochs}, Patience: {self.patience}")
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print("-" * 60)

        for epoch in range(self.epochs):
            t0 = time.time()

            # Training
            train_loss = self._train_epoch(train_loader)

            # Validation
            val_loss = self._validate(val_loader)

            # LR schedule
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            elapsed = time.time() - t0

            print(
                f"  Epoch {epoch+1:3d}/{self.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {lr:.2e} | {elapsed:.1f}s",
                end="",
            )

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(model_name)
                print(" *")
            else:
                self.patience_counter += 1
                print()
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Load best model
        self._load_checkpoint(model_name)
        print(f"\nBest val loss: {self.best_val_loss:.6f}")

        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for batch in loader:
            inputs, targets = batch[0].to(self.device), batch[-1].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0

        for batch in loader:
            inputs, targets = batch[0].to(self.device), batch[-1].to(self.device)
            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)
            total_loss += loss.item()

        return total_loss / len(loader)

    def _save_checkpoint(self, name: str):
        path = self.checkpoint_dir / f"{name}_best.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, path)

    def _load_checkpoint(self, name: str):
        path = self.checkpoint_dir / f"{name}_best.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])


class MultiModalTrainer(Trainer):
    """Trainer for multi-modal (orbit + solar wind) models."""

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for orbit_in, solar_in, targets in loader:
            orbit_in = orbit_in.to(self.device)
            solar_in = solar_in.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(orbit_in, solar_in)
            loss = self.criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0

        for orbit_in, solar_in, targets in loader:
            orbit_in = orbit_in.to(self.device)
            solar_in = solar_in.to(self.device)
            targets = targets.to(self.device)
            predictions = self.model(orbit_in, solar_in)
            loss = self.criterion(predictions, targets)
            total_loss += loss.item()

        return total_loss / len(loader)
