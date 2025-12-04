import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
import logging

from ..utils.metrics import MetricsTracker


class Trainer:
    """
    Trainer for malware classification model.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on
        max_epochs: Maximum number of epochs
        checkpoint_dir: Directory to save checkpoints
        tensorboard_dir: Directory for tensorboard logs
        early_stopping_patience: Patience for early stopping (None to disable)
        logger: Logger instance
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        max_epochs: int = 100,
        checkpoint_dir: Optional[Path] = None,
        tensorboard_dir: Optional[Path] = None,
        early_stopping_patience: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.logger = logger

        # Detect if using binary classification (BCEWithLogitsLoss)
        self.is_binary = isinstance(criterion, nn.BCEWithLogitsLoss)

        # Create checkpoint directory
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tensorboard writer
        self.writer = None
        if tensorboard_dir is not None:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()
        losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.max_epochs} [Train]')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x)

            # For BCEWithLogitsLoss, targets must be float
            if self.is_binary:
                y_loss = y.float()
            else:
                y_loss = y

            loss = self.criterion(logits, y_loss)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            losses.append(loss.item())

            # Compute predictions and probabilities based on loss type
            if self.is_binary:
                # Binary classification: apply sigmoid and threshold at 0.5
                probabilities_pos = torch.sigmoid(logits)
                predictions = (probabilities_pos > 0.5).long()
                # Create 2D probability tensor for metrics tracker
                probabilities = torch.stack([1 - probabilities_pos, probabilities_pos], dim=1)
            else:
                # Multi-class classification: use softmax
                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)

            metrics_tracker.update(predictions, y, probabilities)

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute metrics
        metrics = metrics_tracker.compute()
        metrics['loss'] = metrics_tracker.compute_loss(losses)

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        losses = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1}/{self.max_epochs} [Val]')
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                logits = self.model(x)

                # For BCEWithLogitsLoss, targets must be float
                if self.is_binary:
                    y_loss = y.float()
                else:
                    y_loss = y

                loss = self.criterion(logits, y_loss)

                # Track metrics
                losses.append(loss.item())

                # Compute predictions and probabilities based on loss type
                if self.is_binary:
                    # Binary classification: apply sigmoid and threshold at 0.5
                    probabilities_pos = torch.sigmoid(logits)
                    predictions = (probabilities_pos > 0.5).long()
                    # Create 2D probability tensor for metrics tracker
                    probabilities = torch.stack([1 - probabilities_pos, probabilities_pos], dim=1)
                else:
                    # Multi-class classification: use softmax
                    predictions = torch.argmax(logits, dim=1)
                    probabilities = torch.softmax(logits, dim=1)

                metrics_tracker.update(predictions, y, probabilities)

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute metrics
        metrics = metrics_tracker.compute()
        metrics['loss'] = metrics_tracker.compute_loss(losses)

        return metrics

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        losses = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                logits = self.model(x)

                # For BCEWithLogitsLoss, targets must be float
                if self.is_binary:
                    y_loss = y.float()
                else:
                    y_loss = y

                loss = self.criterion(logits, y_loss)

                # Track metrics
                losses.append(loss.item())

                # Compute predictions and probabilities based on loss type
                if self.is_binary:
                    # Binary classification: apply sigmoid and threshold at 0.5
                    probabilities_pos = torch.sigmoid(logits)
                    predictions = (probabilities_pos > 0.5).long()
                    # Create 2D probability tensor for metrics tracker
                    probabilities = torch.stack([1 - probabilities_pos, probabilities_pos], dim=1)
                else:
                    # Multi-class classification: use softmax
                    predictions = torch.argmax(logits, dim=1)
                    probabilities = torch.softmax(logits, dim=1)

                metrics_tracker.update(predictions, y, probabilities)

        # Compute metrics
        metrics = metrics_tracker.compute()
        metrics['loss'] = metrics_tracker.compute_loss(losses)

        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        if self.logger:
            self.logger.info(f"Starting training for {self.max_epochs} epochs")
            self.logger.info(f"Model parameters: {self.model.get_num_params():,}")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_history.append(val_metrics)

            # Log metrics
            self._log_metrics(train_metrics, val_metrics)

            # Log to tensorboard
            if self.writer:
                self._log_tensorboard(train_metrics, val_metrics)

            # Check for improvement and save checkpoint
            improved = self._check_improvement(val_metrics)
            if improved and self.checkpoint_dir:
                self.save_checkpoint('best_model.pt')

            # Early stopping
            if self.early_stopping_patience is not None:
                if not improved:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        if self.logger:
                            self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                else:
                    self.patience_counter = 0

        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint('final_model.pt')

        if self.logger:
            self.logger.info("Training completed!")

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log metrics to console and logger."""
        train_str = metrics_tracker.format_metrics(
            {k: v for k, v in train_metrics.items() if k != 'confusion_matrix'},
            prefix='train_'
        )
        val_str = metrics_tracker.format_metrics(
            {k: v for k, v in val_metrics.items() if k != 'confusion_matrix'},
            prefix='val_'
        )

        message = f"Epoch {self.current_epoch + 1}/{self.max_epochs} - {train_str} | {val_str}"
        print(message)
        if self.logger:
            self.logger.info(message)

    def _log_tensorboard(self, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log metrics to tensorboard."""
        for key, value in train_metrics.items():
            if key != 'confusion_matrix':
                self.writer.add_scalar(f'train/{key}', value, self.current_epoch)

        for key, value in val_metrics.items():
            if key != 'confusion_matrix':
                self.writer.add_scalar(f'val/{key}', value, self.current_epoch)

    def _check_improvement(self, val_metrics: Dict) -> bool:
        """Check if validation metrics improved."""
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['f1']

        improved = False
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
            improved = True

        return improved

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
        }, checkpoint_path)

        if self.logger:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint['best_val_f1']

        if self.logger:
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")


# Import MetricsTracker for formatting
from ..utils.metrics import MetricsTracker
metrics_tracker = MetricsTracker()
