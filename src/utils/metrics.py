import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from typing import Dict, List
import torch


class MetricsTracker:
    """
    Tracks and computes classification metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all stored predictions and targets."""
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.probabilities: List[np.ndarray] = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor = None
    ) -> None:
        """
        Update tracker with new predictions and targets.

        Args:
            predictions: Predicted class labels
            targets: Ground truth labels
            probabilities: Predicted probabilities (optional, for ROC-AUC)
        """
        self.predictions.append(predictions.cpu().numpy())
        self.targets.append(targets.cpu().numpy())
        if probabilities is not None:
            self.probabilities.append(probabilities.cpu().numpy())

    def compute(self, average: str = 'binary') -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            average: Averaging strategy for multi-class ('binary', 'macro', 'micro')

        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions:
            return {}

        # Concatenate all batches
        y_pred = np.concatenate(self.predictions)
        y_true = np.concatenate(self.targets)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        # Compute ROC-AUC if probabilities are available
        if self.probabilities:
            y_prob = np.concatenate(self.probabilities)
            try:
                if average == 'binary':
                    # For binary classification, use probability of positive class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # For multi-class, use one-vs-rest
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average=average
                    )
            except ValueError:
                # ROC-AUC computation failed (e.g., only one class present)
                metrics['roc_auc'] = 0.0

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        return metrics

    def compute_loss(self, losses: List[float]) -> float:
        """Compute average loss."""
        return np.mean(losses) if losses else 0.0

    def format_metrics(self, metrics: Dict[str, float], prefix: str = '') -> str:
        """
        Format metrics as a readable string.

        Args:
            metrics: Dictionary of metrics
            prefix: Prefix to add to each metric name

        Returns:
            Formatted string
        """
        lines = []
        for key, value in metrics.items():
            if key == 'confusion_matrix':
                continue
            name = f"{prefix}{key}" if prefix else key
            lines.append(f"{name}: {value:.4f}")
        return " | ".join(lines)
