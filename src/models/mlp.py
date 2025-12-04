import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for malware classification.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes (use 1 for binary classification with BCEWithLogitsLoss)
        dropout: Dropout rate
        activation: Activation function name ('relu', 'tanh', 'gelu')
        batch_norm: Whether to use batch normalization
        binary_classification: If True, outputs single logit for binary classification
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
        activation: str = 'relu',
        batch_norm: bool = True,
        binary_classification: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.binary_classification = binary_classification

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self.activation)

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Classification head
        # For binary classification with BCEWithLogitsLoss, use single output
        output_dim = 1 if binary_classification else num_classes
        self.classifier = nn.Linear(prev_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output logits of shape (batch_size, num_classes) for multi-class
            or (batch_size,) for binary classification
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        # For binary classification, squeeze the last dimension
        if self.binary_classification:
            logits = logits.squeeze(-1)

        return logits

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
