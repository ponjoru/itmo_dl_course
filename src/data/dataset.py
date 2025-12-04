import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


class MalwareDataset(Dataset):
    """
    PyTorch Dataset for malware classification.

    Args:
        data_path: Path to the CSV file
        target_column: Name of the target column
        transform: Whether to apply standardization
        scaler: Pre-fitted scaler (for validation/test sets)
    """

    def __init__(
        self,
        data_path: str,
        target_column: str = 'Class',
        transform: bool = True,
        scaler: Optional[StandardScaler] = None,
        indices: Optional[np.ndarray] = None
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.transform = transform

        # Load data
        df = pd.read_csv(data_path)

        # Filter by indices if provided (for train/val/test splits)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        # Separate features and target
        self.y = df[target_column].values.astype(np.int64)
        self.X = df.drop(columns=[target_column]).values.astype(np.float32)

        # Apply standardization
        if transform:
            if scaler is None:
                self.scaler = StandardScaler()
                self.X = self.scaler.fit_transform(self.X)
            else:
                self.scaler = scaler
                self.X = self.scaler.transform(self.X)
        else:
            self.scaler = None

        self.num_features = self.X.shape[1]
        self.num_classes = len(np.unique(self.y))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def get_feature_dim(self) -> int:
        return self.num_features

    def get_num_classes(self) -> int:
        return self.num_classes
