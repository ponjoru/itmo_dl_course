import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
import pandas as pd

from .dataset import MalwareDataset


class MalwareDataModule:
    """
    DataModule for handling train/validation/test splits and dataloaders.

    Args:
        data_path: Path to the CSV file
        target_column: Name of the target column
        batch_size: Batch size for dataloaders
        val_split: Fraction of data for validation
        test_split: Fraction of data for test
        num_workers: Number of workers for dataloaders
        random_seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_path: str,
        target_column: str = 'Class',
        batch_size: int = 64,
        val_split: float = 0.15,
        test_split: float = 0.15,
        num_workers: int = 4,
        random_seed: int = 42,
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.random_seed = random_seed

        self.train_dataset: Optional[MalwareDataset] = None
        self.val_dataset: Optional[MalwareDataset] = None
        self.test_dataset: Optional[MalwareDataset] = None

    def setup(self) -> None:
        """
        Create train/val/test splits with stratification and initialize datasets.
        Stratification ensures class distribution is preserved across splits.
        """
        # Get total number of samples
        df = pd.read_csv(self.data_path)
        n_samples = len(df)
        indices = np.arange(n_samples)

        # Get original class distribution
        y_all = df[self.target_column].values
        unique_classes, counts_all = np.unique(y_all, return_counts=True)

        print(f"\nOriginal dataset:")
        print(f"  Total samples: {n_samples}")
        print(f"  Class distribution:")
        for cls, count in zip(unique_classes, counts_all):
            print(f"    Class {cls}: {count:5d} ({count/n_samples*100:5.2f}%)")

        # Split into train+val and test with stratification
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=y_all  # Stratify by class labels
        )

        # Split train+val into train and val with stratification
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.val_split / (1 - self.test_split),
            random_state=self.random_seed,
            stratify=y_all[train_val_idx]  # Stratify remaining samples
        )

        # Create training dataset (fit scaler)
        self.train_dataset = MalwareDataset(
            data_path=self.data_path,
            target_column=self.target_column,
            transform=True,
            scaler=None,
            indices=train_idx
        )

        # Create validation dataset (use training scaler)
        self.val_dataset = MalwareDataset(
            data_path=self.data_path,
            target_column=self.target_column,
            transform=True,
            scaler=self.train_dataset.scaler,
            indices=val_idx
        )

        # Create test dataset (use training scaler)
        self.test_dataset = MalwareDataset(
            data_path=self.data_path,
            target_column=self.target_column,
            transform=True,
            scaler=self.train_dataset.scaler,
            indices=test_idx
        )

        # Verify stratification by printing class distributions
        print(f"\nStratified data splits created:")
        self._print_split_distribution("Train", self.train_dataset.y, n_samples)
        self._print_split_distribution("Val", self.val_dataset.y, n_samples)
        self._print_split_distribution("Test", self.test_dataset.y, n_samples)

    def _print_split_distribution(self, split_name: str, y_split: np.ndarray, total_samples: int) -> None:
        """
        Print class distribution for a data split.

        Args:
            split_name: Name of the split (Train/Val/Test)
            y_split: Labels for the split
            total_samples: Total number of samples in original dataset
        """
        unique_classes, counts = np.unique(y_split, return_counts=True)
        n_split = len(y_split)

        print(f"  {split_name:5s}: {n_split:5d} samples ({n_split/total_samples*100:5.2f}% of total)")
        for cls, count in zip(unique_classes, counts):
            print(f"         Class {cls}: {count:5d} ({count/n_split*100:5.2f}%)")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_feature_dim(self) -> int:
        """Return number of input features."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first")
        return self.train_dataset.get_feature_dim()

    def get_num_classes(self) -> int:
        """Return number of classes."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first")
        return self.train_dataset.get_num_classes()
