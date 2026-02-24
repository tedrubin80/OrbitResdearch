"""PyTorch and TensorFlow dataset classes for orbit prediction."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OrbitDataset(Dataset):
    """PyTorch Dataset for orbit prediction sequences."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Args:
            inputs: (N, input_steps, features) — input sequences
            targets: (N, horizon_steps, 3) — target positions (x, y, z)
        """
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MultiModalDataset(Dataset):
    """PyTorch Dataset combining orbit positions with solar wind features."""

    def __init__(
        self,
        orbit_inputs: np.ndarray,
        solar_inputs: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Args:
            orbit_inputs: (N, input_steps, orbit_features)
            solar_inputs: (N, input_steps, solar_features)
            targets: (N, horizon_steps, 3)
        """
        self.orbit_inputs = torch.from_numpy(orbit_inputs)
        self.solar_inputs = torch.from_numpy(solar_inputs)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.orbit_inputs)

    def __getitem__(self, idx):
        return self.orbit_inputs[idx], self.solar_inputs[idx], self.targets[idx]


def create_dataloaders(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create PyTorch DataLoaders from train/val/test splits.

    Args:
        splits: Dict from OrbitPreprocessor.temporal_split()
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Dict of DataLoaders keyed by 'train', 'val', 'test'
    """
    loaders = {}
    for split_name, (inputs, targets) in splits.items():
        dataset = OrbitDataset(inputs, targets)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def create_multimodal_dataloaders(
    orbit_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    solar_inputs_splits: dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create DataLoaders for multi-modal (orbit + solar wind) training."""
    loaders = {}
    for split_name in orbit_splits:
        orbit_in, targets = orbit_splits[split_name]
        solar_in = solar_inputs_splits[split_name]
        dataset = MultiModalDataset(orbit_in, solar_in, targets)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def create_tf_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
):
    """Create a TensorFlow dataset from numpy arrays.

    Returns a tf.data.Dataset or None if TF is not available.
    """
    try:
        import tensorflow as tf

        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(inputs), 10000))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    except ImportError:
        print("TensorFlow not available, skipping TF dataset creation")
        return None
