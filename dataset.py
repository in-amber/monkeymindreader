"""
Dataset module for Neural Time-Series Forecasting.
Implements per-sample normalization for robustness to session drift.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PerSampleNormalizer:
    """
    Per-sample, per-channel normalization for handling session drift.
    Critical for cross-session generalization where signal magnitude may change.

    Each channel is normalized independently so that high-amplitude channels
    don't dominate the statistics of low-amplitude ones.

    Statistics are computed from observed timesteps only, so normalization
    is identical during training and challenge inference (where future
    timesteps are unavailable).
    """
    @staticmethod
    def normalize(x, observed_steps=None, eps=1e-8, min_std=0.1):
        """
        Normalize each sample and channel independently using observed timesteps.

        Args:
            x: Tensor of shape [B, T, C, F] or numpy array
            observed_steps: Number of observed timesteps to compute stats from.
                If None, uses all timesteps (legacy behavior).
            eps: Small value for numerical stability
            min_std: Minimum std threshold to prevent blow-up on near-constant channels

        Returns:
            Normalized data [B, T, C, F], mean [B, 1, C, F], std [B, 1, C, F]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        B, T, C, F = x.shape

        # Compute statistics from observed portion only, per channel
        ref = x[:, :observed_steps] if observed_steps is not None else x
        # ref: [B, T_obs, C, F] → mean/std over time dimension only
        mean = ref.mean(dim=1, keepdim=True)  # [B, 1, C, F]
        std = ref.std(dim=1, keepdim=True)    # [B, 1, C, F]
        std = std.clamp(min=min_std) + eps    # Floor to prevent blow-up

        x_norm = (x - mean) / std
        return x_norm, mean, std

    @staticmethod
    def denormalize(x_norm, mean, std):
        """Reverse normalization.

        Args:
            x_norm: Normalized tensor [B, T, C, F] or [B, T, C] (single feature)
            mean: Per-sample per-channel mean from normalize() [B, 1, C, F]
            std: Per-sample per-channel std from normalize() [B, 1, C, F]

        Returns:
            Denormalized tensor in original units
        """
        if x_norm.dim() == 3:
            # [B, T, C] — single feature (e.g. target predictions)
            # Use first feature's stats: mean[:, :, :, 0], std[:, :, :, 0]
            m = mean[:, :, :, 0]  # [B, 1, C]
            s = std[:, :, :, 0]   # [B, 1, C]
            return x_norm * s + m
        return x_norm * std + mean


class NeuralDataAugmentation:
    """
    Data augmentation for neural forecasting.
    Includes electrode dropout and Gaussian noise.
    """
    def __init__(self, electrode_dropout_prob=0.1, noise_std=0.02):
        self.electrode_dropout_prob = electrode_dropout_prob
        self.noise_std = noise_std

    def electrode_dropout(self, x):
        """Randomly zero out electrodes to simulate missing channels."""
        mask = torch.rand(x.size(0), 1, x.size(2), 1) > self.electrode_dropout_prob
        return x * mask.float().to(x.device)

    def gaussian_noise(self, x):
        """Add Gaussian noise to the data."""
        return x + torch.randn_like(x) * self.noise_std

    def __call__(self, x, training=True):
        if not training:
            return x
        if random.random() < 0.5:
            x = self.electrode_dropout(x)
        if random.random() < 0.5:
            x = self.gaussian_noise(x)
        return x


class NeuralForecastDataset(Dataset):
    """
    Dataset for neural forecasting with per-sample normalization.

    The data format is: N × T × C × F
    - N: number of samples
    - T: 20 timesteps (10 observed + 10 to predict)
    - C: number of channels/electrodes
    - F: 9 features (feature[0] is target, [1:8] are frequency bands)
    """
    def __init__(
        self,
        data_path=None,
        neural_data=None,
        observed_steps=10,
        use_all_features=True,
        augmentation=None,
        training=True
    ):
        """
        Args:
            data_path: Path to .npz file containing neural data
            neural_data: Alternatively, provide data directly as numpy array
            observed_steps: Number of input timesteps (default 10)
            use_all_features: If True, use all 9 features; if False, only feature[0]
            augmentation: Optional NeuralDataAugmentation instance
            training: Whether this is a training dataset
        """
        if neural_data is not None:
            self.data = neural_data
        elif data_path is not None:
            loaded = np.load(data_path)
            # Handle both 'arr_0' key and direct array
            if 'arr_0' in loaded:
                self.data = loaded['arr_0']
            else:
                self.data = loaded[list(loaded.keys())[0]]
        else:
            raise ValueError("Must provide either data_path or neural_data")

        self.observed_steps = observed_steps
        self.use_all_features = use_all_features
        self.augmentation = augmentation
        self.training = training
        self.normalizer = PerSampleNormalizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]  # [T, C, F] = [20, C, 9]

        # Convert to tensor
        sample = torch.tensor(sample, dtype=torch.float32)

        # Normalize using observed timesteps only (no data leakage from future)
        sample = sample.unsqueeze(0)  # [1, T, C, F]
        sample, mean, std = self.normalizer.normalize(
            sample, observed_steps=self.observed_steps
        )
        sample = sample.squeeze(0)  # [T, C, F]
        mean = mean.squeeze(0)      # [1, C, F]
        std = std.squeeze(0)        # [1, C, F]

        # Split into input and target
        x = sample[:self.observed_steps]  # [10, C, F]
        y = sample[self.observed_steps:]  # [10, C, F] - all features for auxiliary loss

        # Apply augmentation to input only
        if self.augmentation is not None and self.training:
            x = x.unsqueeze(0)
            x = self.augmentation(x, training=True)
            x = x.squeeze(0)

        return x, y, mean, std


def load_monkey_data(monkey_name, data_dir='train_data_neuro', include_private=True):
    """
    Load all available training data for a monkey.

    Args:
        monkey_name: 'affi' or 'beignet'
        data_dir: Directory containing the data files
        include_private: Whether to include additional private session data

    Returns:
        Combined numpy array of all data
    """
    all_data = []

    # Load main training data
    main_file = os.path.join(data_dir, f'train_data_{monkey_name}.npz')
    if os.path.exists(main_file):
        loaded = np.load(main_file)
        if 'arr_0' in loaded:
            all_data.append(loaded['arr_0'])
        else:
            all_data.append(loaded[list(loaded.keys())[0]])

    # Load additional private session data if available
    if include_private:
        for filename in os.listdir(data_dir):
            if filename.startswith(f'train_data_{monkey_name}_') and filename.endswith('_private.npz'):
                filepath = os.path.join(data_dir, filename)
                loaded = np.load(filepath)
                if 'arr_0' in loaded:
                    all_data.append(loaded['arr_0'])
                else:
                    all_data.append(loaded[list(loaded.keys())[0]])

    if not all_data:
        raise FileNotFoundError(f"No data found for monkey '{monkey_name}' in {data_dir}")

    return np.concatenate(all_data, axis=0)


def create_dataloaders(
    monkey_name,
    data_dir='train_data_neuro',
    batch_size=32,
    train_split=0.8,
    val_split=0.1,
    use_augmentation=True,
    num_workers=0,
    seed=7
):
    """
    Create train, validation, and test dataloaders for a monkey.

    Args:
        monkey_name: 'affi' or 'beignet'
        data_dir: Directory containing the data files
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation (rest is test)
        use_augmentation: Whether to use data augmentation
        num_workers: Number of dataloader workers
        seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load all data
    data = load_monkey_data(monkey_name, data_dir)

    # Shuffle with fixed seed for reproducibility
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    data = data[indices]

    # Split data
    n_train = int(len(data) * train_split)
    n_val = int(len(data) * val_split)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Create augmentation
    augmentation = NeuralDataAugmentation() if use_augmentation else None

    # Create datasets
    train_dataset = NeuralForecastDataset(
        neural_data=train_data,
        augmentation=augmentation,
        training=True
    )
    val_dataset = NeuralForecastDataset(
        neural_data=val_data,
        augmentation=None,
        training=False
    )
    test_dataset = NeuralForecastDataset(
        neural_data=test_data,
        augmentation=None,
        training=False
    )

    # Only use pin_memory on CUDA (not supported on MPS)
    import torch
    pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
