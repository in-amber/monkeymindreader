"""
Submission model for NSF HDR Neural Forecasting Challenge.
Wraps the NeuralForecaster with the required predict/load interface.
"""

import os
import torch
import numpy as np

from architecture import NeuralForecaster


# ---------------------------------------------------------------------------
# Per-sample normalization (inlined from dataset.py)
# ---------------------------------------------------------------------------

class PerSampleNormalizer:
    @staticmethod
    def normalize(x, observed_steps=10, eps=1e-8):
        """Normalize using stats from the first `observed_steps` timesteps only."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        B, T, C, F = x.shape
        ref = x[:, :observed_steps]
        ref_flat = ref.reshape(B, -1, F)
        mean = ref_flat.mean(dim=1, keepdim=True)   # [B, 1, F]
        std = ref_flat.std(dim=1, keepdim=True) + eps  # [B, 1, F]
        x_flat = x.reshape(B, T * C, F)
        x_norm = (x_flat - mean) / std
        return x_norm.reshape(B, T, C, F), mean, std

    @staticmethod
    def denormalize(x_norm, mean, std):
        """Reverse normalization. Handles [B, T, C] (single-feature) input."""
        if x_norm.dim() == 3:
            m = mean[:, :, 0:1]  # [B, 1, 1]
            s = std[:, :, 0:1]   # [B, 1, 1]
            return x_norm * s + m
        B, T, C, F = x_norm.shape
        x_denorm = x_norm.reshape(B, -1, F) * std + mean
        return x_denorm.reshape(B, T, C, F)


# ---------------------------------------------------------------------------
# Submission Model class
# ---------------------------------------------------------------------------

class Model:
    """
    Submission wrapper for the NeuralForecaster model.

    Interface:
        load()         — loads pre-trained weights from model_{monkey}.pth
        predict(X)     — X: numpy (N, 20, C, F), returns numpy (N, 20, C)
    """

    # Architecture flags enabled since R8b — must match the saved checkpoints
    USE_DUAL_HEADS = True
    USE_TEMPORAL_CONV = True
    BATCH_SIZE = 64  # For batched inference to avoid OOM on large test sets

    def __init__(self, monkey_name='beignet'):
        self.monkey_name = monkey_name
        if monkey_name == 'beignet':
            self.n_channels = 89
        elif monkey_name == 'affi':
            self.n_channels = 239
        else:
            raise ValueError(f'No such monkey: {monkey_name}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalizer = PerSampleNormalizer()
        self._model = None

    def load(self):
        """Load pre-trained weights. Must be called before predict()."""
        base = os.path.dirname(__file__)
        if self.monkey_name == 'beignet':
            path = os.path.join(base, 'model_beignet.pth')
        elif self.monkey_name == 'affi':
            path = os.path.join(base, 'model_affi.pth')
        else:
            raise ValueError(f'No such monkey: {self.monkey_name}')

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint['config']

        self._model = NeuralForecaster(
            n_channels=self.n_channels,
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4),
            n_refinement_iters=config.get('n_refinement_iters', 1),
            dropout=config.get('dropout', 0.186),
            use_dual_heads=self.USE_DUAL_HEADS,
            use_temporal_conv=self.USE_TEMPORAL_CONV,
        ).to(self.device)

        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

    def predict(self, x):
        """
        Run inference on a batch of neural data.

        Args:
            x: numpy array of shape (N, 20, C, F)
               First 10 timesteps are observed; last 10 are masked repeats.

        Returns:
            numpy array of shape (N, 20, C)
            First 10 columns: observed signal (feature 0, original units)
            Last 10 columns:  predicted signal (original units)
        """
        assert self._model is not None, "Call load() before predict()"

        N, T, C, F = x.shape
        output = np.zeros((N, 20, C), dtype=np.float32)

        # Copy observed feature[0] directly (already in original units)
        output[:, :10, :] = x[:, :10, :, 0]

        # Run inference in batches
        for start in range(0, N, self.BATCH_SIZE):
            end = min(start + self.BATCH_SIZE, N)
            batch = x[start:end]  # (B, 20, C, F)

            # Normalize using observed steps only (no leakage from future)
            x_norm, mean, std = self.normalizer.normalize(
                torch.from_numpy(batch).float(), observed_steps=10
            )

            # Feed only the observed 10 steps to the model
            x_obs = x_norm[:, :10, :, :].to(self.device)
            mean = mean.to(self.device)
            std = std.to(self.device)

            with torch.no_grad():
                pred_norm = self._model(x_obs, use_refinement=False, return_aux=False)
                # pred_norm: (B, 10, C) in normalized space

            # Denormalize back to original units
            pred_orig = self.normalizer.denormalize(pred_norm, mean, std)
            output[start:end, 10:, :] = pred_orig.cpu().numpy()

        return output
