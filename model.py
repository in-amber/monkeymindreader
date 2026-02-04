"""
Neural Time-Series Forecasting Model
Factorized Spatiotemporal Encoder with Iterative Refinement

Based on the implementation plan for the NSF HDR Neural Forecasting Challenge.
"""

import math
import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """
    Transforms 9 raw features per electrode into a d-dimensional representation.
    Uses a shared MLP across all electrodes and timesteps.
    """
    def __init__(self, input_dim=9, hidden_dim=32, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # x: [B, T, C, F] -> [B, T, C, d]
        return self.encoder(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal positions.
    No spatial positions - spatial attention is permutation-equivariant for robustness.
    """
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: [1, max_len, 1, d_model] for broadcasting with [B, T, C, d]
        self.register_buffer('pe', pe.unsqueeze(0).unsqueeze(2))

    def forward(self, x):
        # x: [B, T, C, d]
        return x + self.pe[:, :x.size(1), :, :]


class FactorizedSpatiotemporalBlock(nn.Module):
    """
    Core computational unit with factorized attention:
    - Temporal self-attention (within each electrode)
    - Spatial self-attention (across electrodes)
    - Feed-forward network with residual connections
    """
    def __init__(self, d_model, n_heads, dropout=0.1, ff_mult=4):
        super().__init__()
        # Temporal attention (within each electrode)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)

        # Spatial attention (across electrodes)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temporal_mask=None):
        B, T, C, d = x.shape

        # === Temporal Attention ===
        # Reshape to [B*C, T, d] - each electrode attends to its own history
        x_t = x.permute(0, 2, 1, 3).reshape(B * C, T, d)
        attn_out, _ = self.temporal_attn(x_t, x_t, x_t, attn_mask=temporal_mask)
        x_t = self.temporal_norm(x_t + self.dropout(attn_out))
        x = x_t.reshape(B, C, T, d).permute(0, 2, 1, 3)

        # === Spatial Attention ===
        # Reshape to [B*T, C, d] - electrodes attend to each other at each timestep
        x_s = x.reshape(B * T, C, d)
        attn_out, _ = self.spatial_attn(x_s, x_s, x_s)
        x_s = self.spatial_norm(x_s + self.dropout(attn_out))
        x = x_s.reshape(B, T, C, d)

        # === Feed-Forward ===
        x = self.ffn_norm(x + self.ffn(x))
        return x


class PredictionHead(nn.Module):
    """
    Predicts future timesteps from encoded representations.
    """
    def __init__(self, d_model, n_future=10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_future),
        )

    def forward(self, x):
        # x: [B, T, C, d] - take last timestep
        if x.dim() == 4:
            x = x[:, -1, :, :]  # [B, C, d]
        return self.head(x)  # [B, C, 10]


class NeuralForecaster(nn.Module):
    """
    Complete neural forecasting model with:
    - Feature encoding
    - Sinusoidal positional encoding
    - Factorized spatiotemporal blocks
    - Prediction head
    - Iterative refinement
    """
    def __init__(
        self,
        n_channels,
        n_features=9,
        d_model=64,
        n_heads=4,
        n_layers=3,
        n_future=10,
        n_refinement_iters=2,
        dropout=0.15
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.d_model = d_model
        self.n_future = n_future
        self.n_refinement_iters = n_refinement_iters

        self.feature_encoder = FeatureEncoder(n_features, 32, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            FactorizedSpatiotemporalBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.pred_head = PredictionHead(d_model, n_future)

    def encode(self, x):
        """Encode input through feature encoder, positional encoding, and transformer blocks."""
        z = self.feature_encoder(x)
        z = self.pos_encoding(z)
        for block in self.blocks:
            z = block(z)
        return z

    def forward(self, x, use_refinement=True):
        """
        Forward pass with optional iterative refinement.

        Args:
            x: Input tensor of shape [B, T, C, F] where T=10 (observed timesteps)
            use_refinement: Whether to use iterative refinement

        Returns:
            Predictions of shape [B, n_future, C] (only feature[0])
        """
        B, T, C, F = x.shape

        # Initial encoding and prediction
        encoded = self.encode(x)
        pred = self.pred_head(encoded).permute(0, 2, 1)  # [B, 10, C]

        if use_refinement and self.n_refinement_iters > 0:
            for _ in range(self.n_refinement_iters):
                # Create predicted features tensor (zeros for auxiliary features)
                pred_features = torch.zeros(B, self.n_future, C, F, device=x.device)
                pred_features[:, :, :, 0] = pred  # Only fill in predicted feature[0]

                # Concatenate observed and predicted sequences
                full_seq = torch.cat([x, pred_features], dim=1)  # [B, 20, C, F]

                # Re-encode the full sequence
                full_encoded = self.encode(full_seq)

                # Extract encoding for future timesteps and predict
                future_encoded = full_encoded[:, T:, :, :]  # [B, 10, C, d]
                pred = self.pred_head(future_encoded.mean(dim=1)).permute(0, 2, 1)

        return pred

    def predict(self, x):
        """Inference method - same as forward but always uses refinement."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, use_refinement=True)


def get_model_for_monkey(monkey_name, n_channels=None, **kwargs):
    """
    Factory function to create model with correct channel count for each monkey.

    Args:
        monkey_name: 'affi' (Monkey A) or 'beignet' (Monkey B)
        n_channels: Override channel count (auto-detected from data if None)
        **kwargs: Additional arguments to pass to NeuralForecaster

    Returns:
        NeuralForecaster model configured for the specified monkey
    """
    if n_channels is None:
        # Default channel counts based on data files
        # Note: actual data has 89 channels for beignet, 239 for affi
        if monkey_name == 'affi':
            n_channels = 239
        elif monkey_name == 'beignet':
            n_channels = 89  # Actual data has 89 channels
        else:
            raise ValueError(f"Unknown monkey: {monkey_name}. Use 'affi' or 'beignet'.")

    return NeuralForecaster(n_channels=n_channels, **kwargs)
