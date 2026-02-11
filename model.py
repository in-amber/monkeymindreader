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
    Feature-specific encoder with separate pathways for target signal and frequency bands.

    Feature[0] is the target signal we're predicting, while features[1-8] are frequency
    band powers. These have different semantics, so we encode them separately before fusing.
    """
    def __init__(self, input_dim=9, hidden_dim=32, output_dim=64):
        super().__init__()
        half_dim = output_dim // 2

        # Separate encoder for target signal (feature 0)
        self.target_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, half_dim),
            nn.LayerNorm(half_dim),
        )

        # Separate encoder for frequency bands (features 1-8)
        self.freq_encoder = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, half_dim),
            nn.LayerNorm(half_dim),
        )

        # Fusion layer to combine both pathways
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # x: [B, T, C, F] where F=9
        target = self.target_encoder(x[..., :1])       # [B, T, C, d/2]
        freq = self.freq_encoder(x[..., 1:])           # [B, T, C, d/2]
        fused = torch.cat([target, freq], dim=-1)      # [B, T, C, d]
        return self.fusion(fused)                      # [B, T, C, d]


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
    Attention-based prediction head that learns to weight relevant timesteps.

    Instead of just using the last timestep, this head uses a learnable query
    that attends to all encoded timesteps, allowing the model to aggregate
    information from the full temporal context.
    """
    def __init__(self, d_model, n_future=10, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_future = n_future

        # Learnable query token for attending to temporal sequence
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Cross-attention: query attends to encoded sequence
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Prediction MLP
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_future),
        )

    def forward(self, x):
        # x: [B, T, C, d]
        if x.dim() == 3:
            # Already [B, C, d] - just apply head directly
            return self.head(x)

        B, T, C, d = x.shape

        # Reshape to [B*C, T, d] for per-electrode attention
        x_flat = x.permute(0, 2, 1, 3).reshape(B * C, T, d)

        # Expand query for all samples: [B*C, 1, d]
        query = self.query.expand(B * C, 1, d)

        # Cross-attention: query attends to all timesteps
        attended, _ = self.attn(query, x_flat, x_flat)  # [B*C, 1, d]
        attended = self.attn_norm(query + attended)      # Residual connection

        # Reshape back to [B, C, d]
        attended = attended.squeeze(1).reshape(B, C, d)

        # Predict future timesteps
        return self.head(attended)  # [B, C, n_future]


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

        # Learnable spatial (channel) embeddings — gives the model channel identity
        # so it can learn per-channel dynamics and cross-channel correlations.
        # NOTE: This assumes consistent electrode ordering across sessions. The
        # challenge involves cross-session evaluation where recording drift may
        # shift electrode characteristics. If this hurts generalization, consider
        # removing or adding dropout to these embeddings.
        self.channel_embedding = nn.Parameter(
            torch.randn(1, 1, n_channels, d_model) * 0.02
        )

        self.blocks = nn.ModuleList([
            FactorizedSpatiotemporalBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.pred_head = PredictionHead(d_model, n_future, dropout=dropout)

        # Per-channel output scaling — lets easy channels be handled trivially
        # while the model focuses capacity on harder patterns.
        # NOTE: Same session-drift caveat as channel_embedding above. If
        # per-channel scale/bias overfit to training sessions, consider removing.
        self.channel_scale = nn.Parameter(torch.ones(1, 1, n_channels))
        self.channel_bias = nn.Parameter(torch.zeros(1, 1, n_channels))

        # Auxiliary head for predicting frequency bands (features 1-8)
        # Uses simpler MLP since auxiliary task is secondary
        self.aux_pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_future * (n_features - 1)),  # Predict all freq bands
        )
        self.n_aux_features = n_features - 1  # 8 frequency bands

    def encode(self, x, causal=True):
        """Encode input through feature encoder, positional encoding, and transformer blocks.

        Args:
            x: Input tensor of shape [B, T, C, F]
            causal: If True, use causal masking in temporal attention (each timestep
                    can only attend to itself and previous timesteps)
        """
        z = self.feature_encoder(x)
        z = self.pos_encoding(z)
        z = z + self.channel_embedding[:, :, :z.size(2), :]

        # Create causal mask for temporal attention
        T = z.size(1)
        temporal_mask = None
        if causal:
            temporal_mask = torch.triu(
                torch.ones(T, T, device=z.device) * float('-inf'),
                diagonal=1
            )

        for block in self.blocks:
            z = block(z, temporal_mask=temporal_mask)
        return z

    def forward(self, x, use_refinement=True, return_aux=False):
        """
        Forward pass with optional iterative refinement.

        Args:
            x: Input tensor of shape [B, T, C, F] where T=10 (observed timesteps)
            use_refinement: Whether to use iterative refinement
            return_aux: Whether to return auxiliary predictions for frequency bands

        Returns:
            If return_aux=False: pred of shape [B, n_future, C] (feature[0] only)
            If return_aux=True: (pred, pred_aux) where pred_aux is [B, n_future, C, 8]
        """
        B, T, C, F = x.shape

        # Initial encoding and prediction
        encoded = self.encode(x)
        pred = self.pred_head(encoded).permute(0, 2, 1)  # [B, 10, C]
        pred = pred * self.channel_scale + self.channel_bias

        # Auxiliary prediction (frequency bands)
        pred_aux = None
        if return_aux:
            # Use last timestep encoding for auxiliary prediction
            last_encoded = encoded[:, -1, :, :]  # [B, C, d]
            aux_flat = self.aux_pred_head(last_encoded)  # [B, C, n_future * 8]
            pred_aux = aux_flat.view(B, C, self.n_future, self.n_aux_features)
            pred_aux = pred_aux.permute(0, 2, 1, 3)  # [B, n_future, C, 8]

        if use_refinement and self.n_refinement_iters > 0:
            for _ in range(self.n_refinement_iters):
                # Create predicted features tensor
                # Use last observed auxiliary features instead of zeros to avoid
                # distribution shift (model was trained on normalized data with
                # non-zero auxiliary features)
                pred_features = x[:, -1:, :, :].expand(-1, self.n_future, -1, -1).clone()
                pred_features[:, :, :, 0] = pred  # Replace feature[0] with predictions

                # Concatenate observed and predicted sequences
                full_seq = torch.cat([x, pred_features], dim=1)  # [B, 20, C, F]

                # Re-encode the full sequence
                full_encoded = self.encode(full_seq)

                # Extract encoding for future timesteps and predict
                future_encoded = full_encoded[:, T:, :, :]  # [B, 10, C, d]
                pred = self.pred_head(future_encoded.mean(dim=1)).permute(0, 2, 1)
                pred = pred * self.channel_scale + self.channel_bias

        if return_aux:
            return pred, pred_aux
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
