"""
Neural Time-Series Forecasting Model
Factorized Spatiotemporal Encoder with Iterative Refinement
"""

import math
import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, output_dim=64):
        super().__init__()
        half_dim = output_dim // 2
        self.target_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, half_dim), nn.LayerNorm(half_dim),
        )
        self.freq_encoder = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, half_dim), nn.LayerNorm(half_dim),
        )
        self.fusion = nn.Sequential(nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, x):
        target = self.target_encoder(x[..., :1])
        freq = self.freq_encoder(x[..., 1:])
        return self.fusion(torch.cat([target, freq], dim=-1))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).unsqueeze(2))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :, :]


class FactorizedSpatiotemporalBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, ff_mult=4, use_temporal_conv=False):
        super().__init__()
        self.use_temporal_conv = use_temporal_conv
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)
        if use_temporal_conv:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.Dropout(dropout),
            )
            self.conv_gate = nn.Parameter(torch.tensor(0.0))
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model), nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temporal_mask=None):
        B, T, C, d = x.shape
        x_t = x.permute(0, 2, 1, 3).reshape(B * C, T, d)
        attn_out, _ = self.temporal_attn(x_t, x_t, x_t, attn_mask=temporal_mask)
        attn_out = self.dropout(attn_out)
        if self.use_temporal_conv:
            conv_out = self.temporal_conv(x_t.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(self.conv_gate)
            combined = (1 - gate) * attn_out + gate * conv_out
            x_t = self.temporal_norm(x_t + combined)
        else:
            x_t = self.temporal_norm(x_t + attn_out)
        x = x_t.reshape(B, C, T, d).permute(0, 2, 1, 3)
        x_s = x.reshape(B * T, C, d)
        attn_out, _ = self.spatial_attn(x_s, x_s, x_s)
        x_s = self.spatial_norm(x_s + self.dropout(attn_out))
        x = x_s.reshape(B, T, C, d)
        x = self.ffn_norm(x + self.ffn(x))
        return x


class PredictionHead(nn.Module):
    def __init__(self, d_model, n_future=10, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_future = n_future
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_future))

    def forward(self, x):
        if x.dim() == 3:
            return self.head(x)
        B, T, C, d = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * C, T, d)
        query = self.query.expand(B * C, 1, d)
        attended, _ = self.attn(query, x_flat, x_flat)
        attended = self.attn_norm(query + attended)
        attended = attended.squeeze(1).reshape(B, C, d)
        return self.head(attended)


class NeuralForecaster(nn.Module):
    def __init__(self, n_channels, n_features=9, d_model=64, n_heads=4, n_layers=3,
                 n_future=10, n_refinement_iters=2, dropout=0.15,
                 use_dual_heads=False, use_temporal_conv=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.d_model = d_model
        self.n_future = n_future
        self.n_refinement_iters = n_refinement_iters
        self.use_dual_heads = use_dual_heads
        self.feature_encoder = FeatureEncoder(n_features, 32, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, n_channels, d_model) * 0.02)
        self.channel_scale = nn.Parameter(torch.ones(1, 1, n_channels))
        self.channel_bias = nn.Parameter(torch.zeros(1, 1, n_channels))
        self.blocks = nn.ModuleList([
            FactorizedSpatiotemporalBlock(d_model, n_heads, dropout, use_temporal_conv=use_temporal_conv)
            for _ in range(n_layers)
        ])
        if use_dual_heads:
            self.near_head = PredictionHead(d_model, n_future, dropout=dropout)
            self.far_head = PredictionHead(d_model, n_future, dropout=dropout)
            self.head_gate = nn.Parameter(torch.linspace(-1.0, 1.0, n_future))
        else:
            self.pred_head = PredictionHead(d_model, n_future, dropout=dropout)
        self.aux_pred_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_future * (n_features - 1)),
        )
        self.n_aux_features = n_features - 1

    def encode(self, x, causal=True):
        z = self.feature_encoder(x)
        z = self.pos_encoding(z)
        z = z + self.channel_embedding[:, :, :z.size(2), :]
        T = z.size(1)
        temporal_mask = None
        if causal:
            temporal_mask = torch.triu(torch.ones(T, T, device=z.device) * float('-inf'), diagonal=1)
        for block in self.blocks:
            z = block(z, temporal_mask=temporal_mask)
        return z

    def _apply_pred_head(self, encoded):
        if self.use_dual_heads:
            near_pred = self.near_head(encoded)
            far_pred = self.far_head(encoded)
            gate = torch.sigmoid(self.head_gate)
            return (1 - gate) * near_pred + gate * far_pred
        else:
            return self.pred_head(encoded)

    def forward(self, x, use_refinement=True, return_aux=False):
        B, T, C, F = x.shape
        encoded = self.encode(x)
        pred = self._apply_pred_head(encoded).permute(0, 2, 1)
        pred = pred * self.channel_scale + self.channel_bias
        pred_aux = None
        if return_aux:
            last_encoded = encoded[:, -1, :, :]
            aux_flat = self.aux_pred_head(last_encoded)
            pred_aux = aux_flat.view(B, C, self.n_future, self.n_aux_features).permute(0, 2, 1, 3)
        if use_refinement and self.n_refinement_iters > 0:
            for _ in range(self.n_refinement_iters):
                pred_features = x[:, -1:, :, :].expand(-1, self.n_future, -1, -1).clone()
                pred_features[:, :, :, 0] = pred
                full_seq = torch.cat([x, pred_features], dim=1)
                full_encoded = self.encode(full_seq)
                future_encoded = full_encoded[:, T:, :, :]
                pred = self._apply_pred_head(future_encoded.mean(dim=1)).permute(0, 2, 1)
                pred = pred * self.channel_scale + self.channel_bias
        if return_aux:
            return pred, pred_aux
        return pred
