#!/usr/bin/env python
"""
Training and evaluation script with visualization.
Trains models for both monkeys and generates performance graphs.
"""

import os
import csv
import sys
import math
import json
import time
import platform
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model import get_model_for_monkey
from dataset import create_dataloaders, PerSampleNormalizer

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def get_environment_info():
    """Collect environment info for experiment reproducibility."""
    info = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        info['gpu_vram_gb'] = round(total / 1e9, 2)
    # Try to get git commit hash
    try:
        info['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        info['git_commit'] = 'unknown'
    return info


# Training mode configurations
TRAINING_MODES = {
    'mini': {
        # For laptop with 8GB RAM, no GPU, limited time
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 3,
        'learning_rate': 1e-3,
        'n_epochs': 50,
        'batch_size': {'beignet': 16, 'affi': 8},
        'patience': 15,
        'dropout': 0.15,
    },
    'mega': {
        # For high-performance PC with 32GB RAM, NVIDIA GPU, unlimited time
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'learning_rate': 3e-4,
        'n_epochs': 200,
        'batch_size': {'beignet': 64, 'affi': 16},
        'patience': 50,
        'min_epochs': 30,
        'dropout': 0.2,
    },
}


class ForecastingLoss(nn.Module):
    def __init__(self, smoothness_weight=0.0, aux_weight=0.01):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.aux_weight = aux_weight
        self.main_loss_fn = nn.HuberLoss(delta=1.0)
        self.mse = nn.MSELoss()

    def forward(self, pred, target_main, pred_aux=None, target_aux=None,
                return_components=False):
        """
        Compute loss with optional auxiliary and smoothness components.

        Args:
            pred: Main predictions [B, T, C] for feature 0
            target_main: Main targets [B, T, C] for feature 0
            pred_aux: Auxiliary predictions [B, T, C, 8] for frequency bands (optional)
            target_aux: Auxiliary targets [B, T, C, 8] for frequency bands (optional)
            return_components: If True, also return dict of individual loss components
        """
        # Main prediction loss (Huber for robust training, MSE for tracking)
        main_loss = self.main_loss_fn(pred, target_main)
        total_loss = main_loss

        # Smoothness regularization
        smoothness_loss_val = 0.0
        if self.smoothness_weight > 0:
            diff = pred[:, 1:, :] - pred[:, :-1, :]
            smoothness_loss = (diff ** 2).mean()
            smoothness_loss_val = smoothness_loss.item()
            total_loss = total_loss + self.smoothness_weight * smoothness_loss

        # Auxiliary loss for frequency bands
        aux_loss_val = 0.0
        if pred_aux is not None and target_aux is not None and self.aux_weight > 0:
            aux_loss = self.mse(pred_aux, target_aux)
            aux_loss_val = aux_loss.item()
            total_loss = total_loss + self.aux_weight * aux_loss

        if return_components:
            components = {
                'main': main_loss.item(),
                'smoothness': smoothness_loss_val,
                'aux': aux_loss_val,
            }
            return total_loss, components
        return total_loss


def get_lr_scheduler(optimizer, total_steps, warmup_fraction=0.1):
    warmup_steps = int(warmup_fraction * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_experiment_log(
    monkey_name, mode, config, device, n_channels, n_params,
    n_train, n_val, n_test, history, test_mse, test_mse_original,
    best_val_mse, best_epoch, epochs_trained, early_stopped,
    total_train_time, per_channel_mse, per_timestep_mse, pred_stats,
):
    """Save comprehensive experiment record to experiments/ directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join('experiments', f'{monkey_name}_{mode}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)

    # Build experiment record
    experiment = {
        'timestamp': timestamp,
        'total_train_time_seconds': round(total_train_time, 1),
        'environment': get_environment_info(),
        'monkey': monkey_name,
        'mode': mode,
        'config': {k: v for k, v in config.items()},
        'loss_config': {'smoothness_weight': 0.0, 'aux_weight': 0.01, 'main_loss': 'huber', 'huber_delta': 1.0},
        'device': str(device),
        'dataset': {
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'n_channels': n_channels,
            'n_features': 9,
            'observed_steps': 10,
            'future_steps': 10,
        },
        'model': {
            'n_params': n_params,
        },
        'training': {
            'epochs_trained': epochs_trained,
            'best_epoch': best_epoch,
            'early_stopped': early_stopped,
            'final_lr': history['lr'][-1] if history['lr'] else None,
        },
        'results': {
            'test_mse_normalized': float(test_mse),
            'test_mse_original': float(test_mse_original),
            'best_val_mse_normalized': float(best_val_mse),
            'per_timestep_mse_original': [float(x) for x in per_timestep_mse],
            'per_channel_mse_original': {
                'mean': float(per_channel_mse.mean()),
                'std': float(per_channel_mse.std()),
                'min': float(per_channel_mse.min()),
                'max': float(per_channel_mse.max()),
                'worst_5_channels': [int(x) for x in np.argsort(per_channel_mse)[-5:][::-1]],
                'best_5_channels': [int(x) for x in np.argsort(per_channel_mse)[:5]],
            },
            'prediction_stats': pred_stats,
        },
    }

    # Make batch_size JSON-serializable (might be a dict)
    if 'batch_size' in experiment['config'] and isinstance(experiment['config']['batch_size'], dict):
        experiment['config']['batch_size'] = experiment['config']['batch_size'].get(
            monkey_name, experiment['config']['batch_size']
        )

    # Write experiment.json
    with open(os.path.join(experiment_dir, 'experiment.json'), 'w') as f:
        json.dump(experiment, f, indent=2)

    # Write training_log.csv (epoch-by-epoch, easy to load in pandas)
    csv_path = os.path.join(experiment_dir, 'training_log.csv')
    fields = [
        'epoch', 'train_loss', 'val_loss', 'val_mse', 'val_mse_original',
        'loss_main', 'loss_smoothness', 'loss_aux',
        'lr', 'grad_norm', 'epoch_time',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(epochs_trained):
            writer.writerow([
                i,
                history['train_loss'][i],
                history['val_loss'][i],
                history['val_mse'][i],
                history['val_mse_original'][i],
                history['loss_main'][i],
                history['loss_smoothness'][i],
                history['loss_aux'][i],
                history['lr'][i],
                history['grad_norm'][i],
                history['epoch_time'][i],
            ])

    print(f"Experiment log saved to {experiment_dir}/", flush=True)
    return experiment_dir


def plot_experiment(
    experiment_dir, history, all_preds_original, all_targets_original,
    per_channel_mse, per_timestep_mse, monkey_name, best_epoch,
):
    """Generate experiment analysis plots."""

    epochs = range(len(history['train_loss']))

    # --- 1. Training curves (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(epochs, history['loss_main'], label='Main', alpha=0.8)
    ax.plot(epochs, history['loss_smoothness'], label='Smoothness', alpha=0.8)
    ax.plot(epochs, history['loss_aux'], label='Auxiliary', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['val_mse'], label='Normalized', alpha=0.8)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Validation MSE (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['val_mse_original'], color='#e74c3c', alpha=0.8)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (Original Units)')
    ax.set_title('Validation MSE (Original Units)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history['grad_norm'], color='#9b59b6', alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(epochs, history['lr'], color='#e67e22', alpha=0.6, linestyle='--')
    ax2.set_ylabel('Learning Rate', color='#e67e22')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm', color='#9b59b6')
    ax.set_title('Gradient Norm & Learning Rate')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{monkey_name.capitalize()} - Training Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 2. Prediction examples (2x3) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    n_samples = len(all_preds_original)
    for i, ax in enumerate(axes.flat):
        sample_idx = i * max(1, n_samples // 6)
        if sample_idx >= n_samples:
            sample_idx = n_samples - 1
        ch = 0
        target = all_targets_original[sample_idx, :, ch]
        pred = all_preds_original[sample_idx, :, ch]
        sample_mse = np.mean((pred - target) ** 2)
        ax.plot(range(10), target, 'b--', linewidth=2, label='Target')
        ax.plot(range(10), pred, 'r-', linewidth=2, label='Prediction')
        ax.set_title(f'Sample {sample_idx} (MSE: {sample_mse:.1f})')
        ax.set_xlabel('Future Timestep')
        ax.set_ylabel('Signal (Original)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'{monkey_name.capitalize()} - Predictions (Original Units, Ch 0)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 3. Per-channel MSE ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sorted_idx = np.argsort(per_channel_mse)

    ax = axes[0]
    ax.bar(range(len(per_channel_mse)), per_channel_mse[sorted_idx], color='#3498db', alpha=0.7)
    ax.set_xlabel('Channel (sorted by MSE)')
    ax.set_ylabel('MSE (Original Units)')
    ax.set_title(f'Per-Channel MSE (sorted)')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.boxplot(per_channel_mse, vert=True)
    ax.set_ylabel('MSE (Original Units)')
    ax.set_title(f'Channel MSE Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate worst channels
    worst_5 = sorted_idx[-5:][::-1]
    stats_text = f"Mean: {per_channel_mse.mean():.1f}\nStd: {per_channel_mse.std():.1f}\nWorst: ch {worst_5[0]} ({per_channel_mse[worst_5[0]]:.1f})"
    ax.text(1.3, per_channel_mse.max() * 0.9, stats_text, fontsize=9, verticalalignment='top')

    plt.suptitle(f'{monkey_name.capitalize()} - Per-Channel Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'channel_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 4. Per-timestep MSE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(10), per_timestep_mse, color='#e74c3c', alpha=0.7)
    for bar, mse in zip(bars, per_timestep_mse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + per_timestep_mse.max() * 0.01,
                f'{mse:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Future Timestep (0 = next, 9 = furthest)')
    ax.set_ylabel('MSE (Original Units)')
    ax.set_title(f'{monkey_name.capitalize()} - Per-Timestep MSE')
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'timestep_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Experiment plots saved to {experiment_dir}/", flush=True)


def train_and_evaluate(
    monkey_name,
    data_dir='train_data_neuro',
    mode='mini',
    device=None,
    use_tensorboard=True,
    **overrides
):
    """Train model and return metrics history.

    Args:
        monkey_name: 'beignet' or 'affi'
        data_dir: Directory containing training data
        mode: 'mini' (laptop, quick testing) or 'mega' (high-performance, full training)
        device: Torch device (auto-detected if None)
        use_tensorboard: Whether to log to TensorBoard
        **overrides: Override any mode config value (e.g., n_epochs=100)
    """
    # Get mode configuration
    if mode not in TRAINING_MODES:
        raise ValueError(f"Unknown mode: {mode}. Use 'mini' or 'mega'.")
    config = TRAINING_MODES[mode].copy()

    # Apply any overrides
    for key, value in overrides.items():
        if key in config:
            config[key] = value

    # Extract config values
    d_model = config['d_model']
    n_heads = config['n_heads']
    n_layers = config['n_layers']
    learning_rate = config['learning_rate']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size'][monkey_name]
    patience = config['patience']
    min_epochs = config.get('min_epochs', 0)
    dropout = config['dropout']
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f"\n{'='*60}", flush=True)
    print(f"Training model for: {monkey_name} (mode: {mode})", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Model: d={d_model}, heads={n_heads}, layers={n_layers}", flush=True)
    print(f"Training: lr={learning_rate}, epochs={n_epochs}, batch={batch_size}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        monkey_name, data_dir=data_dir, batch_size=batch_size
    )

    # Get number of channels from data
    sample_x, _, _, _ = next(iter(train_loader))
    n_channels = sample_x.shape[2]

    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples: {len(val_loader.dataset)}", flush=True)
    print(f"Test samples: {len(test_loader.dataset)}", flush=True)
    print(f"Channels: {n_channels}", flush=True)

    # Create model
    model = get_model_for_monkey(
        monkey_name,
        n_channels=n_channels,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}", flush=True)

    # Setup TensorBoard
    writer = None
    if use_tensorboard:
        log_dir = os.path.join('runs', f'{monkey_name}_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}", flush=True)

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = n_epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, total_steps)
    criterion = ForecastingLoss(smoothness_weight=0.0, aux_weight=0.01)

    # Training history (per-epoch)
    history = {
        'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_mse_original': [],
        'lr': [], 'grad_norm': [], 'epoch_time': [],
        'loss_main': [], 'loss_smoothness': [], 'loss_aux': [],
    }

    best_val_mse = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    train_start_time = time.time()

    print(f"\nStarting training...", flush=True)

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        epoch_main = 0.0
        epoch_smooth = 0.0
        epoch_aux = 0.0
        epoch_grad_norm = 0.0
        for x, y, mean, std in train_loader:
            x, y = x.to(device), y.to(device)
            # Split target: y is [B, T, C, F], we need main (feature 0) and aux (features 1-8)
            y_main = y[:, :, :, 0]  # [B, T, C]
            y_aux = y[:, :, :, 1:]  # [B, T, C, 8]

            optimizer.zero_grad()
            pred, pred_aux = model(x, y=y, return_aux=True)
            loss, components = criterion(
                pred, y_main, pred_aux, y_aux, return_components=True
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            epoch_main += components['main']
            epoch_smooth += components['smoothness']
            epoch_aux += components['aux']
            epoch_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        n_batches = len(train_loader)
        train_loss /= n_batches
        epoch_main /= n_batches
        epoch_smooth /= n_batches
        epoch_aux /= n_batches
        epoch_grad_norm /= n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_mse_original = 0.0
        mse_fn = nn.MSELoss()
        normalizer = PerSampleNormalizer()
        with torch.no_grad():
            for x, y, mean, std in val_loader:
                x, y = x.to(device), y.to(device)
                mean, std = mean.to(device), std.to(device)
                y_main = y[:, :, :, 0]
                y_aux = y[:, :, :, 1:]

                pred, pred_aux = model(x, return_aux=True)
                val_loss += criterion(pred, y_main, pred_aux, y_aux).item()
                val_mse += mse_fn(pred, y_main).item()

                # MSE in original units (what the challenge grades on)
                pred_orig = normalizer.denormalize(pred, mean, std)
                y_orig = normalizer.denormalize(y_main, mean, std)
                val_mse_original += mse_fn(pred_orig, y_orig).item()
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_mse_original /= len(val_loader)

        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_mse_original'].append(val_mse_original)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['grad_norm'].append(epoch_grad_norm)
        history['epoch_time'].append(epoch_time)
        history['loss_main'].append(epoch_main)
        history['loss_smoothness'].append(epoch_smooth)
        history['loss_aux'].append(epoch_aux)

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss/main', epoch_main, epoch)
            writer.add_scalar('Loss/smoothness', epoch_smooth, epoch)
            writer.add_scalar('Loss/aux', epoch_aux, epoch)
            writer.add_scalar('MSE/val_normalized', val_mse, epoch)
            writer.add_scalar('MSE/val_original', val_mse_original, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('GradNorm', epoch_grad_norm, epoch)

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val MSE: {val_mse:.6f} | Original MSE: {val_mse_original:.2f} | {epoch_time:.1f}s", flush=True)

        if patience_counter >= patience and epoch >= min_epochs:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    total_train_time = time.time() - train_start_time
    epochs_trained = len(history['train_loss'])

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()

    test_mse = 0.0
    test_mse_original = 0.0
    all_preds = []
    all_targets = []
    all_preds_original = []
    all_targets_original = []
    normalizer = PerSampleNormalizer()

    with torch.no_grad():
        for x, y, mean, std in test_loader:
            x, y = x.to(device), y.to(device)
            mean, std = mean.to(device), std.to(device)
            y_main = y[:, :, :, 0]  # Only feature 0 for test MSE

            pred = model(x, return_aux=False)
            test_mse += mse_fn(pred, y_main).item()

            # Denormalize to original units
            pred_orig = normalizer.denormalize(pred, mean, std)
            y_orig = normalizer.denormalize(y_main, mean, std)
            test_mse_original += mse_fn(pred_orig, y_orig).item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_main.cpu().numpy())
            all_preds_original.append(pred_orig.cpu().numpy())
            all_targets_original.append(y_orig.cpu().numpy())

    test_mse /= len(test_loader)
    test_mse_original /= len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds_original = np.concatenate(all_preds_original, axis=0)
    all_targets_original = np.concatenate(all_targets_original, axis=0)

    # Per-channel MSE (original units) — which electrodes are hardest?
    # all_preds_original: [N, T=10, C]
    per_channel_mse = np.mean((all_preds_original - all_targets_original) ** 2, axis=(0, 1))  # [C]

    # Per-timestep MSE (original units) — which future steps are hardest?
    per_timestep_mse = np.mean((all_preds_original - all_targets_original) ** 2, axis=(0, 2))  # [T=10]

    # Prediction statistics (original units)
    pred_stats = {
        'pred_mean': float(all_preds_original.mean()),
        'pred_std': float(all_preds_original.std()),
        'pred_min': float(all_preds_original.min()),
        'pred_max': float(all_preds_original.max()),
        'target_mean': float(all_targets_original.mean()),
        'target_std': float(all_targets_original.std()),
        'target_min': float(all_targets_original.min()),
        'target_max': float(all_targets_original.max()),
    }

    print(f"\nTest MSE (normalized): {test_mse:.6f}", flush=True)
    print(f"Test MSE (original units): {test_mse_original:.2f}", flush=True)
    print(f"Best Val MSE (normalized): {best_val_mse:.6f} (epoch {best_epoch})", flush=True)
    print(f"Total training time: {total_train_time:.1f}s", flush=True)

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'config': {
            'monkey_name': monkey_name,
            'mode': mode,
            'n_channels': n_channels,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
        },
        'test_mse': test_mse,
        'test_mse_original': test_mse_original,
        'best_val_mse': best_val_mse,
    }, f'checkpoints/model_{monkey_name}_{mode}.pth')

    # Save experiment log
    experiment_dir = save_experiment_log(
        monkey_name=monkey_name,
        mode=mode,
        config=config,
        device=device,
        n_channels=n_channels,
        n_params=n_params,
        n_train=len(train_loader.dataset),
        n_val=len(val_loader.dataset),
        n_test=len(test_loader.dataset),
        history=history,
        test_mse=test_mse,
        test_mse_original=test_mse_original,
        best_val_mse=best_val_mse,
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
        early_stopped=patience_counter >= patience,
        total_train_time=total_train_time,
        per_channel_mse=per_channel_mse,
        per_timestep_mse=per_timestep_mse,
        pred_stats=pred_stats,
    )

    # Generate experiment plots
    plot_experiment(
        experiment_dir=experiment_dir,
        history=history,
        all_preds_original=all_preds_original,
        all_targets_original=all_targets_original,
        per_channel_mse=per_channel_mse,
        per_timestep_mse=per_timestep_mse,
        monkey_name=monkey_name,
        best_epoch=best_epoch,
    )

    # Close TensorBoard writer
    if writer:
        writer.close()

    # Clear GPU memory
    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'mode': mode,
        'history': history,
        'test_mse': test_mse,
        'test_mse_original': test_mse_original,
        'best_val_mse': best_val_mse,
        'predictions': all_preds,
        'targets': all_targets,
        'predictions_original': all_preds_original,
        'targets_original': all_targets_original,
        'n_params': n_params,
        'n_channels': n_channels,
        'experiment_dir': experiment_dir,
    }


def plot_results(results, output_dir='results'):
    """Generate performance visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {'beignet': '#2ecc71', 'affi': '#3498db'}

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for monkey, data in results.items():
        ax.plot(data['history']['train_loss'], label=f'{monkey}', color=colors[monkey], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for monkey, data in results.items():
        ax.plot(data['history']['val_loss'], label=f'{monkey}', color=colors[monkey], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation MSE
    ax = axes[0, 2]
    for monkey, data in results.items():
        ax.plot(data['history']['val_mse'], label=f'{monkey}', color=colors[monkey], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Validation MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning Rate
    ax = axes[1, 0]
    for monkey, data in results.items():
        ax.plot(data['history']['lr'], label=f'{monkey}', color=colors[monkey], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Test MSE Comparison
    ax = axes[1, 1]
    monkeys = list(results.keys())
    test_mses = [results[m]['test_mse'] for m in monkeys]
    bars = ax.bar(monkeys, test_mses, color=[colors[m] for m in monkeys])
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE by Monkey')
    for bar, mse in zip(bars, test_mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Example Predictions
    ax = axes[1, 2]
    # Pick first monkey for example
    monkey = list(results.keys())[0]
    preds = results[monkey]['predictions']
    targets = results[monkey]['targets']

    # Plot first sample, first few channels
    sample_idx = 0
    for ch in range(min(3, preds.shape[2])):
        ax.plot(targets[sample_idx, :, ch], '--', alpha=0.7, label=f'Target Ch{ch}' if ch == 0 else '')
        ax.plot(preds[sample_idx, :, ch], '-', alpha=0.7, label=f'Pred Ch{ch}' if ch == 0 else '')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Signal')
    ax.set_title(f'Example Predictions ({monkey}, sample 0)')
    ax.legend(['Target', 'Prediction'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create prediction comparison plots for each monkey
    for monkey, data in results.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        preds = data['predictions']
        targets = data['targets']

        for i, ax in enumerate(axes.flat):
            if i >= len(preds):
                break
            sample_idx = i * (len(preds) // 6)
            ch = 0  # First channel

            ax.plot(range(10), targets[sample_idx, :, ch], 'b--', linewidth=2, label='Target')
            ax.plot(range(10), preds[sample_idx, :, ch], 'r-', linewidth=2, label='Prediction')

            mse = np.mean((preds[sample_idx, :, ch] - targets[sample_idx, :, ch])**2)
            ax.set_title(f'Sample {sample_idx} (MSE: {mse:.4f})')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Signal')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Prediction Examples - {monkey.capitalize()}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/predictions_{monkey}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to {output_dir}/", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train neural forecasting models')
    parser.add_argument('--mode', type=str, default='mini', choices=['mini', 'mega'],
                        help='Training mode: mini (laptop) or mega (high-performance)')
    parser.add_argument('--monkey', type=str, default=None, choices=['beignet', 'affi'],
                        help='Train only one monkey (default: both)')
    args = parser.parse_args()

    mode = args.mode
    monkeys = [args.monkey] if args.monkey else ['beignet', 'affi']

    print(f"\n{'='*60}", flush=True)
    print(f"TRAINING MODE: {mode.upper()}", flush=True)
    print(f"Config: {TRAINING_MODES[mode]}", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = {}

    # Train models
    for monkey in monkeys:
        results[monkey] = train_and_evaluate(monkey, mode=mode)

    # Generate plots
    plot_results(results, output_dir=f'results_{mode}')

    # Print summary
    print("\n" + "="*60, flush=True)
    print(f"TRAINING SUMMARY ({mode.upper()} MODE)", flush=True)
    print("="*60, flush=True)

    for monkey, data in results.items():
        print(f"\n{monkey.upper()}:", flush=True)
        print(f"  Channels: {data['n_channels']}", flush=True)
        print(f"  Parameters: {data['n_params']:,}", flush=True)
        print(f"  Best Val MSE (normalized): {data['best_val_mse']:.6f}", flush=True)
        print(f"  Test MSE (normalized): {data['test_mse']:.6f}", flush=True)
        print(f"  Test MSE (original units): {data['test_mse_original']:.2f}", flush=True)
        print(f"  Epochs trained: {len(data['history']['train_loss'])}", flush=True)

    # Save summary JSON
    output_dir = f'results_{mode}'
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        'mode': mode,
        'config': TRAINING_MODES[mode],
        'results': {
            monkey: {
                'n_channels': data['n_channels'],
                'n_params': data['n_params'],
                'best_val_mse': float(data['best_val_mse']),
                'test_mse': float(data['test_mse']),
                'test_mse_original': float(data['test_mse_original']),
                'epochs_trained': len(data['history']['train_loss']),
            }
            for monkey, data in results.items()
        }
    }

    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # List experiment directories
    print("\nExperiment logs (git-tracked):", flush=True)
    for monkey, data in results.items():
        if 'experiment_dir' in data:
            print(f"  {data['experiment_dir']}/", flush=True)

    print("\n" + "="*60, flush=True)
    print("Training complete!", flush=True)
    print("Experiment logs saved to experiments/ (push to git for remote analysis)", flush=True)
    print("Plots also saved to " + f"results_{mode}/", flush=True)
    print("View TensorBoard logs with: tensorboard --logdir runs", flush=True)
    print("="*60, flush=True)


if __name__ == '__main__':
    main()
