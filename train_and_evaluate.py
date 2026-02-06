#!/usr/bin/env python
"""
Training and evaluation script with visualization.
Trains models for both monkeys and generates performance graphs.
"""

import os
import sys
import math
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model import get_model_for_monkey
from dataset import create_dataloaders

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)


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
        'n_refinement_iters': 0,
    },
    'mega': {
        # For high-performance PC with 32GB RAM, NVIDIA GPU, unlimited time
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'learning_rate': 3e-4,
        'n_epochs': 200,
        'batch_size': {'beignet': 64, 'affi': 64},
        'patience': 30,
        'dropout': 0.2,
        'n_refinement_iters': 1,
    },
}


class ForecastingLoss(nn.Module):
    def __init__(self, smoothness_weight=0.01, aux_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.aux_weight = aux_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target_main, pred_aux=None, target_aux=None):
        """
        Compute loss with optional auxiliary and smoothness components.

        Args:
            pred: Main predictions [B, T, C] for feature 0
            target_main: Main targets [B, T, C] for feature 0
            pred_aux: Auxiliary predictions [B, T, C, 8] for frequency bands (optional)
            target_aux: Auxiliary targets [B, T, C, 8] for frequency bands (optional)
        """
        # Main prediction loss
        main_loss = self.mse(pred, target_main)
        total_loss = main_loss

        # Smoothness regularization (reduced from 0.05 to 0.01 for sharper predictions)
        if self.smoothness_weight > 0:
            diff = pred[:, 1:, :] - pred[:, :-1, :]
            smoothness_loss = (diff ** 2).mean()
            total_loss = total_loss + self.smoothness_weight * smoothness_loss

        # Auxiliary loss for frequency bands
        if pred_aux is not None and target_aux is not None and self.aux_weight > 0:
            aux_loss = self.mse(pred_aux, target_aux)
            total_loss = total_loss + self.aux_weight * aux_loss

        return total_loss


def get_lr_scheduler(optimizer, total_steps, warmup_fraction=0.1):
    warmup_steps = int(warmup_fraction * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    dropout = config['dropout']
    n_refinement_iters = config['n_refinement_iters']

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
    sample_x, _ = next(iter(train_loader))
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
        n_refinement_iters=n_refinement_iters,
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
    criterion = ForecastingLoss(smoothness_weight=0.01, aux_weight=0.1)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'lr': []
    }

    best_val_mse = float('inf')
    best_model_state = None
    patience_counter = 0

    print(f"\nStarting training...", flush=True)

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Split target: y is [B, T, C, F], we need main (feature 0) and aux (features 1-8)
            y_main = y[:, :, :, 0]  # [B, T, C]
            y_aux = y[:, :, :, 1:]  # [B, T, C, 8]

            optimizer.zero_grad()
            pred, pred_aux = model(x, use_refinement=False, return_aux=True)
            loss = criterion(pred, y_main, pred_aux, y_aux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        mse_fn = nn.MSELoss()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_main = y[:, :, :, 0]
                y_aux = y[:, :, :, 1:]

                pred, pred_aux = model(x, use_refinement=False, return_aux=True)
                val_loss += criterion(pred, y_main, pred_aux, y_aux).item()
                val_mse += mse_fn(pred, y_main).item()  # MSE only on main task
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MSE/val', val_mse, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | MSE: {val_mse:.6f}", flush=True)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()

    test_mse = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_main = y[:, :, :, 0]  # Only feature 0 for test MSE

            pred = model(x, use_refinement=False, return_aux=False)
            test_mse += mse_fn(pred, y_main).item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_main.cpu().numpy())

    test_mse /= len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"\nTest MSE: {test_mse:.6f}", flush=True)
    print(f"Best Val MSE: {best_val_mse:.6f}", flush=True)

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
            'n_refinement_iters': n_refinement_iters,
            'dropout': dropout,
        },
        'test_mse': test_mse,
        'best_val_mse': best_val_mse,
    }, f'checkpoints/model_{monkey_name}_{mode}.pth')

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
        'best_val_mse': best_val_mse,
        'predictions': all_preds,
        'targets': all_targets,
        'n_params': n_params,
        'n_channels': n_channels,
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
        print(f"  Best Val MSE: {data['best_val_mse']:.6f}", flush=True)
        print(f"  Test MSE: {data['test_mse']:.6f}", flush=True)
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
                'epochs_trained': len(data['history']['train_loss']),
            }
            for monkey, data in results.items()
        }
    }

    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60, flush=True)
    print("Training complete! Results saved to results/", flush=True)
    print("View TensorBoard logs with: tensorboard --logdir runs", flush=True)
    print("="*60, flush=True)


if __name__ == '__main__':
    main()
