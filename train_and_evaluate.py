"""
Training and evaluation script with visualization.
Trains models for both monkeys and generates performance graphs.
"""

import os
import math
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import get_model_for_monkey
from dataset import create_dataloaders


class ForecastingLoss(nn.Module):
    def __init__(self, smoothness_weight=0.05):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_loss = self.mse(pred, target)
        total_loss = pred_loss
        if self.smoothness_weight > 0:
            diff = pred[:, 1:, :] - pred[:, :-1, :]
            smoothness_loss = (diff ** 2).mean()
            total_loss = total_loss + self.smoothness_weight * smoothness_loss
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
    n_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    device=None
):
    """Train model and return metrics history."""

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"Training model for: {monkey_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        monkey_name, data_dir=data_dir, batch_size=batch_size
    )

    # Get number of channels from data
    sample_x, _ = next(iter(train_loader))
    n_channels = sample_x.shape[2]

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Channels: {n_channels}")

    # Create model
    model = get_model_for_monkey(
        monkey_name,
        n_channels=n_channels,
        d_model=64,
        n_heads=4,
        n_layers=3,
        n_refinement_iters=2,
        dropout=0.15
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = n_epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, total_steps)
    criterion = ForecastingLoss(smoothness_weight=0.05)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'lr': []
    }

    best_val_mse = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
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
                pred = model(x)
                val_loss += criterion(pred, y).item()
                val_mse += mse_fn(pred, y).item()
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | MSE: {val_mse:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
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
            pred = model(x)
            test_mse += mse_fn(pred, y).item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    test_mse /= len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"\nTest MSE: {test_mse:.6f}")
    print(f"Best Val MSE: {best_val_mse:.6f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'config': {
            'monkey_name': monkey_name,
            'n_channels': n_channels,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 3,
        },
        'test_mse': test_mse,
        'best_val_mse': best_val_mse,
    }, f'checkpoints/model_{monkey_name}.pth')

    return {
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

    print(f"\nPlots saved to {output_dir}/")


def main():
    results = {}

    # Train both models
    for monkey in ['beignet', 'affi']:
        results[monkey] = train_and_evaluate(
            monkey,
            n_epochs=100,
            batch_size=32,
            learning_rate=1e-3
        )

    # Generate plots
    plot_results(results)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for monkey, data in results.items():
        print(f"\n{monkey.upper()}:")
        print(f"  Channels: {data['n_channels']}")
        print(f"  Parameters: {data['n_params']:,}")
        print(f"  Best Val MSE: {data['best_val_mse']:.6f}")
        print(f"  Test MSE: {data['test_mse']:.6f}")
        print(f"  Epochs trained: {len(data['history']['train_loss'])}")

    # Save summary JSON
    summary = {
        monkey: {
            'n_channels': data['n_channels'],
            'n_params': data['n_params'],
            'best_val_mse': float(data['best_val_mse']),
            'test_mse': float(data['test_mse']),
            'epochs_trained': len(data['history']['train_loss']),
        }
        for monkey, data in results.items()
    }

    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("Training complete! Results saved to results/")
    print("="*60)


if __name__ == '__main__':
    main()
