"""
Training script for Neural Time-Series Forecasting Model.
Includes loss function, optimizer configuration, scheduler, and early stopping.
"""

import os
import math
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import get_model_for_monkey
from dataset import create_dataloaders


class ForecastingLoss(nn.Module):
    """
    Combined loss function for neural forecasting.
    - Primary: MSE between predictions and targets
    - Auxiliary: Temporal smoothness regularization
    """
    def __init__(self, smoothness_weight=0.05):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, T, C] where T=10
            target: Ground truth [B, T, C]

        Returns:
            Total loss value
        """
        # Primary loss: prediction MSE
        pred_loss = self.mse(pred, target)
        total_loss = pred_loss

        # Auxiliary: temporal smoothness
        if self.smoothness_weight > 0:
            diff = pred[:, 1:, :] - pred[:, :-1, :]
            smoothness_loss = (diff ** 2).mean()
            total_loss = total_loss + self.smoothness_weight * smoothness_loss

        return total_loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=15, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def get_lr_scheduler(optimizer, total_steps, warmup_fraction=0.1):
    """
    Cosine annealing learning rate scheduler with warmup.
    """
    warmup_steps = int(warmup_fraction * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    mse_fn = nn.MSELoss()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            mse = mse_fn(pred, y)

            total_loss += loss.item()
            total_mse += mse.item()
            n_batches += 1

    return total_loss / n_batches, total_mse / n_batches


def train(
    monkey_name,
    data_dir='train_data_neuro',
    output_dir='checkpoints',
    n_epochs=200,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=0.01,
    d_model=64,
    n_heads=4,
    n_layers=3,
    n_refinement_iters=2,
    dropout=0.15,
    smoothness_weight=0.05,
    patience=15,
    device=None,
    use_tensorboard=True
):
    """
    Main training function.

    Args:
        monkey_name: 'affi' or 'beignet'
        data_dir: Directory containing training data
        output_dir: Directory to save checkpoints
        n_epochs: Maximum number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        d_model: Model embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        n_refinement_iters: Number of refinement iterations
        dropout: Dropout rate
        smoothness_weight: Weight for smoothness loss
        patience: Early stopping patience
        device: Device to train on (auto-detected if None)
        use_tensorboard: Whether to log to TensorBoard
    """
    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{monkey_name}_{timestamp}"

    # Setup TensorBoard
    writer = None
    if use_tensorboard:
        log_dir = os.path.join(output_dir, 'logs', run_name)
        writer = SummaryWriter(log_dir)

    # Create dataloaders
    print(f"Loading data for monkey: {monkey_name}")
    train_loader, val_loader, test_loader = create_dataloaders(
        monkey_name,
        data_dir=data_dir,
        batch_size=batch_size,
        use_augmentation=True
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    # Detect number of channels from data
    sample_x, _ = next(iter(train_loader))
    n_channels = sample_x.shape[2]  # [B, T, C, F]
    print(f"Detected {n_channels} channels from data")

    # Create model
    model = get_model_for_monkey(
        monkey_name,
        n_channels=n_channels,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_refinement_iters=n_refinement_iters,
        dropout=dropout
    )
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98)
    )

    total_steps = n_epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, total_steps, warmup_fraction=0.1)

    # Setup loss and early stopping
    criterion = ForecastingLoss(smoothness_weight=smoothness_weight)
    checkpoint_path = os.path.join(output_dir, f'best_model_{monkey_name}.pth')
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)

    # Training loop
    print(f"\nStarting training for {n_epochs} epochs...")
    best_val_mse = float('inf')

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_mse = validate(model, val_loader, criterion, device)

        # Logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MSE/val', val_mse, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if val_mse < best_val_mse:
            best_val_mse = val_mse

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val MSE: {val_mse:.6f}")

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model and evaluate on test set
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_loss, test_mse = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f} | Test MSE: {test_mse:.6f}")

    # Save final model with config
    final_path = os.path.join(output_dir, f'model_{monkey_name}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'monkey_name': monkey_name,
            'n_channels': model.n_channels,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'n_refinement_iters': n_refinement_iters,
            'dropout': dropout,
        },
        'test_mse': test_mse,
    }, final_path)
    print(f"Saved final model to: {final_path}")

    if writer:
        writer.close()

    return model, test_mse


def main():
    parser = argparse.ArgumentParser(description='Train Neural Forecasting Model')
    parser.add_argument('--monkey', type=str, default='beignet',
                        choices=['affi', 'beignet'],
                        help='Monkey name (affi or beignet)')
    parser.add_argument('--data-dir', type=str, default='train_data_neuro',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--d-model', type=int, default=64,
                        help='Model embedding dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--n-refinement', type=int, default=2,
                        help='Number of refinement iterations')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')

    args = parser.parse_args()

    train(
        monkey_name=args.monkey,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_refinement_iters=args.n_refinement,
        dropout=args.dropout,
        patience=args.patience,
        use_tensorboard=not args.no_tensorboard
    )


if __name__ == '__main__':
    main()
