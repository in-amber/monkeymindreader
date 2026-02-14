#!/usr/bin/env python
"""
Hyperparameter search with successive halving.

Finds optimal hyperparameters efficiently by progressively eliminating
poor-performing configs. Starts many configs with short training,
keeps the top half, extends training, and repeats.

Usage:
    python hyperparam_search.py --monkey beignet --n-configs 16
    python hyperparam_search.py --monkey affi --n-configs 8
"""

import os
import sys
import json
import math
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from model import get_model_for_monkey
from dataset import create_dataloaders, PerSampleNormalizer

sys.stdout.reconfigure(line_buffering=True)

# --- Search space ---
SEARCH_SPACE = {
    'learning_rate': ('log_uniform', 1e-4, 1e-3),
    'weight_decay': ('log_uniform', 0.005, 0.1),
    'dropout': ('uniform', 0.1, 0.3),
    'd_model': ('choice', [96, 128, 192]),
    'n_layers': ('choice', [3, 4, 5, 6]),
    'huber_delta': ('choice', [0.5, 1.0, 2.0]),
    'timestep_weight_max': ('uniform', 1.0, 3.0),
}

# Current best config (always included as config #0 for baseline comparison)
BASELINE_CONFIG = {
    'learning_rate': 3e-4,
    'weight_decay': 0.01,
    'dropout': 0.2,
    'd_model': 128,
    'n_layers': 4,
    'huber_delta': 1.0,
    'timestep_weight_max': 2.0,
}


def sample_config(rng):
    """Sample a random config from the search space."""
    config = {}
    for key, spec in SEARCH_SPACE.items():
        dist_type = spec[0]
        if dist_type == 'log_uniform':
            lo, hi = spec[1], spec[2]
            config[key] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        elif dist_type == 'uniform':
            config[key] = rng.uniform(spec[1], spec[2])
        elif dist_type == 'choice':
            config[key] = rng.choice(spec[1])
    return config


def get_n_heads(d_model):
    """Derive n_heads from d_model ensuring divisibility."""
    if d_model >= 128:
        return 8
    return 4


def make_lr_scheduler(optimizer, total_steps, warmup_fraction=0.1):
    """Cosine decay with linear warmup."""
    warmup_steps = int(warmup_fraction * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_config(config, monkey_name, train_loader, val_loader, n_channels,
                 max_epochs, device, use_early_stopping=False):
    """
    Train a single config and return best val MSE.

    Args:
        config: Hyperparameter dict
        monkey_name: 'beignet' or 'affi'
        train_loader, val_loader: Data loaders
        n_channels: Number of electrode channels
        max_epochs: Maximum epochs to train
        device: Torch device
        use_early_stopping: If True, use patience=50/min_epochs=30.
                            If False, train for exactly max_epochs (screening mode).
    Returns:
        best_val_mse, best_val_mse_orig, epochs_run
    """
    d_model = int(config['d_model'])
    n_heads = get_n_heads(d_model)
    n_layers = int(config['n_layers'])
    dropout = config['dropout']
    lr = config['learning_rate']
    wd = config['weight_decay']
    huber_delta = config['huber_delta']
    timestep_weight_max = config['timestep_weight_max']

    model = get_model_for_monkey(
        monkey_name, n_channels=n_channels,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        n_refinement_iters=0, dropout=dropout,
        use_dual_heads=True, use_temporal_conv=True,
    ).to(device)

    # Gradient accumulation (match effective batch 64)
    batch_size = train_loader.batch_size
    effective_batch_size = 64
    accumulation_steps = max(1, effective_batch_size // batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = max_epochs * (len(train_loader) // accumulation_steps)
    scheduler = make_lr_scheduler(optimizer, total_steps)

    huber = nn.HuberLoss(delta=huber_delta, reduction='none')
    mse_fn = nn.MSELoss()
    normalizer = PerSampleNormalizer()

    best_val_mse = float('inf')
    best_val_mse_orig = float('inf')
    patience_counter = 0
    patience = 50 if use_early_stopping else max_epochs
    min_epochs = 30 if use_early_stopping else 0

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        optimizer.zero_grad()
        for batch_idx, (x, y, mean, std) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y_main = y[:, :, :, 0]

            pred = model(x, use_refinement=False, return_aux=False)

            elementwise_loss = huber(pred, y_main)
            if timestep_weight_max > 1.0:
                T = pred.size(1)
                weights = torch.linspace(1.0, timestep_weight_max, T, device=device)
                weights = weights / weights.mean()
                elementwise_loss = elementwise_loss * weights[None, :, None]
            loss = elementwise_loss.mean()

            (loss / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # --- Validation ---
        model.eval()
        val_mse = 0.0
        val_mse_orig = 0.0
        with torch.no_grad():
            for x, y, mean, std in val_loader:
                x, y = x.to(device), y.to(device)
                mean, std = mean.to(device), std.to(device)
                y_main = y[:, :, :, 0]

                pred = model(x, use_refinement=False, return_aux=False)
                val_mse += mse_fn(pred, y_main).item()

                pred_orig = normalizer.denormalize(pred, mean, std)
                y_orig = normalizer.denormalize(y_main, mean, std)
                val_mse_orig += mse_fn(pred_orig, y_orig).item()

        val_mse /= len(val_loader)
        val_mse_orig /= len(val_loader)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_mse_orig = val_mse_orig
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch >= min_epochs:
            break

    # Cleanup
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    return best_val_mse, best_val_mse_orig, epoch + 1


def successive_halving(monkey_name, n_configs=16, seed=42, data_dir='train_data_neuro'):
    """
    Run successive halving hyperparameter search.

    Schedule: 4 rounds with halving, epoch budgets [30, 60, 120, 200].
    Each round eliminates the bottom 50% of configs.
    Final round uses early stopping for best possible result.
    """
    rounds = [
        {'max_epochs': 30, 'keep_fraction': 0.5, 'early_stopping': False},
        {'max_epochs': 60, 'keep_fraction': 0.5, 'early_stopping': False},
        {'max_epochs': 120, 'keep_fraction': 0.5, 'early_stopping': False},
        {'max_epochs': 200, 'keep_fraction': 1.0, 'early_stopping': True},
    ]

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SEARCH: {monkey_name}")
    print(f"Device: {device}")
    print(f"Configs: {n_configs}, Rounds: {len(rounds)}")
    schedule_str = ' -> '.join(str(r['max_epochs']) + 'ep' for r in rounds)
    print(f"Schedule: {schedule_str}")
    print(f"{'='*60}\n")

    # Create dataloaders (once, shared across all configs)
    batch_size = 64 if monkey_name == 'beignet' else 16
    train_loader, val_loader, _ = create_dataloaders(
        monkey_name, data_dir=data_dir, batch_size=batch_size
    )

    sample_x, _, _, _ = next(iter(train_loader))
    n_channels = sample_x.shape[2]
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {n_channels} channels\n")

    # Sample configs (config #0 is always the current baseline)
    rng = random.Random(seed)
    configs = [BASELINE_CONFIG.copy()]
    for _ in range(n_configs - 1):
        configs.append(sample_config(rng))

    all_results = []
    search_start = time.time()

    for round_idx, round_spec in enumerate(rounds):
        max_epochs = round_spec['max_epochs']
        keep_fraction = round_spec['keep_fraction']
        use_early_stopping = round_spec['early_stopping']
        n_keep = max(1, int(len(configs) * keep_fraction))

        print(f"\n{'='*60}")
        print(f"ROUND {round_idx + 1}/{len(rounds)}: "
              f"{len(configs)} configs x {max_epochs} epochs"
              f"{' (early stopping)' if use_early_stopping else ''}")
        print(f"{'='*60}")

        round_results = []

        for i, config in enumerate(configs):
            config_str = (f"lr={config['learning_rate']:.2e} "
                         f"d={int(config['d_model'])} L={int(config['n_layers'])} "
                         f"drop={config['dropout']:.2f} wd={config['weight_decay']:.3f} "
                         f"delta={config['huber_delta']} tw={config['timestep_weight_max']:.1f}")
            print(f"\n  [{i+1}/{len(configs)}] {config_str}", flush=True)

            start_time = time.time()
            try:
                val_mse, val_mse_orig, epochs_run = train_config(
                    config, monkey_name, train_loader, val_loader,
                    n_channels, max_epochs, device,
                    use_early_stopping=use_early_stopping,
                )
                elapsed = time.time() - start_time
                print(f"           Val MSE: {val_mse:.6f} | Orig: {val_mse_orig:.0f} | "
                      f"Epochs: {epochs_run} | {elapsed:.0f}s", flush=True)

                round_results.append({
                    'config': config.copy(),
                    'val_mse': val_mse,
                    'val_mse_orig': val_mse_orig,
                    'epochs_run': epochs_run,
                    'time': round(elapsed, 1),
                    'round': round_idx + 1,
                })
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"           FAILED ({elapsed:.0f}s): {e}", flush=True)
                round_results.append({
                    'config': config.copy(),
                    'val_mse': float('inf'),
                    'val_mse_orig': float('inf'),
                    'epochs_run': 0,
                    'time': round(elapsed, 1),
                    'round': round_idx + 1,
                    'error': str(e),
                })

        # Rank by val MSE and keep top fraction
        round_results.sort(key=lambda r: r['val_mse'])
        all_results.extend(round_results)

        # Print round rankings
        print(f"\n  Round {round_idx + 1} rankings:")
        for j, r in enumerate(round_results):
            status = "KEPT" if j < n_keep else "eliminated"
            marker = ""
            if r['config'] == BASELINE_CONFIG:
                marker = " [BASELINE]"
            print(f"    #{j+1}: Val MSE {r['val_mse']:.6f} | Orig {r['val_mse_orig']:.0f} | "
                  f"lr={r['config']['learning_rate']:.2e} d={int(r['config']['d_model'])} "
                  f"L={int(r['config']['n_layers'])} | {status}{marker}", flush=True)

        # Advance survivors
        survivors = round_results[:n_keep]
        configs = [r['config'] for r in survivors]

    total_time = time.time() - search_start

    # --- Final Report ---
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE")
    print(f"Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)")
    print(f"{'='*60}")

    best = survivors[0]
    print(f"\nBest config (val MSE {best['val_mse']:.6f}, orig {best['val_mse_orig']:.0f}):")
    for k, v in sorted(best['config'].items()):
        if isinstance(v, float):
            if abs(v) < 0.01 or abs(v) > 100:
                print(f"  {k}: {v:.2e}")
            else:
                print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Compare to baseline
    baseline_results = [r for r in all_results
                       if r['config'] == BASELINE_CONFIG and r['round'] == len(rounds)]
    if baseline_results:
        bl = baseline_results[0]
        delta = (best['val_mse_orig'] - bl['val_mse_orig']) / bl['val_mse_orig'] * 100
        print(f"\nvs Baseline: {delta:+.1f}% (baseline orig: {bl['val_mse_orig']:.0f})")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('experiments', f'hypersearch_{monkey_name}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    # Make configs JSON-serializable
    def serialize_config(cfg):
        return {k: int(v) if isinstance(v, np.integer) else v for k, v in cfg.items()}

    serializable_results = []
    for r in all_results:
        sr = r.copy()
        sr['config'] = serialize_config(sr['config'])
        if sr['val_mse'] == float('inf'):
            sr['val_mse'] = None
            sr['val_mse_orig'] = None
        serializable_results.append(sr)

    search_record = {
        'monkey': monkey_name,
        'n_configs': n_configs,
        'seed': seed,
        'search_space': {k: list(v) for k, v in SEARCH_SPACE.items()},
        'rounds': rounds,
        'total_time_seconds': round(total_time, 1),
        'best_config': serialize_config(best['config']),
        'best_val_mse': best['val_mse'],
        'best_val_mse_orig': best['val_mse_orig'],
        'all_results': serializable_results,
    }

    results_path = os.path.join(results_dir, 'search_results.json')
    with open(results_path, 'w') as f:
        json.dump(search_record, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    return best['config'], best['val_mse']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter search with successive halving')
    parser.add_argument('--monkey', type=str, required=True, choices=['beignet', 'affi'],
                        help='Monkey to search on')
    parser.add_argument('--n-configs', type=int, default=16,
                        help='Number of initial configs (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for config sampling')
    args = parser.parse_args()

    best_config, best_mse = successive_halving(
        args.monkey, n_configs=args.n_configs, seed=args.seed
    )
