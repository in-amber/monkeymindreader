# Plan: Comprehensive Experiment Logging

## Context
Training runs on a remote PC (Windows, NVIDIA GPU) but analysis happens here on the laptop. Current results directories (`results_mini/`, `results/`, `checkpoints/`, `runs/`) are all gitignored, so nothing can be pushed. We need a git-tracked experiment logging system that captures enough information to fully understand each training run without re-running it.

## Approach
Create an `experiments/` directory (NOT gitignored) where each training run saves a comprehensive, self-contained record. Modify `train_and_evaluate.py` to generate these logs automatically.

## Files to modify
- **train_and_evaluate.py** — add experiment logging throughout
- **.gitignore** — ensure `experiments/` is NOT listed (it isn't currently, so no change needed)

## Output structure per experiment
```
experiments/
  {monkey}_{mode}_{YYYYMMDD_HHMMSS}/
    experiment.json         # Complete structured experiment record
    training_log.csv        # Epoch-by-epoch metrics (pandas-friendly)
    training_curves.png     # Loss + MSE over epochs
    predictions.png         # Sample prediction plots (6 examples)
    channel_analysis.png    # Per-channel MSE bar/box plot
    timestep_analysis.png   # Per-timestep MSE (which future steps are hardest)
```

## What `experiment.json` contains

### Environment
- hostname, platform, Python version, PyTorch version, CUDA version, GPU name/VRAM
- git commit hash (which code version was used)
- timestamp (start + end), total training time

### Configuration
- Full training mode config (d_model, n_heads, n_layers, lr, epochs, batch_size, patience, dropout, refinement_iters)
- Loss config (smoothness_weight, aux_weight)

### Dataset info
- n_train, n_val, n_test samples, n_channels, n_features

### Model info
- n_params, architecture summary string

### Training history (also in CSV)
- Per-epoch: train_loss, val_loss, val_mse, learning_rate, epoch_time_seconds
- Per-epoch loss components: main_loss, smoothness_loss, aux_loss (broken out from combined loss — requires modifying ForecastingLoss to return components)
- Gradient norm per epoch

### Evaluation results
- test_mse (overall)
- best_val_mse, best_epoch
- per_channel_mse: list of MSE for each channel
- per_timestep_mse: list of MSE for each of the 10 future timesteps
- prediction_stats: mean, std, min, max for predictions and targets
- early_stopped: bool, stopped_epoch

## Changes to `train_and_evaluate.py`

1. **Add `save_experiment_log()` function** — takes all collected data and writes `experiment.json` + `training_log.csv`

2. **Modify `ForecastingLoss.forward()`** — add `return_components=False` parameter. When True, return a dict `{total, main, smoothness, aux}` alongside the scalar loss for logging

3. **In training loop** — collect per-epoch:
   - Loss components (main, smoothness, aux) separately
   - Gradient norm (after clip_grad_norm_ — it returns the norm)
   - Epoch wall-clock time

4. **In test evaluation** — compute:
   - Per-channel MSE: `mse_per_channel[c]` for each channel
   - Per-timestep MSE: `mse_per_step[t]` for each of the 10 future steps
   - Prediction statistics (mean, std, min, max)

5. **Add `plot_experiment()` function** — generates the 4 PNG files per experiment:
   - `training_curves.png`: 2x2 grid (train loss, val loss, val MSE, LR)
   - `predictions.png`: 2x3 grid of sample predictions with per-sample MSE
   - `channel_analysis.png`: bar chart of per-channel MSE (sorted)
   - `timestep_analysis.png`: bar chart of MSE by future timestep

6. **Add `get_environment_info()` function** — collects platform, GPU, versions, git hash

7. **Wire it all together** at end of `train_and_evaluate()` — call save + plot before returning

## Verification
- Run a 2-epoch mini training for beignet to verify the logging works
- Check that `experiments/` directory is created with all expected files
- Verify `experiment.json` has all fields populated
- Verify `training_log.csv` is valid CSV with correct columns
- Verify all 4 PNGs are generated
