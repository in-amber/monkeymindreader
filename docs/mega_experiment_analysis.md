# Mega Training Experiment Analysis

**Date:** 2026-02-10
**Experiments:** `beignet_mega_20260210_110826`, `affi_mega_20260210_113526`
**Hardware:** NVIDIA GTX 1660 SUPER (6.4GB VRAM), PyTorch 2.10.0+cu126

## Results Summary

| | **Beignet** (89 ch) | **Affi** (239 ch) |
|---|---|---|
| Test MSE (original) | **84,059** | **53,426** |
| Test MSE (normalized) | 0.678 | 1.495 |
| Best epoch | 30 / 200 | 47 / 200 |
| Early stopped | epoch 61 | epoch 78 |
| Training time | 4.5 min | 27 min |
| Params | 1.19M | 1.19M |
| Train samples | 686 | 917 |
| Val samples | 85 | 114 |
| Test samples | 87 | 116 |

Note: Affi has higher normalized MSE but *lower* original MSE — its signals have lower per-sample variance, so a given normalized error translates to less absolute error.

## Key Findings

### 1. Predictions are too smooth — the model clips extremes

Both monkeys show the model's prediction range is much narrower than the target range:

| | Pred range | Target range | Coverage |
|---|---|---|---|
| Beignet | [-5023, 7977] = 13,000 | [-7881, 12039] = 19,920 | **65%** |
| Affi | [-4220, 3549] = 7,770 | [-9719, 4059] = 13,778 | **56%** |

The prediction plots confirm this visually — prediction lines (red) are much flatter/smoother than target lines (blue). This is classic regression-to-the-mean, exacerbated by:
- MSE loss (penalizes large errors quadratically, encouraging conservative predictions)
- Smoothness regularization (`smoothness_weight=0.01`) actively penalizing sharp changes
- Small dataset encouraging the model to learn the mean rather than the dynamics

### 2. Far-future timesteps are dramatically harder

MSE roughly doubles or triples from step 0 to step 9:

| Timestep | Beignet MSE | Affi MSE |
|---|---|---|
| 0 (next) | 42,489 | 17,965 |
| 1 | 53,481 | 30,762 |
| 2 | 64,119 | 36,286 |
| 3 | 71,790 | 47,379 |
| 4 | 79,796 | 62,383 |
| 5 | 84,274 | 70,550 |
| 6 | 83,928 | 69,123 |
| 7 | 93,757 | 63,685 |
| 8 | 105,134 | 65,851 |
| 9 (furthest) | 111,807 | 70,145 |

- **Beignet**: nearly monotonic 2.6x degradation from step 0 to step 9
- **Affi**: 3.9x increase from step 0 to step 5, then plateaus — suggesting it outputs a flat prediction for the far future

### 3. Outlier channels dominate overall MSE

The channel MSE distributions are extremely heavy-tailed:

- **Beignet**: channel 23 has MSE **713,322** (9x the mean of 79,058). Std across channels (77,937) nearly equals the mean. Worst 5: channels 23, 86, 19, 72, 51.
- **Affi**: channel 158 has MSE **700,959** (13x the mean of 53,413). Worst 5 channels are spatially clustered: 158, 155, 176, 154, 177.

A handful of extreme channels contribute disproportionately to the overall score. The bottom ~80% of channels are relatively well-predicted; the top ~5% are terrible.

### 4. Validation is noisy due to tiny datasets

With only 85 (beignet) / 114 (affi) validation samples, the validation MSE curves oscillate significantly even as training loss descends smoothly. The "best epoch" may be somewhat lucky rather than a true optimum. This makes early stopping unreliable — the model might benefit from longer training with a different stopping criterion.

### 5. Refinement is untrained

The model has an iterative refinement mechanism (`n_refinement_iters=1`), but training uses `use_refinement=False`. The refinement pathway has never been trained. This is a missed opportunity — or a potential source of harm if turned on at inference without training.

### 6. Training dynamics

- **Gradient norms** are healthy: ~2-4 for Affi, ~1.5-3.5 for Beignet (well below the clip threshold of 1.0... wait, they exceed 1.0, meaning clipping is active). Gradients are being clipped every epoch.
- **Learning rate** barely decayed — cosine schedule was designed for 200 epochs but training stopped at 61/78, so LR only dropped from 0.0003 to ~0.00026 (87% of initial). The model never entered the low-LR fine-tuning phase.
- **Auxiliary loss** stays ~1.1 throughout training and barely decreases — suggesting the frequency band prediction task is much harder and possibly not helping the main task.

## Improvement Ideas

### High Impact

#### 1. Reduce or remove smoothness regularization [IMPLEMENTED]
**Why:** The prediction plots make this clear — the model is too smooth. The real neural signal is jagged; penalizing temporal jitter actively prevents the model from fitting the truth.
**Change:** `smoothness_weight: 0.01 → 0.001 or 0.0`
**Effort:** One-line config change.

#### 2. Huber loss instead of MSE [IMPLEMENTED]
**Why:** The heavy-tailed channel MSE means a few hard channels dominate the gradient. MSE squares the error, giving those outlier channels even more gradient weight. Huber loss transitions to linear for large errors, reducing their dominance and letting the model improve more uniformly.
**Change:** Replace `nn.MSELoss()` with `nn.HuberLoss(delta=1.0)` in `ForecastingLoss`.
**Effort:** Small code change in loss function.

#### 3. Per-channel output scaling
**Why:** The model uses the same prediction head for all channels, but channels have very different magnitudes and dynamics. A learnable scale + bias per channel would let easy channels be handled trivially while the model focuses capacity on harder patterns.
**Change:** Add `nn.Parameter` for per-channel scale and bias in the model's output layer.
**Effort:** Small model change.

#### 4. Autoregressive decoding for far-horizon steps
**Why:** Currently all 10 future steps are predicted in parallel from the same encoding. An autoregressive approach (predict step 0, feed back, predict step 1, ...) would let each step condition on the previous prediction, potentially reducing the far-horizon degradation.
**Change:** Modify `forward()` to decode one step at a time, feeding predictions back.
**Effort:** Medium — requires reworking the prediction head and forward pass.

### Medium Impact

#### 5. Learnable spatial (channel) embeddings
**Why:** Spatial attention is currently permutation-equivariant — the model has no notion of channel identity. Adding a learnable embedding per channel would let it learn which channels co-vary. Especially relevant for Affi where the worst channels are spatially clustered (154-177).
**Change:** Add `nn.Embedding(n_channels, d_model)` added to the encoded representation.
**Effort:** Small model change.

#### 6. Train with refinement (curriculum)
**Why:** The refinement mechanism exists but is untrained. Training normally for N epochs, then enabling refinement for finetuning, would let the model learn from its own predictions.
**Change:** Enable `use_refinement=True` for later epochs.
**Effort:** Small training loop change.

#### 7. Gradient accumulation for Affi
**Why:** Affi uses `batch_size=16` (vs Beignet's 64) due to 239-channel VRAM constraint. This leads to noisier gradients and contributes to the noisy validation curve.
**Change:** Accumulate gradients over 4 mini-batches for effective batch size 64.
**Effort:** Small training loop change.

#### 8. Stronger augmentation
**Why:** With only ~700-900 training samples for a 1.2M param model, overfitting is likely. Current augmentation (electrode dropout + Gaussian noise, each 50% probability) is light.
**Ideas:** Time warping, channel permutation, mixup between samples, magnitude scaling.
**Effort:** Medium — new augmentation classes.

### Experimental / Lower Priority

#### 9. Timestep-weighted loss
**Why:** Far-future steps have 2-4x higher MSE. Giving them higher loss weight would force the model to allocate more capacity there.
**Change:** Weight the MSE per-timestep in `ForecastingLoss`.
**Effort:** Small loss change.

#### 10. Separate near/far prediction heads
**Why:** Near-future (steps 0-4) and far-future (steps 5-9) have qualitatively different difficulty. Separate heads could specialize.
**Effort:** Medium model change.

#### 11. Re-evaluate auxiliary loss [IMPLEMENTED]
**Why:** The auxiliary loss (frequency band prediction) stays flat at ~1.1 throughout training and barely decreases. It may be eating model capacity without helping the main task.
**Change:** Try `aux_weight: 0.1 → 0.01 or 0.0` to see if main MSE improves.
**Effort:** One-line config change.

## Recommended Next Experiment

**Combine improvements #1, #2, and #11** in a single run — they're all config/loss changes with no model architecture modifications:

- `smoothness_weight: 0.01 → 0.0` (remove smoothness penalty)
- Main loss: MSE → Huber (delta=1.0)
- `aux_weight: 0.1 → 0.01` (reduce auxiliary loss influence)

This would test whether the model's core issue is the loss function holding it back from fitting the actual signal dynamics.
