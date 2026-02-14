# Experiment Changelog

Running log of model changes, experiments, and results. MSE values are normalized
(per-sample normalization). Original-units MSE shown in parentheses where relevant.

## Current Active Configuration
- **Architecture**: Factorized spatiotemporal encoder + dual near/far PredictionHeads + temporal conv
- **Loss**: Huber (delta=1.0) with timestep weighting (1.0→2.0 ramp), MSE for validation/early stopping
- **Loss weights**: smoothness=0.0, aux=0.0 (disabled)
- **Early stopping**: patience=50, min_epochs=30 (mega mode)
- **Training**: Gradient accumulation for Affi (effective batch 64 from 4x16)
- **Refinement**: n_refinement_iters=1 in mega config, but use_refinement=False in all code paths

---

## Round 1 — Baseline (commit 74e299b)
First mega experiment with MSE loss, smoothness regularization, and default settings.

| Monkey | Val MSE | Test MSE | Test MSE (orig) |
|--------|---------|----------|-----------------|
| Beignet | 0.795 | 0.678 | 84,059 |
| Affi | 1.310 | 1.495 | 53,426 |

## Round 2 — Huber loss, remove smoothness, reduce aux (commit 8ccbf81) [KEPT]
Switched training loss from MSE to Huber (delta=1.0), removed smoothness penalty,
reduced aux weight from 0.1 to 0.01. Validation/early stopping still uses raw MSE.

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R1 |
|--------|---------|----------|-----------------|-------|
| Beignet | **0.727** | 0.748 | 73,397 | -12.7% orig |
| Affi | 1.374 | 1.595 | 55,452 | +3.8% orig (premature early stop) |

**Verdict**: Kept. Clear improvement for Beignet. Affi regressed due to premature
early stopping (best epoch 15), not the loss changes themselves.

## Round 3 — Channel embeddings, per-channel scaling, early stopping fix (commit fc52207) [REVERTED partially]
Added learnable channel embeddings, per-channel output scale/bias, increased
patience to 50, added min_epochs=30 for mega mode.

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R2 |
|--------|---------|----------|-----------------|-------|
| Beignet | 0.752 | 0.694 | 75,466 | +2.8% orig |
| Affi | **1.314** | **1.470** | **51,579** | -7.0% orig |

**Verdict**: Channel embeddings and per-channel scaling reverted (marginal/slightly
negative effect on Beignet). Early stopping fix (patience=50, min_epochs=30) kept —
clearly helped Affi train properly (best epoch 74 vs 15).

## Round 4 — Autoregressive decoding with teacher forcing (commit f1ab1dc) [REVERTED]
Replaced parallel PredictionHead with autoregressive step-by-step decoding. Training
uses teacher forcing (encode full 20-step sequence with causal masking). Inference
decodes one step at a time, feeding predictions back as input.

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R2 |
|--------|---------|----------|-----------------|-------|
| Beignet | 0.798 | 0.732 | 79,656 | +8.5% orig |

**Verdict**: Reverted. Autoregressive error accumulation severely degraded later
timesteps (t10/t1 ratio: 7.8x vs 3.5x for parallel prediction). First-step prediction
was excellent (t1: 16,751 vs 28,585) but gains wiped out by compounding errors.
Training was ~12x slower due to encoding 20-step sequences.

## Round 5 — Noise injection + TF validation (commit 34126e2) [REVERTED]
Added Gaussian noise (scale=0.2) to future targets during teacher forcing to simulate
autoregressive errors. Switched validation to teacher forcing for speed.

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R2 |
|--------|---------|----------|-----------------|-------|
| Beignet | 0.245* | 0.829 | 99,691 | +35.8% orig |

*Val MSE artificially low because validation used teacher forcing (not comparable).

**Verdict**: Reverted. Worst test performance of all rounds. Noise injection failed
to address the fundamental train/inference gap. Teacher forcing validation broke the
early stopping signal. Error accumulation worsened (t10/t1 ratio: 9.3x).

## Round 6 — Revert to R2 architecture, re-add channel embeddings/scaling (commit 9156936)
Reverted autoregressive changes. Re-added channel embeddings and per-channel output
scaling from R3, along with the early stopping fix. Changed dataset seed to 7.

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R2 |
|--------|---------|----------|-----------------|-------|
| Beignet | 0.536 | 0.700 | 74,755 | -1.8% orig |
| Affi | 1.491 | 1.606 | 56,842 | +2.5% orig |

**Verdict**: Kept as new baseline. Clean architecture with channel embeddings
providing slight benefit. This is the starting point for the Phase 1-4 improvement plan.

## Round 7 — Phase 1: Grad accumulation, timestep weights, drop aux (commit d73cc2e) [KEPT]
Three quick wins applied together:
1. **Gradient accumulation** for Affi: effective batch 64 from 4x16 mini-batches
2. **Timestep-weighted loss**: linear ramp 1.0→2.0 across future steps, normalized
3. **Disabled auxiliary loss**: aux_weight=0.0 (was 0.01, empirically not helping)

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R6 |
|--------|---------|----------|-----------------|-------|
| Beignet | 0.536 | 0.837 | 74,437 | -0.4% orig |
| Affi | 1.491 | **1.368** | **50,966** | **-10.3% orig** |

Per-timestep detail (Affi): t4-t8 all improved 7-19%. Best epoch 29 vs 14 (trained 2x longer).
Per-channel variance (Affi): std 51,077 → 25,341 (-50%), worst channel 763K → 178K (-77%).

**Verdict**: Kept. Major Affi improvement from gradient accumulation stabilizing training.
Beignet flat (expected — already had batch_size=64). Timestep weights showed clear effect
on Affi's per-timestep profile but minimal overall impact on Beignet.

## Round 8a — Phase 2: Dual heads + temporal conv + per-channel norm (commit a8b115b) [PARTIALLY REVERTED]
Three changes tested together:
1. **Dual near/far prediction heads**: two PredictionHead instances blended by learned sigmoid gate
2. **Temporal convolution**: depthwise-separable Conv1d (kernel=3) parallel to attention, learned gate blend
3. **Per-channel normalization**: each channel gets own mean/std from T=10 observed steps

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R7 |
|--------|---------|----------|-----------------|-------|
| Beignet | 3.013 | 3.233 | 71,168 | -4.4% orig |

**Verdict**: Per-channel normalization reverted. Normalized MSE appeared 4x worse due to
changed normalization scale, while original MSE actually improved slightly. However, training
was severely destabilized — noisy val curves, gradient spikes. With only T=10 observed steps,
per-channel std estimates are too noisy for stable training. Dual heads and temporal conv
retained for clean re-test without the normalization change.

## Round 8b — Phase 2: Dual heads + temporal conv only (commit b503536) [KEPT]
Same architectural changes as R8a but with global normalization restored. Testing whether
the dual heads and temporal conv help independently of the normalization change.

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R7 |
|--------|---------|----------|-----------------|-------|
| Beignet | 0.514 | 0.880 | 73,089 | -1.8% orig |
| Affi | 1.583 | **1.444** | **49,396** | **-3.1% orig** |

Beignet trained longer (best epoch 59 vs 34), suggesting the model is using the extra
capacity well. Per-timestep profile shows mid-horizon (t4-t6) improvement (-3.7% to -6.0%)
offset by early-step regression (t1: +15%). Per-channel variance increased (std 53K vs 36K).

Affi improved cleanly — per-channel std also decreased (23,773 vs 25,341). Training curves
smooth and stable for both monkeys.

**Verdict**: Kept. Consistent improvement for both monkeys. Dual heads and temporal conv
add genuine value without destabilizing training. Architecture locked in for Phase 3
(hyperparameter search).

**Cumulative progress vs R2 (pre-plan baseline):**
- Beignet: 73,397 → 73,089 (-0.4%)
- Affi: 55,452 → 49,396 (-10.9%)

---

## Key Learnings
1. **Huber loss** is a clear win over MSE for training — better handles outlier channels
2. **Smoothness regularization** was unnecessary with Huber loss
3. **Early stopping** needs sufficient patience (50) and min_epochs (30) for Affi
4. **Channel embeddings / per-channel scaling** had negligible impact, not worth the complexity
5. **Autoregressive decoding** produces excellent single-step predictions but compounding
   errors make it unsuitable without more sophisticated error correction. The 12x training
   slowdown makes experimentation impractical.
6. **Teacher forcing for validation** produces misleading val MSE — don't use it for early stopping
7. **Gradient accumulation** is critical for Affi — batch_size=16 is too noisy, effective batch 64
   stabilizes training and allows the model to train 2x longer before early stop
8. **Per-channel normalization** with T=10 is too noisy — std from 10 timesteps destabilizes
   training. Don't retry without significantly more observed steps or precomputed dataset statistics
9. **Auxiliary loss** (freq band prediction) provides no measurable benefit after 5+ rounds at
   various weights (0.1, 0.01). Safe to disable entirely.
10. **Dual near/far heads + temporal conv** provide modest but consistent gains (-1.8% Beignet,
    -3.1% Affi). The temporal conv's learned gate and dual head blending give the model more
    flexibility without destabilizing training. Models also train longer before early stopping.
