# Experiment Changelog

Running log of model changes, experiments, and results. MSE values are normalized
(per-sample normalization). Original-units MSE shown in parentheses where relevant.

## Current Active Configuration
- **Architecture**: Factorized spatiotemporal encoder + dual near/far PredictionHeads + temporal conv
- **Model size**: d_model=128, n_heads=8, n_layers=4 (~1.36M params)
- **Loss**: Huber (delta=1.0) with timestep weighting (1.0→1.44 ramp), MSE for validation/early stopping
- **Loss weights**: smoothness=0.0, aux=0.0 (disabled)
- **Optimizer**: AdamW lr=3.46e-4, weight_decay=0.02, dropout=0.186
- **SWA**: snapshot every 5 epochs from min_epochs, averaged at end of training
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
12. **Tuned non-size hyperparameters** (lr, wd, dropout, tw) transferred cleanly from the search
    to d=128/L=4 — gave -3.0% Beignet, -1.7% Affi on test without any architecture change.
11. **Hyperparameter search / successive halving** favors fast-converging (often larger) configs
    in early rounds. d=128/L=4 was eliminated at 30 epochs despite being competitive long-term.
    The actual test gain from d=192/L=6 was only 0.26% but 11x slower — not worth it.
    The transferable findings are the non-size hyperparameters: lr≈3.46e-4, dropout≈0.186,
    timestep_weight_max≈1.44. weight_decay should be scaled with model size.

## Round 9 — Phase 3: Hyperparameter search + full eval (hyperparam_search.py)
16 configs × 4 rounds (30/60/120/200 epochs) successive halving on Beignet.
Searched: lr, weight_decay, dropout, d_model, n_layers, huber_delta, timestep_weight_max.
Total search time: ~14.3 hours.

Search winner: d=192, L=6, lr=3.46e-4, wd=0.047, dropout=0.186, delta=1.0, tw=1.44

Full eval with winning config (Beignet only):

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R8b |
|--------|---------|----------|-----------------|--------|
| Beignet | 0.506 | 0.834 | 72,900 | -0.26% |

**Verdict**: d=192/L=6 gives only +0.26% improvement on test but takes 11x longer to train
(8,323s vs 750s, 4.28M vs 1.36M params). Not worth it — especially since Affi would take
~35 hours at this size. Reverted model size to d=128/L=4. Non-size hyperparameters from
the search (lr=3.46e-4, dropout=0.186, tw=1.44) applied to d=128/L=4. weight_decay scaled
down to 0.02 (0.047 was tuned for the larger model). Running R10 to evaluate.

## Round 10 — Phase 3 eval: d=128/L=4 with tuned hyperparameters [KEPT]
Same architecture as R8b, with hyperparameters from the search applied:
lr 3e-4→3.46e-4, weight_decay 0.01→0.02, dropout 0.2→0.186, timestep_weight_max 2.0→1.44.

| Monkey | Val MSE | Test MSE | Test MSE (orig) | vs R8b |
|--------|---------|----------|-----------------|--------|
| Beignet | 0.511 | 0.829 | 70,931 | **-3.0%** |
| Affi | 1.528 | 1.379 | 48,540 | **-1.7%** |

Both monkeys improved cleanly. Training faster than R8b (Beignet 661s vs 750s). Per-channel
std also decreased for both (Beignet 48,885 vs 53,013; Affi 22,008 vs 23,773).

**Cumulative progress vs R2 baseline**: Beignet -3.4%, Affi -12.5%.

**Verdict**: Kept as new baseline. Adding SWA (Phase 4) for final submission.

Key learning: successive halving favors large models that converge fast in early rounds.

## Round 11 — Phase 4: SWA (FINAL SUBMISSION RUN) [KEPT]
SWA added to training loop: snapshot every 5 epochs from min_epochs=30, averaged weights
evaluated on val set after training, kept if better than best single checkpoint.
All other config identical to R10.

| Monkey | Val MSE | Test MSE (orig) | vs R10 | SWA used? |
|--------|---------|-----------------|--------|-----------|
| Beignet | 0.531 | 64,390 | **-9.1%** | Yes |
| Affi | 1.517 | 48,476 | **-0.1%** | — |

SWA gave a huge improvement for Beignet despite slightly worse val MSE (0.531 vs 0.511).
This is expected SWA behavior: flatter basin generalizes better to test even at slightly higher
val loss. Affi's SWA result was comparable to single best checkpoint (marginal difference).

**Cumulative progress vs R2 baseline**: Beignet **-12.3%**, Affi **-12.6%**.
Training time: Beignet 661s, Affi 2,797s.

### Hidden Test Results (COMPETITION EVALUATION)
| Monkey | Hidden Test MSE | Local Test MSE | Ratio |
|--------|----------------|----------------|-------|
| Beignet | 380,000 | 64,390 | 5.9x |
| Affi | 490,000 | 48,476 | 10.1x |

**Root cause analysis**: The 5.9-10.1x inflation is entirely explained by **session amplitude
mismatch**, not a model bug. Original-unit MSE scales as (session_std)². The hidden test
sessions have systematically higher signal amplitude than our training sessions:
- Beignet: hidden test needs avg std ≈ 1,207 vs training avg 497 → ratio² = 5.9x ✓
- Affi: hidden test needs avg std ≈ 918 vs training avg 289 → ratio² = 10.1x ✓

Per-sample normalization removes the **global** scale for the model, but not **per-channel
relative amplitude differences**. The model's `channel_scale`, `channel_bias`, and
`channel_embedding` learned per-channel amplitude ratios from training sessions. In hidden
test sessions where those ratios differ, those parameters introduce systematic error.
The competition overview (docs/neural_problem_overview.pdf) explicitly identifies
cross-session generalization as the core challenge.

---

## Key Learning 13 — Cross-session amplitude drift
- MSE evaluation in original units is NOT session-invariant: MSE ∝ (session_std)²
- Per-sample global normalization removes global scale for the MODEL but the EVALUATION
  is in original units — higher-amplitude sessions inherently give higher original-unit MSE
  even with perfect normalized predictions
- Per-channel relative amplitudes survive global normalization and can shift across sessions
- `channel_scale`, `channel_bias`, `channel_embedding` can encode session-specific
  per-channel amplitude patterns that don't transfer to unseen sessions
- **Fix**: Per-channel amplitude augmentation BEFORE normalization (R12)

---

## Round 12 — Cross-session generalization: per-channel amplitude augmentation
**Problem**: Hidden test sessions have ~2-3x higher signal amplitude. Per-channel amplitude
ratios between electrodes can also differ from training sessions. The model's channel-specific
parameters (`channel_scale`, `channel_bias`, `channel_embedding`) may encode session-specific
per-channel ratios that don't transfer.

**Fix**: Added `per_channel_scale` augmentation to `NeuralDataAugmentation`, applied
**before per-sample normalization** in `__getitem__`. Each channel is independently scaled
by a log-normal random factor (std=0.7, applied with prob=0.8):
- 68% of channels in [0.5, 2.0]× their original amplitude
- 95% of channels in [0.25, 4.0]× their original amplitude

After global normalization, the model sees training samples with randomized per-channel
amplitude ratios — forcing it to learn temporal patterns that are robust to which channels
are loud or quiet relative to the session mean.

All other config identical to R11 (same architecture, same optimizer, same SWA).

*(Results pending)*
