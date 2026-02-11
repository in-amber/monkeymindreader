# Experiment Changelog

Running log of model changes, experiments, and results. MSE values are normalized
(per-sample normalization). Original-units MSE shown in parentheses where relevant.

## Current Active Configuration
- **Architecture**: Factorized spatiotemporal encoder + parallel attention-based PredictionHead
- **Loss**: Huber (delta=1.0) for training, MSE for validation/early stopping
- **Loss weights**: smoothness=0.0, aux=0.01
- **Early stopping**: patience=50, min_epochs=30 (mega mode)
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
