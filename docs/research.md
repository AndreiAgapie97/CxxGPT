# Native C++ research log

Research work lives on `gpt/research`, based on Claude's `main` (initial
baseline `62b8b87`). The GitHub default branch currently points to the older
inference prototype. The user's branch instruction takes precedence over
CLAUDE.md's instruction to commit to `main`. Published research commits are
integrated with ordinary merges; neither branch is force-pushed.

## Numerical correctness: 2026-10-13

The original kernel and training-engine tests passed with GCC 14.2, including
ASan/UBSan. New regression tests exposed these failures:

| Path | Reproduction | Original behavior | Correct behavior |
| --- | --- | --- | --- |
| Cross-entropy | Target 1, logits `[0,-1000,-2000]` | Float loss capped near 87; double capped near 708; numerical derivative zero while backward returns -1 | Loss 1000; derivative -1 |
| Global gradient norm | All gradients have magnitude `1e20` | fp32 squares overflow; clipping destroys direction | Finite norm from double squares, preserving direction |
| Global gradient norm | All gradients have magnitude `1e-30` | fp32 squares underflow to zero; clipping skipped | Nonzero double norm; clipping applied |
| Data sampler | Three-token sequence in six-token range | Final valid start excluded | All three valid starts reachable |
| Data sampler | Range has exactly `sequence_length + 1` tokens | Rejected | Sole valid window sampled |

Cross-entropy now computes `log(sum(exp(z-max(z)))) + (max(z)-z_target)`
before discarding the target logit. Softmax probabilities remain available for
backward. The float exponential is still an approximation, so this does not
claim bit-exact derivatives of that approximation. It removes the large,
artificial plateau introduced by clamping target probabilities.

Gradient clipping retains fixed blocks for reproducibility but performs both
multiplication and accumulation in double precision. This addresses finite
fp32 gradients; it does not repair NaN or infinite gradients upstream.

Sampling uses the inclusive start interval `[lo, hi-sequence_length-1]` and
validates shape and bounds before subtraction. This intentionally changes
the sampled windows for an existing RNG seed. Old checkpoints remain readable,
but resumed trajectories across this fix need not match the old implementation.

Tests also compare four AdamW updates, including bias correction, moments,
matrix-only weight decay, and quantization invalidation, against independent
double-precision arithmetic. Existing dense and MoE gradient checks remain.

## Limits and next work

* Ternary training retains fp32 master weights and AdamW moments. The inference
  bitplanes occupy two stored bits per weight; `log2(3)` is the information
  bound, not the current storage footprint. Whole-model and training memory
  include unquantized embeddings, norms, router weights, and buffers.
* Rounding and hard top-k selection are discontinuous. A straight-through
  estimator is a chosen surrogate gradient, not the exact derivative of the
  discrete forward pass. Validation must state which derivative it checks.
* With selected-logit softmax and top-1 routing, the selected gate is identically
  one, so the task loss gives the router no gradient within a fixed routing
  region. Bias-based balancing can change selection but is not router learning.
* Checkpoints save weights, moments, data RNG and router biases; optimizer
  hyperparameters, learning-rate schedule, data identity and CLI settings must
  also be reproduced externally for an identical continuation.
* Small synthetic tasks are useful for detecting bugs and measuring forgetting;
  they cannot establish real reasoning ability or reinforcement learning.
