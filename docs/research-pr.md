# Validate native ternary training and add reproducible research experiments

Confident wrong predictions previously capped cross-entropy near 87 in float
while backward still returned a nonzero gradient. Gradient clipping also
squared in fp32, and sampling excluded the final valid window. This change
uses stable log-sum-exp loss, double norm arithmetic, and inclusive window
sampling with regression tests that reproduce the original failures.

BitLinear now exposes its identity straight-through gradient contract, tested
against finite differences of the stated surrogate. Tests cover AdamW against
double arithmetic, actual ternary MoE learning, identical checkpoint
continuation, non-finite numerical checks, policy-gradient expectation, invalid
configurations, and the real single-window CLI path. Unequal masked microbatches
now optimize one global token mean; validation uses the same weighting. Empty
groups skip optimizer updates. Combined-batch, finite-difference, clipped
optimizer and CLI checkpoint comparisons cover all four architecture variants.
Warm-start, divergence and checkpoint-protection regressions preserve saved
state on failures. Native diagnostics separate intermediate digits, delimiters
and final digits.

Native C++ experiments compare dense, ternary, MoE and ternary MoE with recorded
seeds and matched active FFN width. They demonstrate bounded replay retention
and reward-based bandit learning. The original tiny reasoning runs at 200 and
2,000 steps improve
token loss but achieve only 0–3.3% exact answers. Raw results and limitations are
recorded in `docs/research.md` and `docs/results/2026-10-13`. All twelve
2,000-step checkpoints reproduce byte-identically with token-weighted validation.
A new disjoint 600-question experiment finds that final-digit-only training
lowers teacher-forced digit loss but gives no consistent exact-answer gain.
The full default objective is retained; new records are in
`docs/results/2026-10-14`. A fixed larger-capacity follow-up
improves ternary MoE to 13/19/64 exact answers out of a fresh 600-question
probe across three seeds, from 12/6/8 for the small controls. Dense changes
from 16/19/7 to 10/14/20. The best MoE seed still answers no difficulty-3
questions; compute grows about 6.3x, so this is not an architecture ranking.
CPU memory accounting separates vector payload/capacity from process RSS.
Fp32 models drop an unused quantized copy; four pre/post CLI checkpoints are
byte-identical. A 216-process sweep demonstrates context-cache growth,
embedding dominance and transient construction costs. GPU runtime remains pending.

Claude's CUDA backend, streaming pipeline, laptop/overnight setup, Muon
optimizer, evaluation/loading updates, and compact GPU optimizer state are
merged into this branch. Compatible conflicts preserve both changes; no
Claude branch is overwritten and no force-push is used.

Validation:

- GCC 14.2 and Clang 19.1 builds with `-Wall -Wextra -Werror`; kernel and engine
  suites, AdamW/Muon CLI regressions, ASan/UBSan and TSan including the real
  CLI and experiment smoke runs.
- Scalar and AVX2 engine checks; reproducible native experiments and checkpoint
  storage audit. Decode benchmarks record warmups, alternating raw trials,
  CPU quotas and weight storage. Corrected trained cases show 1.33x dense and 1.74x MoE observed
  end-to-end speedup; embedding-heavy controls reduce whole-weight savings
  to 1.07–1.12x. The fp32 clone now preserves learned router bias, with a
  regression test and a separate quantized-model reference check. Earlier
  trained-MoE timings are marked confounded. These are limited VM measurements.
- CUDA host/device compile checks for `sm_89` using Clang and CUDA 12.4 headers.
  The stable-loss, masked-accumulation and optimizer-state paths have CPU/GPU
  regression fixtures; **GPU runtime assertions remain pending `make gpu-test` on NVIDIA hardware. Keep this PR
  draft until that validation is recorded.**

The branch targets `main`, where Claude's current development lives. The GitHub
default branch still points to the older inference prototype.
