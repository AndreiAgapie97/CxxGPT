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
configurations, and the real single-window CLI path.

Native C++ experiments compare dense, ternary, MoE and ternary MoE with recorded
seeds and matched active FFN width. They demonstrate bounded replay retention
and reward-based bandit learning. Reasoning runs at 200 and 2,000 steps improve
token loss but achieve only 0–3.3% exact answers. Raw results and limitations are
recorded in `docs/research.md` and `docs/results/2026-10-13`.

Claude's CUDA backend, streaming data pipeline, and laptop setup commits are
merged into this branch. Compatible conflicts preserve both changes; no
Claude branch is overwritten and no force-push is used.

Validation:

- GCC 14.2 and Clang 19.1 builds with `-Wall -Wextra -Werror`; kernel and engine
  suites, CLI regressions, ASan/UBSan and TSan including experiment smoke runs.
- Scalar and AVX2 engine checks; reproducible native experiments and checkpoint
  storage audit. Timing results are explicitly marked as noisy.
- CUDA host/device compile checks for `sm_89` using Clang and CUDA 12.4 headers.
  The stable-loss port has an analytic CPU/GPU regression fixture; **GPU runtime
  assertions remain pending `make gpu-test` on NVIDIA hardware. Keep this PR
  draft until that validation is recorded.**

The branch targets `main`, where Claude's current development lives. The GitHub
default branch still points to the older inference prototype.
