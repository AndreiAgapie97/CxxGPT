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

## Masked-token accumulation: 2026-10-13 follow-up

A microbatch returns its mean loss over `n_i` valid targets. Equal weighting
of these means overweights sparsely masked microbatches and includes empty
ones in the denominator. Training now uses `L = sum_i(n_i * L_i) / N`, where
`N = sum_i n_i`, and scales each backward call by `n_i / N`. Each valid token
therefore contributes `1/N`, as in one combined batch. Validation uses the
same token-weighted reduction over sampled batches. Training log intervals
still average optimizer-step means, not all tokens across the interval.

The trainer stages token IDs before forward/backward so it knows `N` without
accumulating large unnormalized gradients. Host staging holds two int arrays:
`2 * sizeof(int) * batch * sequence * accumulation` bytes plus counts.
Activation storage remains at microbatch size. Sampling order and data RNG
consumption are unchanged for nonempty groups. All-empty groups advance the
sampler but skip forward, AdamW moments/decay, router balancing and schedule
updates; an entirely masked training range is rejected. Within a nonempty
group, empty microbatches still participate in router-load accounting, matching
a combined batch. No denominator is invented for an empty evaluation: its
aggregate loss is zero, so callers should not treat that as an accuracy score.

The native fixture has valid counts `[0,1,3,8]`. For dense, ternary, MoE and
ternary MoE, double gradients differ from one combined batch by less than
`5e-17`. Dense global-loss central finite differences give relative error
`1.5e-8`; the negative control using the former equal-micro rule fails the
objective/gradient comparison. Ternary comparisons validate partition
invariance of the chosen STE, not derivatives of discrete rounding.
Three clipped float AdamW updates differ by at most `6e-8` in parameters
and `7e-10` in moments. CLI tests run four updates on unequal and sparse masks
for all four variants: macro/micro checkpoints agree within `2e-8`, data RNG
and router biases match, and token-weighted train/validation CSV fields match
at their printed precision. Sparse cases exercise whole-group skips; fully
masked data and invalid accumulation/logging shapes return checked errors.

`tests/test_gpu.cpp` also exercises this production helper against a CPU
combined batch, including unequal masks and empty groups. Its translation unit
and the GPU-enabled CLI compile, but these CUDA runtime assertions remain
pending NVIDIA hardware. Historical experiment scores below retain their
original batch-mean evaluation semantics and source hashes.

## Trainability and gradient contracts

`bitlinear_forward` and `bitlinear_backward` expose the same quantized
projection used inside the model. Let `Xq` and `Wq` be quantized at the current
master weights and activations. The chosen identity straight-through estimator
uses `dX = dY Wq` and `dW_master = dY^T Xq`, accumulating into existing buffers.
It treats the quantizer Jacobians as identity and detaches scales and rounding
decisions. There is no gradient-based scale optimization in this version.

The test freezes quantized base values, defines local surrogates
`Qx(X) = Xq + (X-Xbase)` and `Qw(W) = Wq + (W-Wbase)`, then checks every input
and weight derivative of the resulting double-precision scalar loss. It also
checks output against independent nested sums and checks gradient accumulation.
This validates the stated surrogate; finite differences of the original
quantizer would answer a different mathematical question.

A native ternary MoE test (four routed experts, top-2, one shared expert)
learns a cyclic next-token task in 31 updates: loss 2.588 to 1.285 with seed 18.
It verifies master-weight updates, refreshed quantized copies, saved moments,
RNG and router biases, and a bit-identical next update after checkpoint reload
under the same optimizer settings and build.

The numerical test helpers now reject non-finite gradients and losses explicitly.
Previously, `std::max(current_error, NaN)` could silently leave an error at zero.

For a policy `pi(a|s)` with reward `r(a)` and action-independent baseline `b`,
descent on `(r-b) * cross_entropy(a)` is descent on `-(r-b) log pi(a|s)`.
Taking the expectation over actions gives the gradient of `-E[r]`: the baseline
term vanishes because `sum_a pi(a) grad log pi(a) = 0`. The policy-gradient test
sums over all actions and compares this expectation against finite differences
of `-sum_a pi(a) r(a)` for four baselines. This caught and corrected a sign error
in the new bandit experiment before its successful results were accepted.

## Integration and validation

Claude's `6707e8b` (CUDA training backend) was fetched during this run and merged
as `653d20d`. Three conflicts were resolved by preserving both changes:

* `.gitignore`: keep both research and CUDA build outputs excluded.
* `Makefile`: clean both output sets, preserve CUDA targets and research
  sanitizer smoke runs.
* CLI: retain the new `Backend` evaluation dispatch and the corrected inclusive
  validation-range check.

CPU validation after integration: GCC 14.2 and Clang 19.1 builds with
`-Wall -Wextra -Werror`; kernel/engine tests; ASan/UBSan; TSan; and the native
experiment smoke under both sanitizers. A CLI smoke trains and validates on
exactly nine tokens for an eight-token sequence. Scalar and AVX2 engine tests
also passed before the configuration extraction (which moves declarations
without changing calculations).

The CUDA backend was compile-checked in Clang's device and host modes for
`sm_89` using CUDA 12.4 toolkit headers and libdevice from NVIDIA wheels.
Clang warns that CUDA 12.4 is only partially supported. The GPU-enabled CLI and
GPU test translation units also compile with GCC warnings treated as errors.
This VM has no NVIDIA GPU; `make gpu-test` and CUDA runtime behavior remain
unvalidated here. The CI nvcc jobs for CUDA 12.8 and 13.0 remain present.

The CUDA kernel was subsequently ported to the stable loss. Thread zero saves
the target gap in shared memory, then every thread synchronizes before logits
are overwritten. Both device and host modes compile with warnings treated as
errors (the known CUDA-version warning is explicitly suppressed). An analytic
whole-model fixture produces logits near `[32000,-32000,0]`: the counted mean
loss is 48000 and a losing output-weight gradient is -0.5. This fixture passes
on CPU; `tests/test_gpu.cpp` now checks the same loss, gradients and an all-ignored
batch on GPU. **The new GPU assertions have not run here. Run `make gpu-test`
on NVIDIA hardware before relying on the CUDA change.** GPU norm reduction
already squares and accumulates in double.

Claude's `ef38470` (streaming tokenization and public-data downloader) was also
merged as `90b4f1a`. The merge was automatic and retained our loss and sampling
fixes. The integrated CPU suite, including Claude's added data/tokenizer tests,
passes under both compilers and sanitizers. The downloader's help command runs;
no external corpus was downloaded during these experiments.

Configuration checks now reject nonpositive model/head dimensions, negative
hidden sizes or expert counts, and zero top-k before division or allocation.
The CLI integration test checks seven invalid configurations return a checked
error, and that single-window training actually includes a validation result.

## Reproducible experiments

Raw CSVs, evaluation transcripts, compiler flags, budgets, CPU information and
source hashes are in [results/2026-10-13](results/2026-10-13). The main toy and
2,000-step reasoning runs use source commit `653d20d`; the 200-step reasoning
run uses `66d66c6`. Later commits add validation and integration changes. Use
the recorded commit and hashes when reproducing a run exactly.

### Transition environment and bounded replay

The environment has eight states, with task A incrementing by one and task B
by three modulo eight. Each context begins with a task marker. All eight
starting states are evaluated exhaustively; this is a fit test, not held-out
generalization. Dense FFN width is 64; MoE has four width-32 experts, top-2,
and no shared expert, matching active FFN width. The router adds 128 active
parameters. Dimension is 32, one layer, four heads, batch two, sequence 16,
learning rate 0.003, AdamW defaults, clip 1.0 and balancing rate 0.001.

Each variant trains on A for 200 updates. Two copies then train for another
200 updates: B alone, or B with a 50% chance of replacing a trajectory with
one from the first 64 observed A trajectories. The buffer holds 8,192 bytes
of sample arrays; B never regenerates old trajectories from the A environment.
Runs use seeds 1, 2 and 3, with variant order rotated across seeds.

| Variant | A after learning | A after B, no replay | A after B, replay | B after replay |
| --- | ---: | ---: | ---: | ---: |
| Dense | 100% | 0% | 98.3–100% | 98.3–100% |
| Dense ternary | 100% | 0% | 99.2–100% | 99.2–100% |
| MoE | 100% | 0% | 99.2–100% | 98.3–100% |
| MoE ternary | 100% | 0% | 99.2–100% | 100% |

Without replay, B reaches 100% in every run, while A is forgotten. With
replay, B sees fewer new-task trajectories at the same update budget. This
comparison measures that retention/plasticity tradeoff; it is not equal
new-task exposure. Replaying the first 64 observations may be unrepresentative
in a larger environment. Two runs of the same binary and seeds reproduced
every recorded non-timing field exactly before integration.

### Reward-based contextual bandit

The policy observes a random state, samples an action, and receives reward one
for the correct successor and zero otherwise. Backward receives the sampled
action and reward advantage, without a correct-action training label. A running
baseline is updated after the reward and eight interactions accumulate per
optimizer step. Each variant starts afresh and runs 500 updates (4,000 actions).

Final greedy accuracy is 100% for all seeds of dense, dense ternary and MoE;
ternary MoE reaches 87.5%, 100%, 100%. Learning with scalar rewards is therefore
possible in this tiny setting, with a seed-dependent shortfall. This is a
single-decision bandit, not multi-step reinforcement learning, exploration in
an open environment, or a continual reward task. The policy-gradient estimator
is checked independently in expectation; that does not prove convergence.

### Reasoning through the existing CLI

`scripts/reasoning-baseline.sh` generates 2,000 training and 500 validation
problems (80,102 / 20,009 byte tokens) with seed 20261013. Tasks are addition
and comparison, train difficulties 1–2, with 60 held-out test questions across
difficulties 1–3. Test questions are excluded from both token sets; the
generator does not deduplicate training against validation. Model dimension
is 64, one layer, four heads, sequence 128, batch two. Dense FFN width 128
matches top-2 width-64 MoE compute, with four experts and no shared expert.
There are three initialization/data seeds at each budget. Every variant at a
given budget uses the same schedule, data samples and five validation batches.

| Variant | Median validation loss, 200 steps | Median validation loss, 2,000 steps | Exact answers at 2,000 steps (seeds 1/2/3, out of 60) |
| --- | ---: | ---: | --- |
| Dense | 0.991 | 0.551 | 1 / 1 / 1 |
| Dense ternary | 1.161 | 0.658 | 0 / 2 / 2 |
| MoE | 1.001 | 0.565 | 1 / 1 / 2 |
| MoE ternary | 1.158 | 0.608 | 0 / 1 / 1 |

Validation losses are means of sampled batch means, not a full-corpus score.
The longer budget improves teacher-forced token loss, but exact answers remain
0–3.3%. Transcripts show incorrect carries, copied digits and inconsistent
final answers even when some intermediate steps are correct. Low token loss
on repetitive answer formatting does not establish arithmetic ability. These
tiny models and runs do not establish a ranking of architectures.

### Memory and computation

Training-array bytes count master weights, gradients, full-sized quantized
copies and Adam moments. They exclude activations, caches, routing buffers,
allocator capacity and replay. Inference bytes count the actual packed weight
arrays and scales plus the unquantized weights; they exclude KV caches and
work buffers. They are not peak process memory.

| Toy variant | Total / active parameters | Training arrays (bytes) | Inference weights (bytes) |
| --- | ---: | ---: | ---: |
| Dense | 10,656 / 10,656 | 213,120 | 42,624 |
| Dense ternary | 10,656 / 10,656 | 213,120 | 4,240 |
| MoE | 16,928 / 10,784 | 338,560 | 67,712 |
| MoE ternary | 16,928 / 10,784 | 338,560 | 6,312 |

Training memory is not reduced by ternary storage in this implementation.
The small ten-token vocabulary makes the toy weight compression look strong:
roughly 10–11x. Auditing the 256-token reasoning checkpoints with the native
[storage audit](results/2026-10-13/storage-audit.cpp) gives 230,144 to 76,560
bytes for dense (3.0x), and 329,472 to 83,752 bytes for MoE (3.9x). A much
larger vocabulary can make unquantized embeddings dominate further.

Step and decode timing medians are retained in the raw toy CSV. The existing
CLI decode benchmark was also run for 128 forced input tokens and eleven
interleaved fp32/packed trials, on three threads, using the trained ternary
checkpoints. Its fp32 baseline holds the *same ternary weights* in float and
does not quantize activations. Argmax agrees on 98% of positions, which measures
activation quantization on that stream, not reasoning accuracy. Observed
speedups were 1.15x for MoE and 2.34x for dense, but concurrent compilation and
VM scheduling contaminate these timings. **Treat them as instrumentation
checks, not controlled performance evidence.** A dedicated quiet benchmark
with varied vocabulary, batch sizes, cache states and repeated A/B ordering
remains necessary.

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
* Masked accumulation is now token-weighted. Historical experiments used
  accumulation one and batch-mean validation; reproduce their recorded source
  rather than silently comparing those scores to the new validation metric.
* Small synthetic tasks are useful for detecting bugs and measuring forgetting;
  they cannot establish real reasoning ability or reinforcement learning.
