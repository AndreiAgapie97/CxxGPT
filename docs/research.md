# Native C++ research log

Research work lives on `gpt/research`, based on Claude's `main` (initial
baseline `62b8b87`). The GitHub default branch and PR target are now `main`
(verified 2026-10-14 after the owner's change). The user's branch instruction takes precedence over
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
at their printed precision. Further CLI comparisons with Muon agree within
`6e-8` in parameters and `2e-9` in moments. Sparse cases exercise whole-group skips; fully
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

### Continued integration and validation

Claude's `fbf1f19` (CPU/GPU Muon) and `9130fbe` (README and GPU probe) were
merged as `5e9ee10`. Header and suite conflicts preserve both optimizers and
all numerical, STE, training and policy-gradient tests. The README retains
Claude's project guide and results, with explicit packed-storage and GPU
validation qualifications. AdamW and Muon both pass the masked CLI checkpoint
comparisons. Muon adds independent double optimizer and orthogonalization
references and checkpoint tests.

`5ce0eff` (overnight setup, lower-peak token mixing, prompt-length evaluation
batching) was merged as `c68c2a7`. The sole conflict retained both whole-model
STE and reasoning-evaluation tests. `09adc67` (compact GPU Muon second moments
and an adaptive batch-size probe) was merged as `4b9b4e0` without conflicts.
The device buffer layout and state round-trip assertions compile; their
runtime and memory savings have not been measured on GPU hardware.

GCC 14.2 and Clang 19.1 warning-as-error builds, all CPU suites, real CLI
regressions, ASan/UBSan, TSan and native experiment sanitizer smoke runs pass
after integration. Sanitizer targets now compile the real CLI and exercise
its masked training, checkpoint and benchmark paths too. Scalar and AVX2
engine checks passed before the data/evaluation update. CUDA host/device
`sm_89` checks pass after the GPU-memory update; GPU test translation units
and the GPU-enabled CLI compile. **None of the CUDA runtime assertions have
run in this environment.** New laptop scripts pass shell syntax checks;
no public corpus download or overnight GPU run was attempted.

The twelve existing 2,000-step reasoning checkpoints were reevaluated after
the prompt-length batching change. Exact-answer counts are unchanged, still
0–2 out of 60. Thus improving batching robustness does not resolve the poor
reasoning accuracy of these tiny models. Transcripts, comparisons, source
and checkpoint hashes are in [follow-up results](results/2026-10-13/follow-up).
Claude reports better larger-model results on `main`; this run has not
reproduced those measurements, and one-seed comparisons do not measure seed
variance.

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

The historical training-array bytes below count master weights, gradients,
full-sized quantized copies and Adam moments. The 2026-10-14 memory follow-up
removes the unused quantized copy for fp32 CPU models. They exclude activations, caches, routing buffers,
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

### Quiet decode and CPU-quota investigation

The benchmark now accepts `--warmup` and `--log`, records raw trial CSVs,
alternates which path runs first, and reports the median of each path.
`scripts/decode-benchmark.sh` preserves source, binary and checkpoint hashes,
flags, budgets, CPU affinity and cgroup counters, and forces a rebuild using
the recorded compiler. Both implementations use the same ternary projection
weights, forced token stream and thread count; the fp32 baseline keeps
activations unquantized. Model construction and packing are outside timing;
KV cache reset and argmax are inside.

A quiet matrix at `e5f302c` covered trained seed-1 dense/MoE checkpoints and
untrained 8,192-vocabulary controls, prefixes 16/64/128, one/three threads,
two warmup pairs and 21 recorded alternating pairs. Investigation found that
three CPUs are available to affinity, but `cpu.max = 200000 100000` limits
CPU time to two CPU equivalents. Three-thread paired timing ratios span
roughly 0.09–16.85, so those trials cannot establish reliable performance.
The matrix was repeated with one/two threads at `891f990`, with no concurrent
build or experiment during measurements. All 1,008 trials and 48 summaries
are retained in [follow-up/decode](results/2026-10-13/follow-up/decode).

The following historical rows use two threads and a 128-token prefix. Times
are medians of 21 trials in milliseconds, with warm code/weights and a reset
KV cache. The trained-MoE row is confounded by omitted fp32 router bias; see
the corrected 2026-10-14 measurements below.

| Scenario | fp32 / packed ms | Observed ratio | fp32 / packed weight bytes | Weight reduction |
| --- | ---: | ---: | ---: | ---: |
| Trained dense, vocab 256 | 1.396 / 0.869 | 1.61x | 230,144 / 76,560 | 3.01x |
| Trained MoE, vocab 256 | 1.662 / 1.057 | 1.57x | 329,472 / 83,752 | 3.93x |
| Untrained dense, vocab 8,192 | 7.244 / 5.083 | 1.43x | 2,261,760 / 2,108,176 | 1.07x |
| Untrained MoE, vocab 8,192 | 10.230 / 6.184 | 1.65x | 2,361,088 / 2,115,368 | 1.12x |

The trained matrices record no quota throttling. The larger-vocabulary
matrices each record one throttled period (about 2.6 ms) across their six
cases. Individual 128-token two-thread ratios still span 0.71–2.13 overall;
these are observed implementation timings on one shared VM, not architecture
performance guarantees. Unquantized operations differ between the training
model's decode path and the inference engine, so this comparison does not
isolate ternary arithmetic. Argmax agrees on 97.7% and 98.4% of trained-case
positions; 100% for untrained controls is not task accuracy.

The vocabulary controls demonstrate how unquantized embeddings dominate
whole-model weight storage. CPU training arrays remain 1,150,720 bytes for
the trained dense checkpoint and 1,647,360 bytes for MoE, excluding
activations, routing/replay buffers and allocator capacity. Neither these
array counts nor the packed weight counts measure peak process memory.

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

## Reasoning diagnosis and continued validation: 2026-10-13/14

The native `reasoning_audit` scores gold whole-question completions by byte
role. A separate double analytic fixture checks role boundaries, token counts,
stable extreme loss, aggregation and agreement with the engine. Dense seed 1
has 98.3% delimiter accuracy (loss 0.034), 65.3% intermediate-digit accuracy
(loss 1.075), and 40.0% final-digit accuracy (loss 2.428), despite receiving
correct intermediate steps. Overall gold token accuracy is 79.1%; exact answers
remain 1/60. Non-digit step tokens include semantic operators and words.
These diagnostics locate errors but do not prove a causal explanation.

A fixed 200-update experiment compares full completion labels with final-digit
labels alone, retaining identical context streams and matched sampler state.
It uses dense and ternary MoE checkpoints, three seeds, fresh AdamW state,
learning rate 0.0005, batch two, sequence 128 and two threads. The new probe
has 600 unique questions, excluding all train, validation and original test
question strings. It was not used for checkpoint selection. After this analysis
it becomes a development probe; future tuning needs an independent holdout.

| Variant | Initial exact answers /600 (seeds 1/2/3) | Full labels | Final digits only | Median gold final-digit loss, full / final |
| --- | --- | --- | --- | ---: |
| Dense | 12 / 14 / 10 | 9 / 10 / 10 | 12 / 8 / 7 | 2.394 / 1.801 |
| Ternary MoE | 10 / 8 / 7 | 11 / 8 / 10 | 4 / 7 / 12 | 2.587 / 1.938 |

Final-only loss decreases in all six runs and full-completion validation loss
worsens in all six. Exact answers lose to the full-label control in four of
six comparisons. The default objective is retained. Lower teacher-forced
loss on selected digits does not establish better generated reasoning; a
mixed or weighted objective, intermediate-step correctness, and model capacity
remain appropriate follow-ups. This is an exploratory small-model experiment,
not an architecture ranking or an environmental reinforcement-learning result.

All four original 2,000-step baselines were reproduced with three seeds and
new token-weighted validation. **All twelve checkpoint files are byte-identical**
to their earlier runs. Dense seed 1's same-window validation changes from
legacy batch mean 0.557236 to token mean 0.553520. The training trajectory is
unchanged for these accumulation-one, nonempty batches. Compilation overlapped
reproduction, so its throughput logs are not performance evidence.

Benchmark review exposed a missing learned router bias in the fp32 clone.
`make_fp32_decode_baseline` now copies effective weights and routing state;
a forced-bias decode test matches an explicit reference and differs from a
zero-bias negative control. Claude's `2555c12` inference-reference check was
merged with alternating order, warmups and CSV logs as `97c6c40`. The CSV adds
`ternary_model_agreement`. Weight counts now include routing bias on both
paths. **Prior trained-MoE timing and fp32-agreement comparisons are confounded
by omitted baseline routing state.** Their raw measurements remain preserved.
A quiet 128-token, two-thread rerun gives observed ratios 1.33x dense and 1.74x
MoE; both packed paths match the quantized training-model reference's argmax
on 100% of positions. These limited VM timings do not isolate ternary arithmetic.

Claude's warm-start and divergence updates through `ee1b279` were merged as
`5dc033a`, preserving masked-token weighting. Failed attempts do not advance
optimizer/schedule steps and discard routing loads without bias changes.
Non-finite loss is checked before stepping; ten consecutive failures stop.
Checkpoint writes validate finite stored state and moment-array sizes before
opening the temporary file; weight-only reads check skipped optimizer extent.
CLI tests verify warm-start reset, resume precedence, truncated state, and
byte-identical preservation of an existing checkpoint after failed writes
and a diverging run. Finite state does not guarantee a usable model: very large
finite weights can still overflow activations.

Latest runbook, downloader checks, output-folder creation and optional chain
planning through `1d660db` were integrated as `b55d899`. No unresolved conflicts remain.
Compiler, native/CLI, sanitizer and CUDA compile results, hashes, probe records,
per-seed outcomes and reproduction evidence are retained in
[results/2026-10-14](results/2026-10-14). GPU runtime validation remains pending.

## CPU memory and capacity follow-up: 2026-10-14

Claude's CLI help, BOS/answer sampling, generation-length correction,
fp32-to-ternary warm starts and conditional GPU weight-copy allocation through
`ab1b092` were merged as `f83a87b`. Compatible conflicts retain warmup/raw
benchmark controls, context defaults and both model test sets. All native,
compiler and sanitizer checks pass after integration; CUDA host/device and
GPU-enabled translation units compile. NVIDIA runtime remains pending.

`50efa7d` adds vector payload/capacity accounting and removes the unused fp32
CPU quantized copy. Independent shape formulas check dense activations,
backward scratch, rotary/routing state, KV caches and packed matrices,
including retained training buffers and inactive MoE placeholders. Four
before/after actual CLI checkpoints (dense/MoE, AdamW/Muon) are byte-identical.
Warm-start tests also verify `--init` with `--ternary` converts the saved config
while preserving master weights and resetting optimizer/RNG.

CPU research scripts now default to two threads (decode compares one/two);
explicit overrides remain available. A quiet native memory sweep runs 72 configurations in three fresh processes
each, two threads under the two-CPU quota. Owned payload/capacity match in
every repeat. It covers two vocabulary sizes, training sequences, decode
contexts/batches, optimizers and float/ternary dense/MoE. The fp32 fix saves
394,496/593,152 bytes for the tested dimension-64, two-layer dense/MoE models:
20% of the former five parameter-sized arrays, less of total training memory.
Dense B2/T128 training owns 5,854,232 bytes float versus 6,248,728 ternary.
CPU Muon still keeps full-sized m/v; worker scratch appears in process RSS.

Packed dense KV cache rises from 131,072 to 524,288 bytes between reserved
contexts 128 and 512, consuming 56% then 80% of its array payload. Decoding
only 16 tokens does not reduce the reserved cache. Embeddings dominate at
vocabulary 8192. Model-based decode retains gradients/backward arrays;
therefore its footprint difference from packed inference does not isolate
ternary storage. Process peaks include unpacked construction and cloning,
even after source arrays are freed. Linux VmRSS and getrusage are approximate
counters and can disagree; these records do not establish exact peak heap
allocation or GPU memory. All raw measurements and definitions are in
[memory results](results/2026-10-14/memory-follow-up).

`ce302b7` fixes a capacity experiment before scoring: original full objective,
2,000 updates, batch two, sequence 128, AdamW lr 0.003, warmup 20, three seeds.
Larger models change dimension 64 / one layer / FFN 128 to dimension 128 / two layers / FFN 256
(expert width 64 to 128). Every paired data RNG matches. A fresh 600-question
probe excludes train/validation and both older probes; an independent parser
confirms exclusion. The existing train/validation sets share 51 question
strings, so sampled validation remains a development metric.

| Variant | Small exact /600, seeds 1/2/3 | Large exact /600 | Small / large median final-digit CE |
| --- | --- | --- | ---: |
| Dense | 16 / 19 / 7 | 10 / 14 / 20 | 2.373 / 2.398 |
| Ternary MoE | 12 / 6 / 8 | 13 / 19 / 64 | 2.616 / 2.271 |

Larger ternary MoE improves counts in all three seeds, with substantial
variation (2.2–10.7%). Dense improves in one seed and declines in two. The
best MoE seed answers zero difficulty-3 questions on either task: its gains
are on the training difficulty range. This changes width/depth together and
uses approximately 6.3x more linear forward MACs per token, not equal compute.
It does not rank architectures or show that capacity alone solves reasoning.
The fixed learning rate need not suit both sizes. The new probe now becomes
development data and must be excluded from future independent final holdouts.
No environmental or multi-step reinforcement learning ran in this follow-up.
[Capacity results](results/2026-10-14/capacity) preserve exact counts, role loss,
hashes, transcripts, both probes and trained-checkpoint memory profiles.

## Optimizer selection and independent scoring: 2026-10-14

Claude's warm-start scripting and learning-rate documentation through `e5c1658`
were merged as `58e6f7c`; the laptop's staged fp32-to-ternary workflow through
`b7f2322` was merged as `ce4ecad`. Both merges are compatible and conflict-free.
The native numerical code is unchanged. GCC/Clang, native/CLI, ASan/UBSan and
TSan checks pass; the optional selection protocol has independent examples
for median/sum/loss/ID rules, immutable manifests and non-finite rejection.
CUDA runtime is still unavailable; the laptop workflow has not run here.

`6dcb433` presets a 2x2 optimizer/learning-rate grid at the larger capacity:
AdamW/Muon, lr 0.0015/0.003, dense/ternary MoE, three seeds, 2,000 updates,
full labels and the same sampler windows. Six saved AdamW 0.003 checkpoints
are reused; eighteen models train afresh. Existing capacity questions are
used only for development selection. Both selections and all model hashes
are frozen before final scoring: dense Muon 0.0015 and ternary MoE Muon 0.003.
The rule uses median exact answers, sum exact, median validation loss, then
arm ID. Muon beats AdamW in all twelve same-LR development comparisons.

The new seed 20261016 final probe has 600 unique questions, excluding corpora
and all 1,260 older test/development questions. Independent parsing confirms
exclusion. Original train/validation still overlap by 51 questions. Final
scores evaluate the selected recipes and original AdamW 0.003 controls:

| Architecture | Original AdamW exact /600, seeds 1/2/3 | Selected Muon exact /600 | Median accuracy, original / selected |
| --- | --- | --- | ---: |
| Dense | 13 / 9 / 21 | 271 / 389 / 391 | 2.2% / 64.8% |
| Ternary MoE | 8 / 18 / 64 | 353 / 341 / 328 | 3.0% / 56.8% |

All twelve models still answer zero of the 200 difficulty 3 questions. Gold
completions all fit the context. The gains are within the trained digit-length
range; they do not establish general reasoning, correct generated intermediate
steps, or extrapolation. Dense's final comparison changes optimizer and LR;
MoE's holds LR fixed. Different selected recipes and three seeds do not rank
architectures. This final probe now becomes development data and must also
be excluded from future independent final holdouts.

Muon adds five matrix orthogonalization iterations. Completed CLI wall-time
medians at lr 0.0015 are 14.0/31.1 seconds AdamW/Muon dense and 19.5/45.3 MoE.
These limited two-thread shared-VM observations include initialization,
validation and saving; equal updates are not equal computation. Saved controls
have no new wall-time measurement. One interrupted attempt has additional,
unmeasured cost; its last logged update was 1400 and it had no saved checkpoint.
Recovery preserves ten completed arms and verifies unchanged native hashes.
All 24 samplers match references, frozen checkpoint hashes pass, and a completed
recovery recheck preserves every model, audit, final transcript and selection.

[Optimizer results](results/2026-10-14/optimizer) preserve the manifest,
all development/final counts, role loss, timings, interrupted logs, exclusions,
probe text and validation. No environmental or multi-step RL experiment ran
in this follow-up. GPU stable loss, masked accumulation, compact Muon state
and conditional weight buffers still require `make gpu-test` on NVIDIA hardware.

## Hybrid attention and positional encoding integration: 2026-10-14

Claude through `cf183cd` is integrated with an ordinary merge, adding CPU
full-precision attention with ternary FFNs, optional layers without RoPE,
and expression-rewrite supervision. Two compatible conflicts preserve both
per-projection precision choices and the research STE helpers, and retain
the router-bias-preserving decode baseline. That baseline also clears the
hybrid attention flag when constructing its full-precision clone.

Eight double-reference and clipped float accumulation variants now cover
hybrid dense/MoE and all/mixed NoPE. Actual masked/sparse CLI comparisons
cover both AdamW and Muon in the new modes. Warm starts retain hybrid attention,
and checkpoint fixtures cover 11-, 12- and 13-field configurations. Existing
stable loss, norm, sampler, checkpoint protection and memory tests remain.
GCC/Clang, native/CLI, ASan/UBSan and TSan checks pass; CUDA host/device
compilation passes, but NVIDIA runtime remains pending. GPU training explicitly
rejects hybrid attention and NoPE; packed inference supports NoPE and explicitly
rejects hybrid attention. The defaults retain the original architecture.

Main was reviewed independently before integration: its new CPU features
passed their checks, but it still contains probability-clamped cross-entropy
and equal-microbatch weighting because the research changes remain unpublished.
The measured extreme loss is 87.33654 rather than 1000; unequal-mask CLI
partition differences are about 0.00446/0.00423 after four AdamW/Muon updates.
Both fixes remain preserved here. The new switches alone do not establish
that ternary attention causes copying mistakes or improve digit-length
generalization. Integration evidence is retained in
`docs/results/2026-10-14/integration-cf183cd`.

## Declared digit-support experiment and generated traces

`reasoning_traces` retains optional production greedy completions and checks
the native add/cmp trace format against each question. Addition checks column
operands (allowing commutation), propagated carry and arithmetic; comparison
checks the equal prefix and first differing digit or differing lengths.
Metrics separate local arithmetic, grounding, complete traces, exact final
answers, and their intersection. These are strict curriculum-format checks,
not a general proof checker or evidence of the model's internal algorithm.
Missing/extra statements and correct final answers with wrong copied digits
do not pass a complete grounded trace. Native fixtures cover those failures,
overflowing text and 800 independently recomputed gold traces.

`reasoning_curriculum` prepares two paired 2,400-question corpora: a balanced
difficulty 1/2 control and balanced difficulty 1/2/3 expansion. Exactly 1,600
questions are shared at the same problem positions; replacing one third also
changes the relative frequency of old lengths. Every archived available
corpus/test/probe, including all three 600-question research probes, is
excluded from new questions. A fresh 800-question final probe spans add/cmp
difficulties 1–4; a separate 360-question validation split is development only.
All new training, validation and final questions are disjoint except the
declared shared training examples. Question exclusion does not isolate templates.

The fixed script starts both arms from the selected 2,000-update optimizer
checkpoints, three seeds each, with fresh Muon state and a declared lr 0.0005,
warmup 20, clipping 1 and 1,000 successful B2/T128 updates per tail. Each tail
consumes 256,000 sampled tokens; actual valid target counts are replayed and
recorded. The streams differ, so matched sampler draws do not imply identical
windows or supervised-token counts. There is no checkpoint selection or early
stopping. All six initial and twelve final hashes freeze before final scoring.
Within the expanded arm difficulty 3 is trained support; difficulty 4 remains
unseen digit length. This is a recipe/curriculum comparison within each
architecture, not a ranking or matched compute between architectures.

GCC/Clang, native/CLI, ASan/UBSan and TSan fixtures pass for the tools. The
independent Python standard-library checker validates split exclusions,
arithmetic answers, budgets and generated native counts; model execution
itself remains native C++. Results will be recorded after the fixed run.

## CUDA side-branch integration and warm-start precision

Claude's active side branch through `892a286` adds fused causal attention,
bf16 linear matmuls, CUDA-event profiling, GPU hybrid attention/NoPE, and
hybrid ternary nights. The ordinary merge preserves stable GPU loss and
token-weighted accumulation, all research native tests, and both GPU test
sets. On this branch fused attention remains explicit `--fused-attn` until
NVIDIA runtime validation; cuBLAS attention is the default. The earlier
`--no-tf32` flag and boolean GPU constructor remain compatible. No CUDA
speed, GPU memory reduction or real-device correctness is measured here.

The attention CPU emulator checks the actual fused kernel code against
double forward/log-sum-exp and Q/K/V gradients across five shapes, including
partial tiles and head size 128. Its error helper now rejects NaN/infinity
instead of silently treating them as zero error. GPU masked accumulation
fixtures cover eight architectures/configurations with fused and unfused
attention; runtime assertions remain pending hardware.

Integration also exposes a warm-start issue: a saved hybrid flag could not
be cleared, so `TERNARY=all` after a hybrid night still retained fp attention.
Explicit `--fp-attn 0` now overrides it on `--init`, and the all-ternary night
uses that flag. Explicit NoPE overrides are supported on warm starts too;
absent flags retain the saved configuration and resume remains authoritative.
CLI fixtures verify unchanged master weights and fresh optimizer/sampler
state for both conversions, and reject contradictory precision/fused flags.

## Publishing and continuation status

The local research branch preserves all earlier work and ordinary Claude
merges through main `cf183cd` and active Claude branch `892a286`. There are no unresolved merge conflicts. GitHub
reads and fetches work, but Git HTTPS still denies push with HTTP 403 and
the GitHub app's tree-write check returns `Resource not accessible by
integration` (403). No remote `gpt/research` branch or research PR exists.
The PR description is prepared in `docs/research-pr.md`, targeting `main`
and remaining draft until NVIDIA runtime validation is recorded.

The local bundle and binary patch in `/workspace` are refreshed after the
final documentation commit. The existing hourly follow-up remains enabled;
its context tracks completed token weighting, CPU-quota-aware benchmarks,
pending GPU loss/accumulation/Muon validation, seed variation and hard-level
reasoning, mixed supervision, resource measurements, and access checks. It must inspect the preserved artifacts before recreating missing
remote work and must never force-push or alter Claude's branches.
