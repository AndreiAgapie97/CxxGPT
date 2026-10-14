# Reasoning diagnosis and reproduction

Frozen audits use the original twelve 2,000-step byte-vocabulary checkpoints.
`frozen-audit/` records whole-question teacher-forced loss and token accuracy
by character role, exact-answer accuracy, and token-weighted versus legacy
batch-weighted sampled validation. The native audit requires vocabulary 256;
it is not a BPE analysis. Gold completions fit all 60 original questions.
Delimiter tokens are `\n#### `; step-other tokens include semantic operators
and words, not just formatting. Character correctness does not prove arithmetic.

`focus/` and `focus-summary.csv` contain a fixed exploratory experiment at
`29e0c9c`: dense and ternary MoE, seeds 1/2/3, warm-starting the earlier
2,000-step checkpoints with a fresh optimizer and reset data RNG. Each arm
trains 200 steps, batch two, sequence 128, AdamW, lr 0.0005 with cosine decay
to 10%, warmup zero, clip 1, wd 0.1, on two threads. One arm retains all
completion labels; the other labels final-answer digits only. Token streams
are identical, and saved data RNG states agree between every arm pair.
Intermediate steps remain gold context during training. Full validation
masks are used for both arms. Arm order alternates with seed.

The fresh 600-question probe has 100 questions in each add/cmp difficulty
1/2/3 cell, seed 20261014. Exact question strings are unique and excluded
from the original training, validation, and 60-question test sets. This
excludes direct question duplication, not semantic similarity. Data and
checkpoint hashes are retained. This probe was not used for training or
checkpoint selection; after this analysis it should be treated as a development
probe for future experiments, with a new independent final holdout.

Final-digit-only training lowers gold-context digit loss in every run but
worsens full-completion validation loss in every run. Exact-answer scores
are lower than the full-label control in four of six comparisons. This
supports retaining mixed supervision; it does not establish a causal account
of all failures or an architecture ranking. Teacher-forcing gains do not
establish improvements in generated reasoning.

`weighted-baselines/` reruns all four variants for 2,000 steps and three seeds
using the original generation, schedule, batch and learning rate, now on two
threads with token-weighted validation. All twelve checkpoint files are
byte-identical to the earlier three-thread runs (`reproduction.csv`), while
validation loss changes because its aggregation changed. No training-speed
claims are made: compilation ran concurrently with this reproduction.
The run source had the benchmark merge pending; hashes and `source.patch.gz`
record its exact code relative to `29e0c9c`.

`corrected-bench/` is a quiet two-thread, 128-token, 21-pair decode rerun
with two warmup pairs and alternating order. The fp32 clone now carries learned
router biases. Packed and float weight counts include those biases; older
weight counts omitted this small state. Both trained checkpoints match the
ternary training-model path's argmax on all positions. Timing ratios are
limited shared-VM observations, not isolated ternary arithmetic speedups.
Prior trained-MoE timings and fp32-agreement figures are confounded by the
missing baseline routing state. Original raw records remain preserved.

`validation/` records compiler, native/CLI, and sanitizer checks. The `runbook-*` logs cover the final integration through `1d660db`. CUDA host
and device compile checks passed; no NVIDIA runtime tests ran.

`memory-follow-up/` adds 216 isolated native CPU memory probes, payload/capacity
checks and process-counter limitations. `capacity/` records a fixed larger-model
experiment and a second fresh 600-question probe, with strong seed variation
and no hardest-level success for its best seed. Both development probes are
retained as text for exclusion from future independent holdouts.

`optimizer/` records a fixed optimizer/LR grid with development selection and
an independent final probe, frozen manifests, strong in-range exact-answer
gains, zero harder-length success, computation cost and interruption recovery.
Its final probe is now development data too; retain and exclude all three
600-question probe files when creating future independent final holdouts.
