# Optimizer and learning-rate selection

Native C++ model training and evaluation; Bash orchestration and Python
standard-library CSV selection. Start at `6dcb433` with
`THREADS=2 scripts/reasoning-optimizer.sh runs/new-optimizer`. The source/data
paths default to the retained original and capacity runs; the source commit,
compiler, binary hashes, data hashes and protocol are archived here.

The grid is two architectures (dense and ternary MoE), two optimizers (AdamW
and Muon) and two learning rates (0.0015/0.003), with three seeds each: 24
checkpoints. Six AdamW 0.003 controls reuse the verified capacity checkpoints;
18 models train afresh for 2,000 successful updates, batch 2 / sequence 128,
dimension 128 / two layers / four heads, dense FFN 256 or four top-2 experts of width 128,
no shared experts. Warmup 20, cosine minimum 10%, clip 1, weight decay 0.1 and
full-completion supervision are fixed. All 24 end with the same data RNG as
their seed's reference, hence use matched training windows. Optimizer/LR
order and architecture order rotate with seed. No research compilation or
other experiment overlaps these workloads.

The existing capacity probe is development data. Selection maximizes median
exact count over three seeds, then sum exact, then minimizes median sampled
validation loss, then arm ID. Both architecture selections, candidate scores,
checkpoint hashes and probe hashes are frozen in `selection.json` before
any final model scoring. The selected recipes are dense Muon 0.0015 and
ternary MoE Muon 0.003. `development-summary.csv` contains all 24 scores;
Muon beats AdamW in all 12 same-LR/architecture/seed development comparisons.
These comparisons do not establish globally optimal learning rates.

The final probe uses seed 20261016, 100 questions in each add/cmp difficulty
1/2/3 cell. Its 600 unique questions exclude 1,260 original test/development
questions and both training/validation corpora. An independent binary/TSV
parser checks exclusion (`leakage-check.txt`). Original train/validation still
share 51 questions; their sampled validation is a development metric.
Final scoring compares each selected recipe with the original AdamW 0.003
control. Dense's comparison changes both optimizer and learning rate; MoE's
holds the rate fixed. No final scores select recipes or checkpoints.

| Architecture | Recipe | Final exact /600, seeds 1/2/3 | Median final accuracy |
| --- | --- | --- | ---: |
| Dense | Original AdamW 0.003 | 13 / 9 / 21 | 2.2% |
| Dense | Selected Muon 0.0015 | 271 / 389 / 391 | 64.8% |
| Ternary MoE | Original AdamW 0.003 | 8 / 18 / 64 | 3.0% |
| Ternary MoE | Selected Muon 0.003 | 353 / 341 / 328 | 56.8% |

All twelve final models answer **zero of the 200 difficulty 3 questions**.
All 600 gold completions fit the context. The large gain therefore remains
within the trained digit-length range; it does not demonstrate length
extrapolation, different templates, real-world reasoning or correct generated
intermediate steps. Exact answers alone do not validate intermediate reasoning.
Three seeds show substantial variation and do not establish an architecture
ranking. The final probe was independent for this frozen experiment; after
these results it too becomes development data. Future final holdouts must
exclude it, `../capacity/probe.txt`, `../focus/probe.txt` and original tests.

Matched updates do not imply matched compute. Muon adds five Newton–Schulz
iterations for block matrices; embeddings/norms/router remain on AdamW.
Completed CLI wall-time medians at learning rate 0.0015 are 14.0s AdamW versus
31.1s Muon dense, and 19.5s versus 45.3s MoE. These include initialization,
validation and checkpoint writing, use two threads under the two-CPU quota,
and are limited shared-VM observations. Saved AdamW 0.003 controls have no
new wall-time measurement. Do not infer GPU throughput or an isolated kernel
speed comparison from these numbers.

An execution interruption left ten completed arms and one interrupted attempt
whose last logged update was 1400. No checkpoint from that attempt existed;
it restarted from scratch. `74f5afc` adds recovery, verifying original native
binary/source/data hashes and skipping completed arms. Interrupted logs remain
preserved; their unmeasured work is additional cost beyond the 18 completed
runs. Final checkpoint and selection checksums pass. A completed-grid recovery
check preserves all 24 checkpoints, 36 audits, 12 transcripts and the manifest
byte-for-byte. Audit/transcript outputs now publish through temporary files
so interruptions cannot mark partial scores complete. Selection-protocol
fixtures exercise the median versus mean distinction, tie-breaks, non-finite
rejection, immutable manifests and the final-score gate.

`scores.csv` combines development/final role losses, counts, context coverage
and final per-cell scores. `training-time.csv` retains raw completed run times;
`training-time-summary.csv` records medians/ranges. Native/compiler/sanitizer,
selector and recovery checks are in `validation/`. Claude through `b7f2322`
is integrated by ordinary merges, including the laptop's staged fp32-to-ternary
workflow. That workflow was not tested on NVIDIA hardware here. No new CUDA
runtime, environmental interaction or multi-step reinforcement learning ran.

`native-output.tar.gz` retains verbatim native CSV/log/transcript bytes. Readable
text copies remove trailing whitespace; the byte-identical recovery checks
apply to the original run files, not those presentation copies.
