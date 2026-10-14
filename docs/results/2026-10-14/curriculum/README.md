# Declared digit-support curriculum and generated traces

Native training/evaluation source `fac69cd`, after integrating Claude main
through `cf183cd`. Run `THREADS=2 scripts/reasoning-curriculum.sh NEW_DIR`
after reproducing the retained optimizer checkpoints. `RESUME_RUN=1` checks
source/binary/data/protocol/initial hashes and preserves completed evidence.
All native executions in this experiment finish before the subsequent CUDA
side-branch integration and its builds. No research compilation or other
research workload overlaps the measured training processes.

Six initial models are the previously frozen dense Muon0.0015 and ternary
MoE Muon0.003 checkpoints, three seeds, dimension128/layers2/heads4, dense
FFN256 or E4/top2/expert128. Both continued-training arms retain the original
attention/RoPE settings. Each tail starts with fresh Muon state, lr0.0005,
warmup20, cosine minimum0.1, wd0.1, clip1, batch2/sequence128 and 1000 successful
updates. Twelve tails consume 3,072,000 sampled tokens in total; initial
2000-update training costs are reused, not included in these timings.
No hyperparameter, checkpoint or seed is selected on this final probe.

The native preparer creates 2400 unique training problems per arm. Control
balances two/three-digit operands; expansion balances two/three/four-digit
operands. Exactly 1600 questions remain at the same problem positions; the
other third is replaced, reducing the frequency of the old lengths too.
The streams have different byte lengths. Paired data RNG states and update/
sampled-token budgets match, but sampled questions and valid-target budgets
do not. Per-seed valid target counts are 151948/152069/151937 control versus
156317/156155/156694 expansion, identical across architectures for each seed.
These are replayed actual target counts, not estimates from corpus mask ratios.

The new 800-question final probe uses seed20261017, 100 per add/cmp difficulty
1/2/3/4 cell. Native and independent binary/TSV parsing exclude all 4223
question strings in the available prior corpora/tests/probes, including all
three 600-question research probes. All new training, validation360 and final
questions are disjoint except the declared common training questions.
Every gold answer and trace is independently recomputed. An operand-width
check confirms control contains only widths2/3, expansion widths2/3/4, and
final difficulties1/2/3/4 correspond exactly to widths2/3/4/5 in this run.
This check matters because the generator can occasionally cross a width
boundary when incrementing an equal comparison operand. Five-digit answers
can appear in training; five-digit question operands do not. Gold completions
fit all 800 contexts. Question-string exclusion does not isolate templates.

All six initial and twelve final checkpoint hashes freeze before any final
scoring. The common validation split is development only and does not select
or stop a run. Original train/validation overlap remains a limitation of the
initial optimizer selections; new train/validation sets are disjoint.

| Model/arm | Two/three-digit exact /400 (seeds1/2/3) | Four-digit exact /200 | Five-digit exact /200 |
| --- | --- | --- | --- |
| Dense initial | 290 / 385 / 392 | 0 / 0 / 0 | 0 / 0 / 0 |
| Dense control | 383 / 398 / 399 | 0 / 0 / 0 | 0 / 0 / 0 |
| Dense expanded | 309 / 387 / 391 | 132 / 186 / 184 | 0 / 0 / 0 |
| Ternary MoE initial | 366 / 334 / 328 | 0 / 0 / 0 | 0 / 0 / 0 |
| Ternary MoE control | 397 / 392 / 392 | 0 / 0 / 0 | 0 / 0 / 0 |
| Ternary MoE expanded | 387 / 367 / 373 | 154 / 134 / 151 | 0 / 0 / 0 |

Exposure to four-digit operands improves that trained support in every seed:
66–93% dense and 67–77% ternary MoE, versus zero in controls. Comparison is
stronger than addition: expanded four-digit add counts are54/88/84 dense and
61/42/66 MoE; cmp counts78/98/100 and93/92/85. All six expanded tails score
lower on old lengths than their controls, an opportunity cost of this fixed
mixture/update budget. Five improve old-length counts over their initial
model; dense seed3 loses one answer. This is not universal forgetting.
Every model still scores zero on five-digit operands. The experiment supports
learning from added supervision, not extrapolation to the next width or an
architecture ranking. Different initial recipes and compute costs preclude
ranking dense against MoE from these scores.

`reasoning_traces` uses production greedy generation and records every raw
completion as hex. It separates exact answers, complete grounded traces,
their intersection, local arithmetic and question-grounded statements.
Addition checks digits/carry/sums, allowing commuted digit operands; comparison
checks the equal prefix and first differing digit/length in question order.
The grammar is the declared native curriculum format. Alternative valid proof
formats or comparison orderings can be rejected, so this is a strict metric,
not a general proof verifier or evidence of an internal algorithm. Missing/
extra statements and false copied digits cannot pass a complete trace.

Expanded exact totals/800 are441/573/575 dense and541/501/524 MoE;
exact answers with a complete grounded trace are434/573/575 and541/494/518.
Some failures are unambiguously false intermediate arithmetic despite a
correct answer: dense initial seed1 writes `6<6` for `max(62,68)` but answers
68; expanded MoE seed2 writes `2>2` for `max(24,21)` but answers24. These
examples reinforce that exact answers alone do not validate generated work.
An independent regex/integer parser reproduces every native per-cell metric
from all 14,400 completions; teacher coverage and exact counts also agree
with the separate native gold-role audit.

Quiet two-thread completed CLI wall-time medians are15.62/16.21 seconds
control/expanded dense and22.47/23.77 MoE. These include initialization,
validation and checkpoint saves. They are limited shared-VM observations,
not isolated kernel speed, matched valid-target counts, GPU throughput or
matched compute across architectures. `cpu.max=200000 100000` permits two
CPU equivalents despite three affinity CPUs.

A completed-run recovery preserves all18 checkpoints,18 gold audits,18 trace
CSVs,18 complete transcripts, the protocol and summary byte-for-byte.
Native binaries are retained separately under the run's `native/` directory
and hashed; subsequent builds replace the workspace executables, so recovery
correctly requires the original source/binaries. Original logs/hashes and
binary token corpora are retained verbatim in `native-output.tar.gz`.
Readable copies normalize trailing whitespace; byte-identical checks refer
to original run files, not the presentation copies. Checkpoints remain in
the execution workspace and are reproducible from the recorded prior recipes.

This final probe is now development data. Future independent final holdouts
must exclude this800, all three earlier600 probes, original tests and every
new/prior corpus. No environmental or multi-step RL ran in this experiment;
NVIDIA runtime remains unavailable. Subsequent integration of Claude's CUDA
side branch is recorded separately in `../integration-892a286`.
