# Fixed capacity bundle experiment

Source `ce302b7`, GCC 14.2, two threads, quota two CPU equivalents. Reproduce
with `THREADS=2 scripts/reasoning-capacity.sh runs/new-capacity`. The script
expects the recorded original checkpoints/data and the earlier development
probe; paths can be set with `SOURCE_DIR` and `PREVIOUS_PROBE`.

Dense and ternary MoE each run three seeds. Small controls reuse the verified
2,000-step checkpoints (dimension 64 / one layer / FFN 128/expert 64). Larger models
start afresh with dimension 128 / two layers / FFN 256/expert 128. Both use four heads,
batch two / sequence 128, AdamW, 2,000 updates, lr 0.003, warmup 20, cosine minimum 10%,
weight decay 0.1, clip 1, top-2 of four experts, no shared experts. Full completion
supervision is retained. Saved data RNG states match every small/large pair:
training windows and update budgets match. Initializations are not identical
weights across shapes. Learning rate and model size were fixed before scoring.
Arm order rotates by seed. No research build overlaps training.

Probe seed 20261015 gives 100 add/cmp questions at each difficulty 1/2/3, 600
unique exact strings. Native preparation and an independent binary/TSV parser
confirm no intersection with train/val or either earlier probe/test. The
original corpora have 1,921/493 unique questions and share 51 question strings;
their sampled validation is a development metric, not an independent final
test. The fresh probe was not used to select checkpoints; after this analysis
it too becomes development data. Retain `probe.txt`, `excluded.txt` and the
earlier `../focus/probe.txt` when preparing future holdouts. Direct question
exclusion does not exclude shared templates or semantic similarity.

| Variant | Small exact /600, seeds 1/2/3 | Large exact /600 | Small median final-digit CE | Large median final-digit CE |
| --- | --- | --- | ---: | ---: |
| Dense | 16 / 19 / 7 | 10 / 14 / 20 | 2.373 | 2.398 |
| Ternary MoE | 12 / 6 / 8 | 13 / 19 / 64 | 2.616 | 2.271 |

The larger ternary MoE improves exact counts in all three seeds, with large
variation (2.2–10.7%). Dense improves in one seed and declines in two. The best
MoE seed answers 41/100 difficulty 1 addition questions, 4/100 difficulty 2
addition, 10/100 difficulty 1 comparison and 9/100 difficulty 2 comparison.
It answers zero of either task's 100 difficulty 3 questions. Thus the strongest
gain is on the training difficulty range, not the held-out harder range.
All 600 gold completions fit the context for every audited checkpoint.

This changes width, depth and FFN size together: a capacity bundle, not an
isolated width/depth effect. It does not rank architectures or optimizers.
It uses the same updates/tokens, not the same computation: dense parameters
rise from 57,536 to 361,088, MoE from 82,368 to 558,720 (active 57,792 to 362,112).
Linear forward MACs per token, excluding norms/attention/nonlinear work, rise
from 57,344 to 360,448 dense and 57,600 to 361,472 MoE: about 6.3x. The same
learning rate need not be optimal for both sizes. No learning-rate search,
environmental interaction or multi-step reinforcement learning ran.

`summary.csv` preserves every exact count, role loss and token-weighted
validation loss; audit/eval/CSV/log files, hashes and sampler checks are
retained. `decode-memory-raw.csv.gz` profiles seed1 checkpoints in 30 fresh
native processes (three rounds per size/path), batch one, 16 forced tokens,
with the memory probe binary from `50efa7d`. Linux counters have the limits
described in `../memory-follow-up/README.md`. For larger ternary MoE, packed
owned arrays are 558,025 bytes, median VmRSS 2,998,272 and reported peak
9,420,800 bytes; the peak includes the unpacked source model. The fp32 clone
owns 4,779,356 bytes and has median VmRSS 7,073,792. Its Model retains training
buffers, so this comparison does not isolate ternary weight compression.
No same-build small/large training throughput comparison was made: small
checkpoints were reused. No GPU runtime assertions ran.
