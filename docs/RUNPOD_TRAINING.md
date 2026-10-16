# Fresh European Runpod training, 2026-10-15

This run starts from random weights. It uses the repository's native C++/CUDA
trainer, a fresh 16,384-token byte BPE tokenizer, and a $10 total Runpod budget.
It does not require files from the owner's offline computer.

## Data

The target is approximately **10 billion measured, deduplicated training
tokens**, excluding validation/test tokens and sampling repetitions. Assembly
continues while training starts on the committed starter shards. The live count
and completed source shards are in `data/licensed-10b/manifest.json`.

Pinned upstream revisions and raw SHA-256 checksums are in
`configs/licensed-10b-sources.json`. The candidate sources are:

| Source | Selection and provenance |
| --- | --- |
| OpenStax through Common Pile LibreTexts | OpenStax-attributed canonical Bookshelves entries; explicit CC BY 4.0 |
| Common Pile DOAB | Curated open-access books; explicit CC BY 4.0/CC0 and author/source metadata |
| Common Pile arXiv | Openly licensed academic papers, including mathematical text and proofs; these are not all peer reviewed |
| Common Pile PubMed | Licensed biomedical/life-science research used to fill the remaining corpus target |
| OpenAI GSM8K | MIT-licensed human-written worked math problems, separately tracked as a small training supplement |

Common Pile dataset cards are retained under `data/licensed-10b/cards`.
Publisher-reported licenses are retained evidence, not independent certification
or a rights warranty. The bulk pipeline accepts explicit CC BY 4.0 and CC0 1.0
license URLs and rejects ambiguous, missing, noncommercial, no-derivatives and
share-alike records. It also requires source URLs and author metadata.

Each kept document retains its complete upstream metadata, source revision,
raw-shard line number, license, normalized text hash, split, token file, and
exact token offset/count in a compressed attribution file. Raw shards are
checksum-verified before use, then removed locally after committed tokenization
to fit this workspace. They remain retrievable from their pinned upstream URLs.

Splits group all sections of a book, or versions of an arXiv paper, together.
Normalized exact-document duplicates are removed globally. Near duplicates are
not comprehensively removed. The tokenizer sample uses only training splits.

Math coverage has distinct labels/counts for math textbooks, mathematical titles,
and proof-oriented arXiv text (the last is a heuristic). Training upsamples arXiv
and a separately tracked math sampling file. Repeated sampling is not counted as
new data toward the 10B target.

GSM8K retains its original test split and reserves 500 training questions as dev
data. Seven near-copies of held-out questions were removed, leaving 6,966 training
questions. Its answer-only loss masks are preserved. Dev/test questions never
enter this supervised math supplement; public literature is not exhaustively
screened for benchmark quotations.

## Model and run

- 267.480M total parameters; 97.610M active per token.
- Width 768, 12 layers, 12 attention heads, 8 experts, top-2 routing.
- Context 512; batch 16 with 8 accumulation steps (65,536 tokens/update).
- BF16 matrix inputs and fused tensor-core attention; fp32 weights/optimizer.
- Muon; learning rate 0.0015 with warmup, clipping and cosine decay.
- Optimizer/RNG state resumes between 20-minute stages so new data can be added.
- First benchmark measured about 94,000 tokens/second. That is training-step
  throughput, excluding data loading, validation, checkpoint I/O and transfers.

The budget can train on approximately 3B sampled tokens at that speed, **not a
full pass over a 10B-token corpus**. Actual training steps and loss are recorded
in `runs/fresh-eu.csv`; each stage records its exact data paths and command.

The RTX 4090 is in Romania (`EU-RO-1`), Pod `sg78jrp2eapkyj`, at the observed
$0.89/hour compute rate. The checkpoint network volume is `llweco3tg6` (50 GB,
$3.50/month prorated hourly); container storage also bills. Training is scheduled
to end at 07:10 UTC on 2026-10-16, with an independent GPU stop at 07:45 UTC.
Expected total through that deadline is approximately $9.25, below the $10 cap.

CUDA compilation passed. The original GPU suite reported three tolerance
failures in fp32/ternary configurations. Its BF16 MoE/Muon checks passed, and an
additional BF16 MoE check with the selected 64-value attention heads passed
under the existing BF16 tolerances. Full logs are retained; the suite is not
represented as entirely passing.

## Recovery

`/workspace/library-files/CxxGPT-recovery.tar.gz` is an atomically replaced
downloadable archive. It contains the working source, tokenizer, source/attribution
manifests, math supplement, operational scripts without credentials, logs and,
after the first verified copy-back, the latest backed-up checkpoint. Its
`RECOVERY.json` states explicitly whether a checkpoint is included.

Checkpoints are saved atomically on the network volume. A recovery supervisor
hard-links a stable snapshot, downloads it with resumable byte ranges, verifies
its SHA-256, and packages the archive. After final verified recovery, it stops
the Pod and removes this run's billable resources. A separate guard stops the GPU
at the absolute deadline even if the backup supervisor fails. Periodic off-Pod
copies provide recovery if the account balance runs out unexpectedly. These
processes require the cloud workspace to remain running.

The bulk corpus is separate from the recovery archive (about 20 GB of token IDs
at 10B tokens). Its files, counts, provenance and source checksums remain in the
cloud workspace. No credentials, browser callbacks, OAuth tokens or Jupyter
tokens are included in the recovery archive, and nothing is published to GitHub.

## Reproduce data preparation

```sh
make cxxgpt
python3 scripts/prepare_licensed.py
python3 scripts/build_licensed_corpus.py
```

The bulk command resumes committed shards. `prepare_math_supplement.py` prepares
the separate math sampling file after GSM8K has been downloaded using the repo's
`scripts/get_data.py` and encoded with the saved corpus tokenizer.

## Math accuracy and general-language continuation

The broad-pretraining checkpoint scored **6/500 (1.2%)** on the reserved
GSM8K development questions: greedy BF16 GPU decoding, no calculator, up to
400 new tokens. It left 61 answers unfinished; one correct final answer had
an incorrect intermediate calculation. This is a development diagnostic,
not an official-test headline. The laptop ledger records night 2 at
202/1319 (15.3%) on the original test, a different split and training recipe.

After the owner asked for better math while retaining general ability, the
continuation uses approximately **53% worked math and 47% general text** by
sampling weight. The worked problems contain 963,089 locally generated
questions (193,460,542 tokens) and 6,966 human-written GSM8K training questions.
The generator uses separate train/dev sentence patterns. Training questions
sharing 13-word runs with GSM8K dev/test or generated dev were removed.
The mixture weights are word data x4, GSM8K x64, plus 750,046,644 replay
book/paper tokens. Answer masks and QA-aware sampling preserve questions
when a complete problem fits the context.

Every stage records exact answers on all 500 dev questions and all generated
responses. It also measures fixed held-out DOAB book loss with **zero training
updates**; the broad-checkpoint reference is 2.5745. The latest balanced model
and the best math-dev model are retained separately. Test scores do not guide
selection. The final evaluator scores both frozen models on the original
1,319-question test, skipping duplicate computation if their weights match.

The recovery archive retains the latest full checkpoint with optimizer/RNG
state, and an additional best-math inference checkpoint without optimizer
moments. Native export checks every parameter, router bias, training step and
RNG bit-for-bit after reload. Use `--resume` with the full latest checkpoint;
use `--init` for new training from the best inference checkpoint. The bulk
training data remains separate from the download.
