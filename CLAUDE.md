# CxxGPT

## Goal

An experimental language-model project with our own model implementation,
training engine, tokenizer and kernels in C++ and CUDA, trained on public
data. The goal is to improve measured math problem-solving accuracy and
reduce training and inference costs within a laptop's hardware budget.
Current results do not establish general reasoning ability or an overall
efficiency advantage over other implementations.

The approaches being tested are:
- ternary (1.58-bit) weights, trained with BitLinear;
- int8 activations;
- sparse Mixture-of-Experts;
- hand-written kernels, with no Python framework anywhere (small Python
  scripts for data handling are fine; no PyTorch).

Math problem-solving accuracy is the first evaluation focus.

## Where things stand

- `src/`: the training engine (`model.h`, `ops.h`, `gemm.h`, `train.h`),
  tested against double-precision gradients. Dense or MoE FFN, optional
  ternary BitLinear projections, aux-loss-free MoE balancing. AdamW, or
  Muon (`--muon`) for the matrices inside the blocks.
- `src/ternary.h` and `src/infer.h`: inference on bitplane int8 kernels
  (AVX2, AVX-512, VNNI). A trained ternary checkpoint runs on them. The
  embedding/output layer, and attention with `--fp-attn`, are int8 rows
  (one scale per row); `eval-reasoning --as-engine` rounds a model the
  same way. Training, the packer and checkpoints give every weight the
  same -1/0/+1 code, at the rounding thresholds too, with no bf16 cast on
  the way (#40, `tests/test_ternary_codes.h`).
- `--ternary --fp-attn` (ternary FFN, full-precision attention) gave the
  best MoE run on the curriculum (71.9%, against 64.2% all full precision
  and 62.4% all ternary), but chain accuracy "clicks" at a different step
  in each run, so one run per model does not settle it: repeat with
  several seeds (fast on the laptop GPU) before the ternary nights. Those
  seed repeats need new baselines too: since 2026-10-16 the curriculum has
  no degenerate zeros (below), so runs before and after are not
  comparable. Until
  then `night.sh` `TERNARY=1` keeps attention full precision (README,
  "Ternary FFN, full-precision attention").
- `src/tokenizer.h`: byte-level BPE; digits are never merged.
- `src/reasoning.h`: a synthetic reasoning curriculum with held-out harder
  test levels (`gen-reasoning --chain-plan`: chain solutions look the
  variables up first, which more than doubles chain accuracy). Since
  2026-10-16 (an audit) no chain variable is 0 (about 10% of chain answers
  were 0, through "u=n-n" or "f=t-2" with t=2, and then "0*8"), no add
  operand is 0, eq's x is at least 1, and expr never computes a-a: every
  curriculum result before that date (the tables in README and
  docs/research.md) is not comparable with runs after it. `cxxgpt
  prep-qa` turns GSM8K-style JSON Lines into the same format.
  `gen-reasoning --mistakes P` plants one wrong statement and its
  correction in a share P of the training solutions; the wrong statement
  stays visible but gets no loss (`[[no-loss]]` spans, `src/loss_mask.h`,
  also read by `text2tok`). Validation and test stay clean.
- `src/calc.h` (`--tools calc-notes|calc-plain` on `sample` and
  `eval-reasoning`): an exact calculator (fractions, 64-bit, no rounding)
  that finishes the arithmetic the model starts (`<<48/2=` or `16 x 7 = `);
  `eval-reasoning` then prints the score without and with it, and the call
  counts. Neither mode fires on the curriculum's `a*b=c` steps
  (docs/calculator.md). Since 2026-10-17 notes read strict digit-grouping
  commas (`<<24*600,000=` gives `14400000>>`; `1,2345` or `2,3+4` stay
  malformed); plain mode still stops at a comma. Until 2026-10-17 no training file had notes
  (prep-qa strips them without `--keep-calc`; the generators ran with
  `--notes off`), so calc-notes never fired. Night 2's math replay keeps
  them (`operations/build_gsm8k_notes.py`, `build_targeted_math.py` and
  `build_gen_math.py --notes on`): the same problems as before, checked
  byte for byte with the notes removed, every note's result exact (docs/calculator.md,
  "Notes in the training data"). `train --reason-tools calc-notes` takes
  each training exam again with the calculator, on a line of its own
  (docs/calculator.md, "In training's exams").
- Scoring (since 2026-10-17, `canonical_answer` in `src/reasoning.h`):
  the text after `#### ` is compared with the gold answer as a number,
  exactly, digit by digit (no rounding): "57", "57.00", "$57", "57%" and
  "57 dollars" all match "57", "0.8" matches "0.80"; "57 or 58", "5.7",
  "570", "-57", "1 million" and "57 or" do not. Anything that is not
  one number (and its `$`, `%`, unit words, grouping commas, a final
  ".") is compared as text.
  eval-reasoning, the vote (equal numbers are one answer), failure labels
  and self-train all go through it. The old text match changed no saved
  score (README, "How answers are scored").
- `eval-reasoning --tries N --temp T --seed S` (or `--effort
  low|medium|high`: 1/8/32 tries): samples each problem N times and votes
  on the final answer (maj@N, also pass@N and the first try's score). Each
  try has its own seed, so results do not depend on `--batch` or
  `--limit`; every headline names its effort (`(greedy)`, `(maj@8, ...)`).
  `--no-repeat N` (also on `sample`, off by default; `NoRepeat` in
  `src/no_repeat.h`) forbids writing a run of N generated tokens twice,
  against greedy loops (#33); every evaluation prints its unfinished first
  tries. Not yet scored on a dev set. `--labels FILE.jsonl` (#16,
  `src/failure_labels.h`) labels each first try: correct, unfinished,
  format, arithmetic (a written calculation the exact calculator finds
  wrong), passed_gold (the right number in the work, another answer
  given) or other (for a person to judge), with flags for loops and
  numbers from outside the question; one JSON line per problem. Since
  2026-10-17 (#72) it also reads plain arithmetic right after a word
  ("30% of 270 is 270*30/100 = 81", `after_word`; not after one letter
  or an operator word), which the calculator leaves alone: labels made
  before count those as unreadable.
- `train --unlikelihood ALPHA --ul-ngram N` (#34, off by default, CPU
  and GPU): at each trained position, every token that would
  repeat a run of N trained tokens of the same document (the no-repeat
  block's definition, `src/unlikelihood.h`) adds `ALPHA * -log(1 - p)` to
  the loss, never the target, never from masked text. The loss and
  gradient (`ops.h`) are tested against finite differences in double,
  with `1 - p` summed from the other tokens (exact near p = 1). The log
  keeps the cross-entropy as `loss` and adds the term and how often it
  fires (also as two more CSV columns). On the GPU the negatives are
  found on the device too (`ul_count`, `ul_offsets`, `ul_write`) and the
  term computed in two kernels (`ul_fwd`, `ul_fix`, `src/gpu.cu`);
  `test_gpu` checks the negatives, loss, term and gradient against the
  fp64 GPU reference, a saturated term, and a term that never fires.
- `src/gpu.cu` (interface `src/gpu.h`): the CUDA training backend, the
  same model op for op, fp32 with cuBLAS. `train --gpu` stores the
  products' inputs in bf16 by default (`--matmul bf16`, fp32 master
  weights: 1.45x the tokens/s of `--matmul tf32`, the same loss curve);
  ternary models too, since 2026-10-16 (owner's decision): they multiply
  their codes (-1/0/+1 weights, int8 activations: exact in bf16, with
  exact fp32 sums) and apply the scales after the sum, so the forward
  products are exact and only the gradients are rounded, as in bf16
  models (`proj_forward`): 2486 against 3600 ms a step on the ternary
  night model, the same losses over 200 steps from scratch and 100 from
  the RunPod model turned ternary (`review/baselines/curve-tern-*`); not
  yet compared over a whole night. Fused
  attention kernels are the default (`--no-fused-attn` for cuBLAS
  products): on the tensor cores with TF32/bf16
  (`src/attention_tc.cuh`, tested on the GPU only; 1.18x the cuBLAS
  attention's tokens/s on the night-1 model, at 8 x 16, which only fits
  with them), in exact fp32 otherwise (`src/attention.cuh`, also tested on
  a CPU emulator, `tests/cuda_emu.h`). Since 2026-10-16 the tensor-core
  kernels use mma.sync with scores and softmax in registers
  (FlashAttention-2 layout, the same TF32 rounding), and the backward
  writes dqkv instead of adding to a zeroed one: attention 94 + 293 ->
  50 + 191 ms a step, 2348 -> 2205 ms on the night model (-6.1%, 27.9k ->
  29.7k tokens/s; docs/research-plan.md "Speed", item 9).
  `--profile` times each part of a step. Since 2026-10-16 the step's
  sums (loss, gradient norm, clip, the skip of a non-finite step, router
  balancing) stay on the GPU, MoE experts in bf16 run as one grouped
  product per projection without waiting for the host
  (`src/grouped_gemm.cuh`, #49; TF32 and fp32 use cuBLAS per expert,
  faster there; `--moe-grouped`/`--moe-cublas` force one), tokens for
  decoding are chosen there
  (only ids come back), and new models are initialized there
  (`GpuTrainer::init_params`, other weights than the CPU's `Model::init`
  for the same seed). The speed plan's fusions (2026-10-16,
  docs/research-plan.md "Speed", item 7): buffers written instead of
  zeroed, a one-pass RMSNorm backward, SwiGLU's backward in the bf16
  `eproj` product (`launch<..., Fused>`), Muon's norm over many blocks:
  2665 -> 2489 ms a step on the night model (-6.6%, 24.6k -> 26.3k
  tokens/s, the same 7.0 GB); then all the experts' bf16 products on one
  mma.sync kernel that fits two blocks an SM (item 8, #49): 2460 -> 2341
  ms (-4.8%, 28.0k tokens/s); in TF32 (ternary models) the same kernel
  layout: 4186 -> 3810 ms a step, but cuBLAS per expert is faster there,
  so TF32 defaults to it (3627 ms). Training is not bitwise
  reproducible (atomic adds in the gradients), so "same numbers" means
  within the spread of repeated runs.
  `tests/test_gpu.cpp` checks it against `src/gpu_ref.cu`, the model in
  double precision on the GPU (plain kernels, one output per thread,
  sharing no code with `gpu.cu`): losses, gradients by finite differences
  along random directions, optimizer steps, decoding (the reference's
  whole forward on each prefix) and eval-reasoning results. The
  reference itself is checked against the CPU engine in double (losses to
  2e-16, finite differences within 7e-7 of the CPU's gradients). Bounds
  are 3-5x what the RTX 4070 measured, and everything passes: the two
  fp32 attention checks that missed their bounds against the CPU's float
  engine (#27) were that engine's error (7e-7 to 2e-6 against the
  reference). It takes about 2.5 minutes. Training's reasoning exams run
  on the GPU too (`GpuTrainer::decode_begin/decode_step`, a KV cache kept
  in the gradient buffer between steps): 16 s instead of 5 minutes for
  250 problems on the night-1 model. Build with `make gpu`, giving
  `cxxgpt-gpu` (`train --gpu`) and `test_gpu`. While the host waits for the GPU it
  sleeps (`cudaDeviceScheduleBlockingSync`): a quarter of a core instead
  of a whole one during training, at the same speed.
- `docs/research.md`: the research log from the merged `gpt/research`
  branch (stable cross-entropy, token-weighted accumulation, optimizer
  grids, digit curricula), with raw results under `docs/results/`.
- `docs/research-plan.md`: what to try next, in order (a calculator in the
  engine, then more tools through the same hook, taught by a program that
  writes word problems; finer experts, learning from checked solutions,
  place-value digit embeddings, int8 tensor cores for ternary), effort
  levels (more tries, a judge, thinking loops), our own ideas (numbers
  backwards inside the model, planted mistakes, cloning dead experts,
  experts that compute exactly, and more), a ladder for growing past
  8 GB to 1-2 B parameters (recompute, bf16, 8-bit optimizer state, the
  optimizer in CPU RAM, streaming layers), where a GPU step's time goes and
  the speed work in order (a fast fused attention kernel first), and the
  seed-repeat funnel an idea passes before it gets a laptop night.
  Decisions are made on a dev set held out of GSM8K train, never on the
  GSM8K test set, which gives headline numbers only.
- `src/wordgen.h` (`cxxgpt gen-word`): GSM8K-style word problems with exact
  answers and worked solutions (with or without calculator notes). Six
  story shapes (a running quantity, comparisons, totals over people,
  rates, a fraction or percentage of a part, purchases and change), 2-8
  operations, distractor sentences, money spelled three ways; 162 sentence
  patterns, 54 of them held out for the test split. Besides the plan's own
  exact check, a second solver works the five newer shapes out from the
  story's quantities alone. Audited 2026-10-16 (README, "Generated word
  problems"): no "1 cards", realistic payments, kept items with a source,
  at most two cuts per chain, no wording used twice in a story; the tests
  check each rule. `data/targeted-math` was rebuilt with it (README, "The
  training data").
- `src/mathgen.h` (`cxxgpt gen-math --topic equations`, 2026-10-16): word
  problems with one unknown (number puzzles, ages, two kinds of tickets,
  animals or packs, consecutive numbers, rectangles, three-way shares),
  built from a chosen whole-number answer, solved in the text as an
  equation with calculator notes. Every problem is solved again before it
  is written by `equation_check`, which shares no code with the generator
  (trying every whole number the story allows; exactly one must fit), and
  every note by calc.h; tests show the checker rejects broken problems.
  84 sentence patterns, 28 held out for test. `--exclude
  data/licensed-10b/gsm8k-test.txt,data/licensed-10b/gsm8k-dev.txt` is
  needed for any training use: without it a large file repeats a few GSM8K
  test questions almost word for word. With it, 0 of GSM8K test and dev
  share a 13-word run, or their numbers and answer, with 300k problems.
  `src/mathgen_fractions.h` (`--topic fractions`): parts of an amount
  (fractions and percentages, 1-3 steps), price changes in exact cents,
  what percent, ratio splits, a ratio and one count, proportions, unit
  conversions; answers whole or with at most two decimals;
  `fraction_check` works each story out again its own way. 120 patterns,
  40 held out. Both topics share `math_detail` (the story, distractor and
  text) and `math_check` (notes and the stated answer). Not in the
  training mix (the owner decides). Next topics, in order: number theory,
  counting (the answer format of probability needs a decision first).
- `operations/build_libretexts_problems.py` (2026-10-17): 383 human-written
  word problems with their book answers (one number, no worked solution)
  from the CC BY math textbooks in Common Pile's LibreTexts (OpenStax 1e
  copies, made before OpenStax went CC BY-NC-SA in March-April 2026), into
  `data/libretexts-problems/`, with each page's licence and address
  (second build: `data/libretexts-problems-2/`, 475, 93 new, with "Answer:"
  layouts, fractions in questions and nursing dosage problems; the
  LibreTexts file holds about 1,000 such problems at most). Kept
  out: GSM8K dev/test overlaps, copies, unclear pairings
  (`tests/test_libretexts_problems.py`). 100 checked by hand; the books'
  own keys are wrong a few percent of the time. For self-train and
  evaluation, not the mix; not held out (LibreTexts is in training)
  (README, "Textbook word problems with answer keys").
- GSM8K dev set: `scripts/get_data.py` holds out 500 GSM8K train problems
  (the 500 questions with the smallest sha256); `DEV=1 scripts/night.sh`
  validates on it instead of the test set. `cxxgpt leak-check` finds
  questions sharing a 13-word run with training files.
- `cxxgpt qa-context` (`src/qa_context.h`, #15): for every trained answer
  token, the exact share of `sample_batch`'s random windows that show its
  whole question, part of it, or none. At `--seq 512` (2026-10-15): GSM8K
  74/15/11%. `train --sampler qa` (`QaStarts`, `src/train.h`; default
  `random`) moves a window that starts inside a problem back to the
  problem's BOS when the problem fits in a window (longer problems keep
  random starts): whole question 74 -> 100% (GSM8K), at fewer answer tokens
  per window. Same random draws as `random`; validation keeps random
  windows. Accuracy effect not
  measured: the A/B is in docs/research-plan.md ("Answers with their
  questions").
- `train --load-log FILE`: each layer's share per expert over every log
  interval, with a warning for an expert under 1%; `night.sh` writes
  `runs/$RUN.load.csv` and `report.sh` sums it up. `cxxgpt clone-experts`
  (`src/clone_experts.h`) repairs a checkpoint offline: it copies a busy
  expert into a starved one (`--layer --from --to`, or `--auto --load-log`
  for every expert under 1% in the log's last rows) and resets both
  experts' biases and optimizer state. Night 1 had no expert under 1%.
- `CMakeLists.txt`: the native Windows build of the CPU engine (clang-cl;
  `cxxgpt.exe`, `ternary_moe.exe`, `test_engine.exe`), run in CI. The
  vector kernels are picked at run time from what the CPU and Windows
  support. The GPU build on Windows is the next step (the port).
- The model and its data (2026-10-16; README, "The training data" and
  "Data provenance"): the laptop continues the RunPod model (267M MoE,
  97.6M active, from random weights; `runs/runpod-2026-10-16/fresh-eu.ckpt`,
  step 46,946, 3.9% on GSM8K test) on its last mix, `night.sh`'s default:
  `data/targeted-math/` (973,193 audited `gen-word` problems x4, rebuilt
  2026-10-16 by `operations/build_targeted_math.py`; the RunPod model saw
  the pre-audit set, kept as `data/targeted-math-runpod/`; GSM8K
  train without the dev problems and 7 near-copies x64) and 12 shards of
  textbooks, DOAB books and arXiv papers from `data/licensed-10b/` (10B
  Common Pile tokens, CC BY/CC0, with each document's attribution), about
  53% worked math. The tokenizer is `data/licensed-10b/tok.bpe` (16,384,
  sha256 `6e811ced...`), and the GSM8K dev/test files for it are in the
  same folder: never pair a checkpoint with another tokenizer (equal
  vocabulary sizes, so nothing would catch it). `night.sh STEP0=46946
  LR=1.2e-4` continues the checkpoint (docs/LAPTOP.md, "Continuing the
  RunPod model"; not yet run on the laptop). Only openly licensed
  text, human-written problems, our own generators' output and our own
  models' output train the models, never text another company's AI model
  (or one under a restrictive licence) wrote. Our own model's output is
  allowed only if that model trained on clean data alone (owner's
  decision, 2026-10-16; README, "Data provenance").
- `cxxgpt self-train` (`src/self_train.h`, research plan item 3): samples
  `--tries` solutions per training problem (`sample_tries`, the sampler
  and seeds of `eval-reasoning --tries`, CPU or `--gpu`) and keeps a try
  only with the right answer, some work, every calculation right by
  calc.h (notes, the number after a note with its sign, a note with no
  exact decimal only rounded right, never whole, and plain `16 x 7 = 112`, as
  failure_labels.h finds them, sign included, chains `a = b = c` link by
  link), no repeated line (`--allow-loops`), no number from outside the
  question but a few constants (`--allow-outside`), every number of the
  question used (`--allow-unused`; off with `--recipes`), not the gold solution
  (`--keep-gold`) and no `[[no-loss]]` marker. "The same solution" (for
  dedupe and the gold check) is the same outline by default: the
  calculations' values in order, in any words (`--same text` for text
  only); at most `--keep K` per problem. Writes `OUT.txt` (problems),
  `OUT.tok` (prep-qa's masking: question masked, work and gold answer
  trained) and `OUT.json` (settings, counts). `--exclude` drops problems
  repeating dev/test questions; `--recheck AUDIT.jsonl` checks a run's
  tries again without a model. Tests: `tests/test_self_train.h` (scripted
  fake model) and `tests/test_cli.sh`. On rp1 (README, "First runs"): it
  writes back GSM8K train's gold solutions (memorized) and, on new
  gen-word and gen-math problems, the generators' own solutions (both are
  in its mix; gen-word's in other sentence patterns, caught by outline),
  so nothing is kept; pass@8 is 3-4 points above the first try on
  gen-math. Self-generated data is in no mix: the owner decides.
  Only clean solutions may be trained on (owner's rule, 2026-10-17):
  `cxxgpt score-checks` (`src/score_checks.h`) runs the checks on 546
  right tries of rp1 labelled by hand, blind (`tests/checker_labels.jsonl`:
  clean, padded, minor or lucky; tune and held-out halves by problem;
  README "How clean are the kept tries?"): on the held-out half 2.7% of
  the kept tries are clean (11.7% with minor slips), none clean is
  dropped. Try new checks on the tune half (`score-checks --show
  VERDICT:LABEL` prints its tries, never the held-out ones); `test_cli.sh`
  pins the numbers. Five stricter checks, off by default (`--strict` for
  all, on self-train and score-checks; README "Stricter checks",
  2026-10-17): `--strict-outside`, `--check-dead`, `--check-mismatch`,
  `--check-broken`, `--check-computed`. Together: held-out 5.2% clean
  (22.4% with minor), none clean dropped, but they drop 26% of GSM8K
  train's own solutions (13% today). What is left is right arithmetic on
  the wrong story, which no text check catches: the re-draw check is the
  way to tell.
  The re-draw check (`src/redraw.h`, README "The re-draw check",
  2026-10-17): `gen-word`/`gen-math --recipes FILE` writes how to make
  each problem again (the problems unchanged, byte for byte: the
  generators' story choices and number choices are told apart,
  `Choices`); `self-train --recipes FILE` (`--redraws 3`) makes each story
  again with other numbers (same masked question and masked worked
  solution; gen-math's second solver accepts it), turns a try's
  calculations into a formula over the question's numbers (earlier
  results substituted, other numbers constants) and keeps the try only if
  it gives every version's exact answer (verdict `redraw` if not). A
  number with two sources makes the try ambiguous: it passes if one
  choice of sources does (the default since 2026-10-17, owner's
  decision; `--strict-readings` drops it as `ambiguous`), no formula or no versions is
  `unchecked` (`--keep-unchecked`). The generators' own solutions: with
  one reading 36-67% pass (the rest ambiguous), with every reading
  895 of 896. On rp1 (README): no lucky try (it writes the generators'
  solutions), 35-67% of right tries ambiguous with one reading, 0-1%
  with every reading; the failure labels now read "30% of 270 is
  270*30/100 = 81" (`after_word`), which was 195 false failures. Tests:
  `tests/test_redraw.h`, `tests/test_self_train.h`.
- `src/cxxgpt.cpp`: the CLI (train, sample, eval, data prep, benchmarks).
- `scripts/get_data.py`: downloads GSM8K into `data/gsm8k/` as JSON Lines
  (for `prep-qa`), with the 500 dev problems held out of train (the same
  500 as the RunPod data's). Tested against a local fake server; CI's
  `data-sources` job checks the real files (`--check`).
- `scripts/compare.sh`: the architecture comparison on the curriculum.
- For the laptop (native Windows since 2026-10-15; the scripts run in Git
  Bash): `scripts/build_windows.cmd` (build, kernel, engine and GPU tests;
  `scripts/setup_wsl.sh` is the Linux equivalent, kept and run by CI),
  `scripts/probe.sh` (GPU speed and memory per model size; prints the
  night.sh command for each) and `scripts/night.sh` (one night of
  training, validated on the GSM8K test set, or dev with `DEV=1`; `STEP0=`
  continues a copied checkpoint, `INIT=` starts from an earlier night's
  weights; `VAL_SOURCES`: `train --val-sources`, `src/val_sources.h`, each
  held-out file's loss at every validation, GSM8K's final answers (after
  `####`) apart, in `runs/$RUN.val.csv`); `scripts/dashboard.py` shows a
  run live in the browser (only reads night.sh's files;
  `tests/test_dashboard.py`), `scripts/report.sh`
  sums up a night's log, and
  `scripts/ledger.sh RUN "note"` writes its row in `docs/runs.md`, the
  committed ledger of runs (tokens, hours, energy, scores; `runs/` is
  git-ignored). `docs/LAPTOP.md` is the runbook.
- `train --init FILE --ternary` turns a full-precision model ternary for
  the rest of its training (same weights, used differently).
- Steering a run without a cold start (2026-10-16, docs/LAPTOP.md
  "Changing a run while it trains"): `train --schedule wsd` (default
  cosine) keeps the learning rate flat after the warmup and cools down
  over the last `--decay-steps` (1 - sqrt, `lr_wsd` in `src/train.h`), so
  a run can be made longer at a resume, or a copy of a checkpoint cooled
  down with `--resume --steps STEP+D --decay-steps D` (trains exactly as
  an uninterrupted run, `tests/test_cli.sh`). The control file
  (`src/train_control.h`; `--control`, default `OUT.control`), read before
  every step, acted on once a change has stayed 0.5 s (a shell's `>`
  empties the file first): `lr-scale X` (holds while written, also
  after a restart), `save`, `eval`, `reason`, `stop` (once a change; never
  from a file already there at the start).
- Overnight safety: a step with a non-finite gradient is skipped, ten in a
  row stop the run without overwriting the last checkpoint, and
  checkpoints are written to a temporary file and renamed.
- `ternary_moe.cpp`: kernel tests and the kernel benchmark.

## Plan

1. CPU, tiny scale (this VM): compare dense, ternary, MoE and ternary MoE
   on the reasoning curriculum; fix what the evals expose.
2. GPU training on the owner's laptop (RTX 4070 8 GB, native Windows):
   - a CUDA backend for the same model, tested op by op against the CPU
     engine;
   - overnight runs on public data (general text plus math and reasoning
     sets): full precision first, then the last nights switched to
     ternary (`night.sh` `INIT=... TERNARY=1`), which on the curriculum
     cost half as much as ternary throughout; attention stays full
     precision (`TERNARY=1` gives `--ternary --fp-attn`).
3. Inference on our ternary kernels: CPU first, then GPU.

## Hardware

- This VM: 4 vCPUs (AVX-512 + VNNI), 15 GB RAM, ~30 GB free disk, no GPU.
  huggingface.co is blocked; GitHub raw files and PyPI work.
- The owner's laptop:
  - Windows 11, used natively (WSL2 was removed on 2026-10-15: native
    trains the same model 7% faster, without WSL's slow stretches); Visual
    Studio Build Tools 2022, LLVM clang-cl 23, CUDA 13.3, Git Bash; the
    clone is `C:\Users\andre\Desktop\CxxGPT`, with `data/` and `runs/`;
  - 8-core/16-thread AMD (Zen 4, AVX-512), 32 GB RAM;
  - RTX 4070 laptop GPU with 8 GB VRAM;
  - about 90 GB free disk;
  - available for training overnight only. In the day the owner's other
    project (PAROIKOS) needs the CPU: no heavy CPU runs then without
    asking, and timings taken while it runs are not clean.

## Conventions

- C++20 (GCC 13 has no `<print>`, so not C++23 yet).
- Commit directly to `main` (owner's instruction), and push each verified
  step.
- Checks (owner's rule since 2026-10-15; GitHub Actions is free for this
  public repository):
  - before pushing, a quick local check: on the laptop,
    `scripts/build_windows.cmd` (clang-cl with `-Werror`, the kernel,
    engine and GPU tests); on a Linux machine (the VM), build with `-Wall
    -Wextra -Werror` (g++ or clang++) and `make test`. The shell CLI
    tests, asan and tsan are Linux-only: from the laptop, CI runs them;
  - CI runs the full gate on every PR and every push to `main`: g++ and
    clang++ at three CPU levels, `make asan`, `make tsan`, the CUDA
    builds and the native Windows build. A PR merges only with CI green;
    if `main` goes red, fix it next;
  - GitHub's runners have no GPU: `build\test_gpu.exe` on the laptop
    (`build_windows.cmd` runs it), once per batch of changes that touch
    training or `gpu.cu`.
- The CUDA code cannot run on this VM. CI compiles and links it with nvcc;
  here, compile-check it with clang's CUDA mode (`clang++ -x cuda
  --cuda-gpu-arch=sm_89 --cuda-device-only`, then `--cuda-host-only`)
  against the toolkit headers from the PyPI `nvidia-*` wheels. Run
  `make gpu-test` on a GPU before trusting a change to `gpu.cu`.
- Every numeric change is tested against a double-precision reference:
  GPU code against the fp64 model on the GPU (`src/gpu_ref.cu`, in
  `test_gpu`, which also checks that reference against the CPU engine),
  CPU code against the CPU engine's double-precision tests. Wherever the
  code claims two paths agree (SIMD and scalar, decode and forward,
  engine and training model), a test checks it.
- The GPU is the default for training and evaluation, and new model
  features come to the GPU first. The CPU engine stays (owner's decision,
  2026-10-16: its tests run in CI), and refuses, with a clear message,
  any option or configuration it does not implement, rather than
  computing something else (`train` without `--gpu` refuses `--matmul
  tf32/bf16` and `--fused-attn`/`--no-fused-attn`).
- Benchmarks:
  - use honest baselines: vectorized fp32 on the same threads, and later
    real runtimes;
  - this VM is noisy, so interleave A/B runs and report medians.
- The owner is a C++ beginner without an ML background: explain decisions
  in plain words, and keep code readable.
