# Research plan: beating scale on one laptop

Written 2026-10-14, during night 1. This is the plan for what comes after the
first nights. `docs/research.md` is the log of what was already measured.

## Why we need new ideas

At our size, general-purpose models score almost nothing on GSM8K, even after
enormous training. The figures below are approximate, from the published
model cards:

| Model | Parameters | Training tokens | GSM8K |
|---|---|---|---|
| GPT-2 small | 124 M | ~10 B | ~0-2% |
| Pythia-160M | 160 M | 300 B | ~1-2% |
| SmolLM2-135M | 135 M | 2,000 B | ~1-2% |
| SmolLM2-360M | 360 M | 4,000 B | ~3% |
| Qwen2.5-0.5B | 500 M | 18,000 B | ~40% |

One night on the laptop trains about 0.37 B tokens. SmolLM2-135M's budget
would take about 15 years of nights. We cannot win on scale, so every idea
below aims at one of three things:

- getting more out of each parameter;
- getting more out of each token;
- getting more tokens per night.

## The funnel: how an idea earns a night

1. Implement it behind a flag, with tests. The default behaviour is unchanged.
2. Run it on the reasoning curriculum (`scripts/compare.sh`), with **at least
   3 seeds** against the same number of baseline seeds. Single runs disagree
   by several points (CLAUDE.md, "clicks at a different step").
3. Keep it only if it wins clearly across seeds. Otherwise record the result in
   `docs/research.md` and drop it.
4. Only then spend a laptop night on it, compared with the previous night's
   numbers.

## Honest measurement: choose on a dev set, report on the test set

Night 1 uses the GSM8K **test** set both for the validation loss and for the
exams. Using it once is fine. But if we pick settings night after night by
whichever scores best on it, we slowly tune toward those exact 1,319 problems:
benchmaxxing by selection, with nothing dishonest intended.

**Rules.**
1. Decisions (which setting, which idea, when to stop) are made on a **dev
   set** and on the curriculum, never on the test set.
2. The test set gives headline numbers only, at milestones. `docs/research.md`
   records each time it was used.
3. Every tool that helps the model (the calculator, later retrieval) is
   reported as two scores, with and without the tool.
4. Every score states its effort level (see "Effort levels" below): 32 tries
   and one try are not the same test.

**The dev set.**
- `scripts/get_data.py gsm8k` holds out exactly 500 GSM8K train problems:
  the 500 whose question (with runs of whitespace made single spaces) has the
  smallest sha256 digest. A question's digest depends only on its own text,
  so the choice depends neither on Python's random module nor on the order
  of the upstream file. It keeps the upstream file in
  `data/gsm8k/raw/train.jsonl`, writes the remaining problems to
  `train.jsonl` (in upstream order), and writes the held-out problems to
  `dev.jsonl`. Repeating preparation selects the same problems; the official
  test file is unchanged. (The first version of PR 5 picked the 500 with
  `random.sample` and seed `20261014`; that split was never used, and this
  one replaces it.)
- `scripts/prepare_data.sh` also writes `gsm8k/dev.txt` and `dev.tok`.
  `DEV=1 scripts/night.sh` uses those files for both `--val` and
  `--reason-test`. The default remains `DEV=0`, using the test files so the
  original night-1 command is unchanged. Use `DEV=1` when choosing settings.
- MetaMath and OpenMath2 are built from GSM8K train problems, so taking the
  500 out of `gsm8k/train` alone is not enough:
  - MetaMathQA rows carry `original_question`; drop the rows whose original is
    a dev problem.
  - OpenMath2 rows with `problem_source` `gsm8k` repeat the problem text; drop
    exact matches. Rows from `augmented_gsm8k` are new problems with no link
    back to their source, so a small leak remains. Write it down.
- The downloader now applies both filters after normalizing whitespace and
  prints the number of excluded rows separately for MetaMathQA and OpenMath2.
  MetaMath rows without `original_question` stop preparation rather than
  silently bypassing the filter. Preparing either dataset by itself first
  creates/loads the same GSM8K dev split.
- The remaining `augmented_gsm8k` leak is **not removed**: those rows have no
  original-question link, and exact text matching cannot identify reworded
  versions of a dev problem. The dev set is therefore not guaranteed free of
  semantic overlap. The filter deliberately does not discard all augmented
  problems or try to infer their origins.
- Rerun `scripts/prepare_data.sh` to replace old, unfiltered math data. It
  rebuilds question/token pairs when their JSONL source is newer, so an old
  `train.tok` cannot silently retain the removed problems. No prepared data
  files are stored in the repository.
- To audit literal overlap, run
  `./cxxgpt leak-check --questions data/gsm8k/dev.txt --train A.txt,B.txt`.
  The question file may
  be a `prep-qa` five-column problem file (only its question field is used),
  or plain text with one question per line. Words are ASCII letters/digits,
  lowercased; punctuation separates words, and so do the stored escapes
  `\n`, `\t`, `\r` and `\uXXXX` of `.txt` and JSON Lines files. A hit requires 13 consecutive
  words, including across training-file line breaks. Each training file is
  scanned separately and reports its number of hit questions and up to three
  examples (1-based question numbers). Repeated matches count once per
  question. Shorter questions cannot match. This detects literal overlap,
  not paraphrases. The scanner stores the question index and hit flags, a
  fixed 64 KiB read buffer, and at most one question-sized word; training-file
  length and unbroken training tokens cannot make its memory grow.
- Night 1 trained on all of GSM8K train. A model that continues from night 1
  has seen the dev problems, so the dev set is only clean for a model trained
  from scratch. If we keep night 1's weights, use a separate word-problem set
  as dev instead (for example SVAMP or ASDiv; still to check).

## Expert balance: neglected experts in night 1

**What we saw.** Night 1's load (busiest expert's share, worst layer; 1.0 is
even, 4.0 the worst possible at 8 experts, top-2) fell from 3.98 to 1.83 by
step 600, then rose to 2.5-3.2 from step 900 on.

**What the router biases showed at step 2000.** The biases are the last 384
bytes of the checkpoint (`src/train.h`, `save_checkpoint`), so they can be read
without loading the model. Each step moves a bias by ±0.001, toward an even
load, so after 2000 steps no bias can pass ±2.0.
- Layers 0 and 6-11 are healthy: small biases of both signs, spread under 0.8.
  The nudging works there.
- Layers 1-5 each have one or two **neglected experts**, with a large bonus:
  +1.3 to +1.7, and +2.000 for layer 4 expert 2 and layer 5 expert 6. Those two
  were below an even share on every step since the start.
- About 7 of the 96 experts are mostly idle.

**Likely cause (not yet measured).**
- An expert that rarely gets tokens barely learns, so when it is picked it
  does badly, and the router avoids it more.
- Our bias is added to raw router scores (`src/model.h`, `moe_forward`), which
  have no ceiling, so the gap can grow faster than the bias closes it.
- DeepSeek-V3, whose method and default speed (0.001) we use, adds the bias to
  scores squeezed into 0-1 (a sigmoid), where a large enough bias always wins.

**What to try.**
1. Faster nudging, no code: `--bias-rate 0.003` and `0.01`. Watch that the
   healthy layers do not start flip-flopping.
2. Bounded scores: select experts by sigmoid score plus bias, keeping the gate
   weights from the unbiased scores. This needs a change in the CPU engine, the
   GPU and the inference engine, tested against each other as usual.
3. Per-layer load in the log, so this shows during a night instead of only in
   the checkpoint.

**How to test.**
- The curriculum is not a good test here: its small models balanced fine
  (load about 1.1). The trap appears at night 1's size.
- Instead, use short runs of the night-1 model on the GPU: night 1's `train`
  command as `night.sh` builds it, but with `--time-limit 50` (about 600
  steps), one run per setting.
- Keep `--steps 5700` rather than a shorter `HOURS`: `night.sh` derives the
  step count from `HOURS`, and the learning rate should follow the same
  schedule as a night.
- Compare the load lines, and read the biases at the end. Good balance means
  no expert's bias moves at full speed in one direction (|bias| close to steps
  × rate).
- That is about 50 minutes per setting.

**Before night 2.** If night 2 continues from night 1's weights, it inherits
the neglected experts. A faster nudge may revive them, and the short runs
show whether. If night 2 starts fresh (finer experts, bet 2), take the
balancing fix along: more experts means more of them to keep alive.

## Data: teaching it better without code

**What night 1 learns from.** These are rough shares by token-file size, from
night 1's default `MIX` in `night.sh`:

| Source | Share | What it is |
|---|---|---|
| FineWeb-Edu | ~37% | educational web pages |
| OpenMath2 | ~32% | math problems with worked solutions |
| TinyStories | ~24% | simple children's stories |
| MetaMath | ~6.5% | rephrased GSM8K and MATH problems |
| GSM8K train, ×4 | ~0.3% | the 7,473 real problems |

- About a quarter of training goes to children's stories. They help grammar
  early on, but they do not teach reasoning.
- A night trains on about 19% of the 1.96 B tokens, so even at ×4 a night
  shows each GSM8K problem less than once on average.

**1. A math cool-down at the end.**
- What a model sees in the last stretch of training, while the learning rate
  winds down, sticks disproportionately. Llama 3, MiniCPM and OLMo 2 spend
  their last phase on their best, most task-relevant data, and report clear
  gains on math tests from it.
- For us: a night that continues from the previous one (`INIT=`) with a
  math-heavy `MIX`: mostly OpenMath2, MetaMath and GSM8K, little web text, no
  TinyStories. No code: `night.sh` takes `MIX` from the environment.
- With `--init`, the schedule starts over (warmup, then decay from `LR`). A
  cool-down probably wants a lower peak `LR` than a fresh night. Pick it in
  the test below.
- It belongs at the end of the whole run of nights. The plan also makes the
  last nights ternary. Test the two changes one at a time, so we know which
  one did what.

**2. A better everyday mix.**
- Replace most of TinyStories with more math and more educational web text.
- Raise GSM8K's repeat factor. Up to about 4 passes over the same data cost
  little compared with fresh data (Muennighoff et al., 2023), and today's ×4
  gives less than one pass per night.

**How to test.**
- Two continuations from the same checkpoint, same length and `LR`: one with
  the same mix, one with the math mix. Short runs (1-2 hours) first, a night
  only if the gap is clear.
- Night 1 trained on all of GSM8K train, so judge a continuation of it on the
  GSM8K test (sparingly, recorded per the measurement rules) or on a separate
  set such as SVAMP. A model trained from scratch can use the dev set.

## The bets, in the order to try them

Prerequisites, before night 2:
- the review fixes in group A (each night must get a new data order) and B1
  (the GSM8K score must accept "1,080");
- the dev set above;
- an expert-balance setting that keeps every expert in use (above).

Without them, comparisons between nights are not trustworthy.

### 1. A calculator built into the engine

**Idea.**
- GSM8K solutions carry calculator notes, e.g. `48/2 = <<48/2=24>>24`. The
  GSM8K authors used them this way: while sampling, a calculator fills in the
  result.
- Our engine would do the same. When the model has written `<<48/2=`, the
  engine computes `24` exactly and writes `24>>` for it; the model then
  continues.
- Arithmetic becomes exact, at zero cost in parameters or training time. The
  model only has to learn *what* to compute.

**Where.**
- `prep-qa` already has `--keep-calc` (src/cxxgpt.cpp:345, 386). Today the
  notes are dropped by default, so the prepared data has none.
- The calculator itself goes into the generation loop (`src/generate.h`): a
  small exact evaluator for + - * / and parentheses. The curriculum tests
  already contain an expression parser to start from.

**Open questions, to decide by experiment.**
- Should training predict the result tokens inside `<<...>>`, or mask them out
  of the loss? Masking frees capacity; predicting keeps some arithmetic skill.
- Only GSM8K train has the notes; MetaMath and OpenMath2 don't. We could add
  notes automatically wherever a solution writes `a op b = c` and the
  arithmetic checks out. That gives more examples, all verified.
- The curriculum generator could emit notes too, for fast experiments.

**Keeping it from becoming a benchmark trick.**
- Always report two scores, with and without the calculator. The gap tells us
  how much of the failure was arithmetic and how much was reasoning.
- The calculator answers any arithmetic written in its notation, in any task,
  not only GSM8K-shaped problems. It is trained on notes from all our math
  sets (added automatically where the arithmetic checks out), not only on
  GSM8K's own notes.
- It must also help on the reasoning curriculum, not only on GSM8K.

**First experiment.** Run eval-reasoning on the same checkpoint with and
without the calculator. That alone measures how much of the GSM8K failure is
arithmetic. Then a curriculum A/B with notes kept versus dropped.
- Night 1's data has no notes, so its model never writes `<<`. For that model
  the calculator watches for plain arithmetic instead: when the model has
  written `16 x 7 =`, the engine writes `112`. Night 1's step-2000 exam shows
  the kind of error this removes ("16 x 7 = 108").

**Cost.** Small. **Expected effect:** the largest per hour of work.

**Then more tools, through the same hook.** The model writes an opening mark,
the engine computes something exactly and writes the result, and the model
continues. In order:
1. **Calculator** (above): `<<48/2=` → `24>>`.
2. **Named values:** `<<eggs = 16-3-4=` → `9>>`, and later `<<eggs*2=` →
   `18>>`. The engine remembers values by name, so the model cannot copy a
   number wrong from five lines up.
3. **Checker:** `<<check 9*2+4=22` → `yes>>` or `no>>`. The model can test a
   step and correct course.
4. **Mini programs:** a few lines with named values, arithmetic and a short
   loop, run by our own small interpreter. This is TinyGSM's approach
   (Microsoft, 2023: GSM8K-style problems with Python solutions), which brought
   models around 1 B parameters to strong GSM8K scores.
5. **Equation solver** (`<<solve 3x+5=20` → `x=5>>`) and **units and time**
   (`<<2h 15min in min=` → `135>>`), when the problem mix needs them.
6. **Lookup:** see bet 6 (retrieval), much later.

Steps 1, 2 and 4 are one growing interpreter, in C++ and tested like the rest
of the engine. Each tool is reported with and without it, as for the
calculator.

**The teacher: a program that writes word problems.**
- A tool only helps once the model knows when to call it and what to put in.
  Real data barely shows that: only GSM8K has calculator notes.
- So: a generator of GSM8K-style word problems, extending the curriculum in
  `src/reasoning.h`:
  - a few hundred sentence patterns ("X has N apples", "she gives away a
    third", "each box costs $P");
  - random names, objects, units and prices;
  - multi-step chains, optional distractor sentences;
  - worked solutions in GSM8K's format, with the tool calls written in.
- Every answer and every tool call is computed by the program, so all of it is
  correct. It is our own code and our own data, so no outside terms apply.
- **Weakness:** generated problems are less varied than human writing, so a
  model can learn the templates instead of the reasoning. Always mix with real
  data. Judge on problems the generator never made (SVAMP, the GSM8K test): if
  only the generator's own problems improve, the model learned the templates.
- The generator also feeds bet 3: with generated problems the right answer is
  always known.

### 2. Finer experts: more of them, each smaller

**Idea.**
- Night 1 uses 8 experts per layer (hidden size 1024), and each token picks 2.
  Split each expert in half instead: 16 experts of hidden size 512, each token
  picks 4.
- Total weights stay the same (267 M), and so does the work per token
  (4 × 512 = 2 × 1024), so GPU memory stays the same.
- A token now chooses from 1,820 combinations instead of 28, so experts can
  specialize more narrowly and be combined more flexibly. Recent large MoE
  models went this way (DeepSeek-V3: 256 small experts plus one always-on
  expert; Qwen3 MoE: 128 experts, 8 picked). At our size it is untested.
- More experts *of the same size* does not fit: 16 full-size experts add about
  226 M weights, about 2.7-3.6 GB of training memory on top of 5.6 GB.

**Where.** No code: `--experts 16 --topk 4` already halves the expert size,
because the expert hidden size defaults to the FFN size divided by top-k
(`src/config.h`, `ehidden()`). `--shared 1` adds an always-on expert.

**Watch out for.**
- Speed: smaller matrices, twice as many per step. Measure with the probe
  first.
- Balance: more experts are harder to keep evenly busy. Night 1 already has
  neglected experts (see "Expert balance" above), so fix that first or
  alongside.

**First experiment.**
- Probe speed and memory at 8×2, 16×4 and 32×8 (about 10 minutes).
- Curriculum, 3 seeds each: 8×2, 16×4, and 16×4 with `--shared 1`.

**When.** Decide before night 2: weights cannot carry over to a different
expert layout, so switching later means starting from scratch.

**Cost.** Small (GPU hours only).

### 3. Learning from its own checked solutions

**Idea.**
- The model attempts each training problem several times. Math answers can be
  checked exactly, so we keep only the solutions that reached the right answer
  and train on them.
- Repeat. Training effort goes to what the model can almost do, and the
  "teacher" (the answer check) is free.
- Related published methods: STaR (2022) and reinforcement learning with
  verifiable rewards.

**Where.**
- `generate.h` (batched sampling at temperature > 0);
- the answer check from eval-reasoning;
- a new `self-train` step that writes the kept solutions as a token file for
  the next night's mix.

**First experiment.** On the curriculum:
- baseline: N steps;
- the same budget: N/2 steps, then self-generated data for the rest.

Use 3 seeds each.

**Watch out for.**
- Lucky right answers with wrong reasoning: a verifier for the intermediate
  steps helps, and the curriculum already checks steps.
- Losing diversity: keep the original data in the mix (replay).

**Cost.** Medium. This is the direction already agreed with the owner.

### 4. Place-value position information for digits

**Idea.**
- Tell the model each digit's place (ones, tens, hundreds...) through an extra
  embedding, as in "Abacus" embeddings (2024).
- With it, small transformers learned addition that generalizes to numbers far
  longer than those seen in training.
- Our tokenizer already keeps every digit a separate token, which is the
  prerequisite.

**Where.** `src/model.h` / `src/gpu.cu` embedding, plus a per-token digit-place
index computed from the token stream. Everything is tested against the
double-precision reference, as usual.

**First experiment.** The curriculum's held-out harder levels (d=4) are exactly
the length-generalization test. But first fix review finding B4: eq d=4 is
mostly the training distribution.

**Cost.** Medium. This is architectural work of our own.

### 5. Ternary that pays off during training

**Idea.**
- The ternary forward pass is exactly an int8 × ternary product. On this GPU,
  int8 tensor cores are about 8x the TF32 rate (estimate from the Ada ratios).
- A kernel doing that product with int32 accumulate gives the same numbers
  much faster. Our efficiency lever would then mean more tokens per night, not
  only cheaper inference.

**Where.** `src/gpu.cu`; review item E8. It needs an exactness test against
the current fp32 path.

**Cost.** Large. **Effect:** speed for the ternary nights, and later for
ternary from the start, if the seed repeats show ternary costs little.

### 6. Facts outside the weights (retrieval)

**Idea.** Let the model look facts up in a text index instead of memorizing
them, so its few parameters go to reasoning.

**Cost.** Large. **When:** after 1-4, once the model is good enough to use
what it retrieves.

### 7. A hard-wired arithmetic unit inside the network (long shot)

**Idea.**
- A special layer that does exact arithmetic on numbers the model points at,
  like the arithmetic unit in a processor.
- Earlier attempts ("neural arithmetic logic units", 2018) were fragile to
  train.

**When:** only if 1 and 4 leave arithmetic as the bottleneck. A real
breakthrough if it works.

## Effort levels: more thinking for harder questions

Extra thinking time is cheap for us; extra parameters do not fit in 8 GB. Four
ways to turn more time into better answers, each with its own effort knob:

1. **Many tries and a vote.**
   - Sample the same problem N times (temperature above 0) and take the most
     common final answer. Wrong answers tend to scatter; right ones agree.
   - Knob: N, for example `--effort low/medium/high` = 1/8/32 tries.
   - No training. `src/generate.h` already samples with a temperature, in
     batches.
   - It only helps once the right answer is often the most common one. At
     night 1's 0.8% it cannot. Add it when scores reach a few percent.
   - **Status: built** (`eval-reasoning --tries N --temp T --seed S`, or
     `--effort`; README, "Many tries and a vote"). Only finished tries
     vote, each try has its own seed, and the headline is labelled with
     its effort (`(greedy)`, `(maj@8, temp 0.7, seed 1)`). Not yet run on
     a trained checkpoint: measure maj@N against greedy on the dev set
     once scores reach a few percent.
2. **Many tries and a judge.**
   - A second small model reads a solution and scores how likely it is right;
     keep its favorite of N.
   - TinyGSM (Microsoft, 2023) did this with models around 1 B parameters
     and got a large jump on GSM8K.
   - The problem-writing program (bet 1) supplies the judge's training data:
     its solutions come with known right answers, and the model's own wrong
     attempts supply the rest.
3. **Check and retry.**
   - With the checker tool (bet 1), the model tests a step and tries again on
     "no".
   - Knob: the number of retries allowed. Comes with the tool path.
4. **Thinking loops inside the model** (research bet).
   - A block of layers runs several times in a row with the same weights;
     harder problems get more loops. Memory stays the same, only time grows.
   - Train with a random number of loops, so the model works at any count
     afterwards. Knob: the loop count.
   - The "recurrent depth" approach (Huginn, 2025) showed reasoning improving
     as loops are added. Untested at our size.
   - Work: the CPU engine, the GPU and the inference engine, each tested
     against the others as usual.

**Order.**
- 1 when scores allow.
- 3 with the tool path.
- 2 together with the problem-writing program.
- 4 as a research bet. It is the most promising for "smart on a laptop",
  since it adds thinking without adding memory.

## Our own ideas

Ideas that use what this project has and most do not:
- our own engine, so any change to the model is ours to make;
- ternary MoE;
- a program that writes problems with exact answers (bet 1);
- cheap inference;
- math, where every step can be checked, not only the answer.

Each goes through the funnel (flag, tests, 3 seeds on the curriculum) before a
night. In order of cost:

| Idea | Cost | Novelty | Expected impact |
|---|---|---|---|
| A. Numbers backwards inside the model | small | medium | high for arithmetic |
| B. Planted mistakes and corrections | small, with the generator | medium | high for reasoning |
| C. Reviving dead experts by cloning | small | medium | fixes night 1's neglected experts |
| D. A generator at the edge of ability | medium | medium | faster learning |
| E. Exact step grading | medium | high in combination | high at higher effort |
| F. Thinking loops that route anew | large | high | possibly the big one |
| G. Experts that compute exactly | large | high | unknown: the moonshot |

### A. Numbers backwards inside the model

- People add from the right (carries move left). The model writes left to
  right, so it must produce the first digit of a sum before it has worked out
  the carries.
- Small transformers learn arithmetic far better when answers are written
  least-significant digit first (Lee et al., 2023, "Teaching Arithmetic to
  Small Transformers").
- **Ours:** the tokenizer reverses every digit run on the way in and back on
  the way out. Users see normal numbers; inside, the model always works in
  the easy direction. Our tokenizer never merges digits, so a number is a run
  of digit tokens and reversing it is exact.
- **Test:** the curriculum with and without reversal, 3 seeds each. Then
  combined with place-value embeddings (bet 4), since both are about digits.
- **Watch out for:** numbers in prose ("in 1999") reverse too. That is
  harmless as long as it is consistent, but check that years and the like
  still read back correctly.

### B. Planted mistakes and corrections

- The problem generator writes some solutions with a wrong step followed by a
  correction: "16 x 7 = 108. Wait: 16 x 7 = 112."
- Ye, Xu, Li and Allen-Zhu (2024, "Physics of Language Models, part 2.2")
  found that training on mistakes plus corrections makes models reason better
  than training on perfect solutions only: the model learns to notice errors.
- Our generator makes these exact and unlimited. It is also the behavior that
  "check and retry" needs at higher effort levels.
- **Test:** the curriculum with 0%, 10% and 30% of solutions carrying a
  planted mistake. Mask the wrong step out of the loss, so the model learns to
  correct a mistake but not to make it.

`gen-reasoning --mistakes P` implements this experiment for the synthetic
curriculum (all six task families). `P` is a probability in [0,1], default 0;
only accepted training problems draw from a separate mistake RNG. Thus the
questions, final answers, validation tokens and test set stay the same across
fractions with the same seed. At 0, every output is byte-identical to the old
generator. At a nonzero fraction it also writes `OUT.train.text` so the owner
can inspect the planted statements. Errors change one digit, omit a carry or
borrow, or redo a whole calculation with one operator slipped (`4+3+0` as
`4*3+0`, `20-8` as `20+8`); in `cmp`, which has no calculations, the deciding
relation is turned around (`2<3` as `2>3`) or a digit count is off by one.
Corrections use several short phrasings.

The reserved text markers `[[no-loss]]` and `[[/no-loss]]` surround exactly
the wrong statement, excluding its following punctuation and correction.
The generator, and `text2tok --loss-markers`, remove both markers before
encoding and give every enclosed token mask 0 (without the flag, `text2tok`
treats them as plain text, so web text is tokenized as before; re-tokenizing
`OUT.train.text` needs the flag). The wrong statement remains in the context,
so the next trained token begins its correction. Annotation boundaries also
split BPE encoding: no merged token crosses from a wrong span into a trained
span. Nested, unmatched and unclosed markers are errors, including spans
crossing a document separator (with `--loss-markers`). A streaming block waits for a closing marker
within the existing buffer limit. The token-file layout is unchanged.
`gen-reasoning` still masks the question; `text2tok` trains on the rest of
its input as before (including questions), and `--doc-sep` still trims each
document. Use the generator's `.train.tok` for this experiment.

Run on the laptop GPU, with the same architecture, budget and paired seeds
for 0%, 10% and 30%. These commands are an experiment plan, not measured
results; this change was tested on CPU without running the experiment:

```bash
mkdir -p data/planted runs
for p in 0 0.1 0.3; do
  ./cxxgpt gen-reasoning --out data/planted/p$p --seed 41 --mistakes "$p" \
    --chain-plan --expr-rewrite
  for seed in 1 2 3; do
    run="runs/planted-p$p-s$seed"
    ./cxxgpt-gpu train --gpu --data data/planted/p$p.train.tok \
      --val data/planted/p$p.val.tok --seq 256 --dim 256 --layers 4 --heads 4 \
      --experts 8 --topk 2 --ternary --fp-attn --muon --batch 16 --steps 2500 \
      --lr 1.5e-3 --warmup 150 --eval-every 500 --log-every 50 --seed "$seed" \
      --out "$run.ckpt" --log "$run.csv" > "$run.log" 2>&1
    ./cxxgpt eval-reasoning --ckpt "$run.ckpt" --test data/planted/p$p.test.txt \
      --show-failures > "$run.eval" 2>&1
  done
done
```

Compare clean validation loss and exact-answer accuracy per task across
paired seeds. Fixed steps, batch and sequence length give the same sampled
token budget; more correction text means fewer complete problems in that
budget. A gain here must still pass the dev-set funnel before a laptop night.

### C. Reviving dead experts by cloning busy ones

- When an expert has been starved for a long time (its bias pinned near steps
  × rate, as in layers 4 and 5 of night 1), copy the busiest expert of that
  layer into it, add a little noise, reset both biases and both experts'
  optimizer state, and let them split the work.
- The neglected expert starts from something useful instead of its random
  initial weights. Related: sparse upcycling (Komatsuzaki et al., 2022) builds
  MoE experts from copies of a dense layer, but not as a repair during
  training.
- **Test:** the 50-minute runs of the night-1 model from "Expert balance",
  with and without cloning. Read the biases afterwards.

The offline CPU command is now available; it does not run training:

```sh
./cxxgpt clone-experts --ckpt runs/night1.ckpt --out runs/night1-repaired.ckpt \
    --layer 4 --from 0 --to 2 --noise 0.01 --seed 1
# Or use the saved bias rankings in every layer:
./cxxgpt clone-experts --ckpt runs/night1.ckpt --out runs/night1-repaired.ckpt \
    --auto --noise 0.01 --seed 1
# The owner runs the later GPU experiment on the laptop:
INIT=runs/night1-repaired.ckpt RUN=night2 bash scripts/night.sh
```

Expert and layer indices start at zero. Choose either all three explicit
indices or `--auto`. The copy includes the expert's input/output matrices
(`efc`, `eproj`) and its router row. Noise is independent Gaussian noise
with standard deviation `S` times the RMS of each source slice; the default
`S=0.01` is 1% of that slice's weight scale. A zero slice stays zero;
`--noise 0` makes an exact copy. The seed controls only this noise, leaving
the checkpoint's data-sampling RNG unchanged.

`--auto` selects the lowest-bias expert as the source and every expert tied
at the largest bias as a target, independently in each layer. Source ties
use the smallest expert index; equal biases throughout a layer mean no
clone. This is a ranking heuristic, with no starvation threshold: biases
record past balancing nudges, not measured token counts. Inspect the printed
choices before using a repaired checkpoint; an unequal but healthy layer
also receives a clone. Explicit indices allow repairing only known bad
layers from the load log.

For each copied pair, both biases become the layer's original mean, and
both experts' optimizer `m` and `v` slices (including router rows) become
zero. Other weights, biases and optimizer entries keep exactly their bits.
The optimizer step, AdamW/Muon kind and sampler state are retained; a
weights-only checkpoint stays weights-only. Loading and saving use the
existing checkpoint format. `night.sh INIT=...` remains its usual warm
start, so it starts fresh optimizer state even when the repaired checkpoint
contains saved moments.

### D. A generator at the edge of ability

- The generator has knobs: number of steps, size of numbers, distractor
  sentences.
- During training, measure accuracy per knob setting and generate more
  problems where the model is right about half the time.
- Known in reinforcement learning as learning-progress curricula; rare in
  language-model training, and natural with our generator.
- **Test:** the same compute with a fixed mix against the adaptive mix, judged
  on fixed held-out levels.

### E. Exact step grading

- Lightman et al. (2023, "Let's Verify Step by Step") found that a judge
  trained on which step went wrong beats one trained only on final answers.
  Their step labels came from paid human annotators (PRM800K).
- Our generator knows the correct value of every step, so it gives exact step
  labels for free, for the problems it generates.
- Use them for the step-level judge from "Effort levels". At high effort the
  model can then try several next steps and keep the ones the judge trusts,
  like a chess engine searching moves.
- **Test:** best-of-N with an answer-only judge against a step judge, at
  equal N.

### F. Thinking loops that route anew

- Thinking loops ("Effort levels", 4) combined with MoE: on each pass through
  the shared layers, a token is routed again and can use different experts,
  like a small program running a different instruction each step.
- Add a per-token stop decision: easy tokens loop once, hard ones many times.
  Memory stays fixed; effective capacity grows with effort.
- Related: Mixture-of-Recursions (Bae et al., 2025) learns a depth per token.
  Routing to different experts on each loop, in a ternary model, we have not
  seen.
- **Work:** large. The CPU engine, the GPU and the inference engine, each
  tested against the others.

### G. Experts that compute exactly

- Add to each MoE layer an expert that is not neural but C++ code. Inside an
  arithmetic expression it computes the correct next digit from the tokens
  already written and returns that digit's embedding. The router learns when
  to trust it, through the ordinary gate gradient.
- The difference from the calculator: no special notation. The tool lives
  inside the network, and the model blends it in wherever it helps.
- It reads only tokens before the current position, so it cannot see the
  answer it is trained to predict.
- The open question is reading numbers that are not written as `a op b =`.
  Start with the calculator's notation, then widen.
- We know of no exact code experts inside an MoE router.

## Growing past 8 GB: a ladder to 1-2 B parameters

**Where night 1's 5.6 GB goes.** About 3.4 GB is fixed: weights, gradients and
optimizer state, all fp32. About 2.2 GB is activations (intermediate results
kept for the backward pass).

**The ladder, cheapest first.**

| Step | What it does | Frees on the GPU |
|---|---|---|
| 1. Recompute (activation checkpointing) | drop activations and recompute them in the backward pass | most of the 2.2 GB, for ~30% more compute |
| 2. bf16 storage | keep the GPU's working weights and gradients in 16 bits | half of those |
| 3. 8-bit optimizer state | store Adam/Muon state in 8 bits (Dettmers et al., 2021) | about ¾ of that state |
| 4. Optimizer in RAM | fp32 master weights and optimizer state in the laptop's RAM; the CPU does the update (below) | all of it |
| 5. Stream layers | all weights in RAM; each layer goes to the GPU just before it is needed | the GPU holds 1-2 layers at a time |

- Step 2 is review item E1. It first needs the norm-gradient question behind
  the failing `dense bf16` GPU check (review C2).
- With steps 4 and 5, the limit moves from the GPU's 8 GB to WSL's RAM.
  - `.wslconfig` gives WSL 16 GB today. Night 1 uses about 12 GB of it, mostly
    for the token files and the host copy of the model.
  - Raised to about 24 GB, that leaves maybe 18 GB for the master weights and
    optimizer state.
  - At 12-16 bytes per parameter (fp32 master and state) that is about
    1.1-1.5 B parameters. With 8-bit state (about 8 bytes), about 2 B.

**Why it suits our MoE.**
- The work per token depends on the active parameters, and memory on the
  total. Grow the total to 1-2 B (more and bigger experts) while each token
  still uses about 100 M, and the GPU does about the same work per step as in
  night 1.
- The new cost is moving weights in and gradients out each step: roughly
  10-12 GB at 2 B parameters. That is maybe 0.5-1 s per step on a laptop PCIe
  link, against about 4.7 s today. It is partly hidden behind compute.
- More total parameters at the same compute per token lower the loss in
  published MoE scaling studies, with diminishing returns.

**Limits.**
- Data: a 2 B model fills slowly at 0.37 B tokens per night. It is better than
  267 M, not 7× better.
- Engineering: each step takes days to weeks, and each is tested against the
  CPU engine. Streaming is the largest.
- Running it afterwards is easy: 2 B ternary parameters are about 0.5 GB.

**When.** Steps 1 and 2 pay off anyway (memory and speed). Steps 3-5 only once
we know we want a model bigger than 8 GB allows. Bets 2 and 5 are cheaper ways
to get more from the same memory.

### Step 4 in detail: the optimizer's memory in the CPU's RAM

**Why not split the computing work between CPU and GPU instead?**
- The CPU is about 20× slower at this work: roughly 1 TFLOP/s at its best,
  against the GPU's sustained 10-11.
- The two halves would have to combine their results after every step,
  about 1 GB each way.
- CPU and GPU share one cooling system, and the GPU already throttles at
  87 °C. Loading the CPU could cost more GPU speed than the CPU adds.

**What to pool instead: memory.**
- For each weight, the optimizer keeps extra numbers of its own, and it
  touches them once per step, at the very end.
- They could live in the laptop's 32 GB of RAM. The CPU would do that final
  update (AVX-512, as in our CPU engine), and the GPU would keep all the heavy
  work. This is the idea of DeepSpeed's ZeRO-Offload.
- **Gain:** roughly 1-2 GB of GPU memory, for a bigger model, more experts or
  longer text.
- **Cost:** sending about 1 GB each way per step, maybe 0.2-0.3 s of a ~4.7 s
  step, partly hideable behind the next step's work.

**Test.** The result must match the all-GPU update exactly (same numbers,
tested like every other GPU change).

## Speed: more tokens per second

A night reads about 350 M tokens (night 1: 5,352 steps of 65,536 tokens at
~15,000 tokens/s). That is roughly 9-10 trillion operations per second,
around 40% of what the GPU can do in TF32. Faster training means more
reading per night, which is what the model needs most.

**Where a step goes.** Measured on 2026-10-14 after night 1, night-1 model
and settings (4 sequences of 512 per micro-batch, 32 micro-batches),
`cxxgpt-gpu train --profile`, 4.4 s per step:

| part | share |
|---|---|
| matrix products (forward + backward) | 47% |
| attention (forward + backward, cuBLAS) | 22% |
| Muon | 10% |
| output layer and loss | 7% |
| MoE routing (including the per-layer host sync) | 6% |
| norms, RoPE, adds, SwiGLU | 7% |

**Settings tried** (10 steps each, same model):

| setting | tokens/s | GPU memory |
|---|---|---|
| night 1 as run: TF32, 4 x 32 | 14,900 | 5.6 GB |
| `--matmul bf16`, 4 x 32 | 15,050 | 5.6 GB |
| `--fused-attn`, 4 x 32 | 7,700-9,500 | 5.0 GB |
| `--fused-attn`, 8 x 16 | 10,200 | 6.7 GB |
| `--fused-attn --matmul bf16`, 8 x 16 | 11,500 | 6.7 GB |
| cuBLAS attention, 8 x 16 | 2,900 (spills out of GPU memory) | 7.9 of 8.2 GB |

What this says:
- `--matmul bf16` only asks cuBLAS to round fp32 inputs internally
  (`CUBLAS_COMPUTE_32F_FAST_16BF`); the data stays fp32 in memory, and it
  gains nothing. (Measured later: the products are not limited by moving
  data. TF32 runs them at 13-21 TFLOPS, near this GPU's TF32 ceiling, and
  `FAST_16BF` simply never gets the bf16 rate; inputs stored in bf16 do.)
  Storing weights and activations in bf16 is the real version: the bf16
  rung of the ladder above.
- Our fused attention is slow (plain fp32 loops, no tensor cores) but uses
  less memory.
- Bigger micro-batches pay: with the same attention, 8 sequences instead of
  4 is 33% faster, and bf16 starts to help (+12%) at that size.
- 8 sequences do not fit with the cuBLAS attention: the Windows driver
  spills into system RAM and the step is 5x slower. That is why the probe
  picks 4.

**Order of work.**
1. **A fast fused attention kernel** on the tensor cores (the
   FlashAttention design: tiles of queries and keys in shared memory, the
   softmax computed on the fly, no stored attention matrix). It removes
   most of the 22% and the memory that blocks 8-sequence micro-batches.
   Tested against the CPU engine like the current kernels. Estimate:
   1.3-1.6x overall, more with bf16 storage on top.
   **Done** (`src/attention_tc.cuh`, the default since 2026-10-14),
   measured on the same model:

   | setting | tokens/s | attention (fwd + bwd) | GPU memory |
   |---|---|---|---|
   | cuBLAS attention, 4 x 32 (night 1) | 15,000 | 343 + 589 ms | 5.6 GB |
   | tensor-core fused, 4 x 32 | 16,900 | 91 + 287 ms | 5.0 GB |
   | tensor-core fused, 8 x 16 | 17,500 | 90 + 282 ms | 6.7 GB |

   The probe (20 steps) measured 17,760 tokens/s at 8 x 16 for night 1's
   model, 0.48 B tokens a night against 0.40 B, and 17,000 at 4 x 32 for
   its ternary version. 1.13-1.18x, short of the estimate: attention was
   22% of the step, so
   even free attention would give 1.27x, and the 8-sequence micro-batch
   helps less than it did with the slow kernels (+4%, against +33%). The
   kernels run at about 7 TFLOPS, roughly a third of the tensor cores'
   TF32 rate, so there is still room (bigger tiles, the backward pass in
   one kernel with atomic adds, raw `mma` instructions), but attention is
   now under 10% of the step and the matrix products are 50%: the next
   gains are in items 2, 3 and 5.
2. **bf16 storage** for weights and activations (the ladder's bf16 rung,
   after the norm-gradient question in the review). **Done** as
   `--matmul bf16` (2026-10-14), for the matrix products' inputs: a bf16
   copy of the weights, refreshed after each step (7 ms), and activations
   written in bf16 by the kernels that make them; weights, gradients,
   optimizer state, the router and attention stay fp32. cuBLAS on
   the night-1 shapes, one 65,536-token step's worth of products:

   | inputs | time |
   |---|---|
   | fp32, TF32 (default) | 2.47 s |
   | fp32, `FAST_16BF` (the old `--matmul bf16`) | 2.43 s |
   | bf16, fp32 sums | 1.22 s |

   In training (night-1 model, 8 x 16): 24,200 tokens/s against 17,600
   (1.37x), the step 2.7 s against 3.7 s, GPU memory 7.0 GB against 6.7.
   Converting at each product instead of writing bf16 at the source gave
   only 20,800 and 7.4 GB. Not for ternary models yet (their quantized
   values are not exact in bf16; int8 tensor cores are the way there).
3. **Muon in bf16** (16% of the step after item 2). **Done** with
   `--matmul bf16` (2026-10-14): Newton�Schulz's products take bf16 copies
   of X and A, with fp32 sums, X kept in fp32, and cuBLAS adding the b A and
   a X terms in. Muon went from 434 to 288 ms a step; with item 5's first
   step, the night-1 model trains at 25,600 tokens/s (2.55 s a step, steady
   over a 200-step run). Reference Muon runs Newton�Schulz all in bf16.
4. **Exams on the GPU.** Each one runs on the CPU while the GPU waits: about
   5 minutes for 250 problems, about 15 minutes per night with the final
   exam. Run them on the GPU, or less often.
5. **No host sync in MoE routing:** fixed-capacity expert slots and one
   batched product for all experts, which also allows CUDA graphs. First
   step done (2026-10-14): the counts come back asynchronously into pinned
   memory while the GPU already groups the rows, and the row offsets are
   computed on the GPU. That saved only about 20 ms of the 236 ms "moe
   routing" a step, because under WSL each wait for the GPU costs about
   0.5 ms whatever is queued. The rest is about 70 ms of forward kernels
   and 107 ms of backward ones, all limited by memory bandwidth.
6. The review's smaller items: the attention softmax per sequence, no
   memsets in front of accumulations.

**Size nights by the real speed.** `night.sh` sizes a night from the probe's
tokens/s with a 10% margin, but night 1 lost about 30 minutes to the exams,
the validation passes and slow stretches, and stopped at step 5,352 of 5,700
(the learning rate was 99% of the way down, so little was lost). Pass the
speed measured on the previous night as `TOKS`.
