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
- About 500 problems taken out of GSM8K train with a fixed seed, kept in a
  file, and never trained on. `night.sh` uses them for `--val` and
  `--reason-test`.
- MetaMath and OpenMath2 are built from GSM8K train problems, so taking the
  500 out of `gsm8k/train` alone is not enough:
  - MetaMathQA rows carry `original_question`; drop the rows whose original is
    a dev problem.
  - OpenMath2 rows with `problem_source` `gsm8k` repeat the problem text; drop
    exact matches. Rows from `augmented_gsm8k` are new problems with no link
    back to their source, so a small leak remains. Write it down.
- `scripts/get_data.py` reads neither `original_question` nor matches problem
  texts today, so this needs a small change there and in
  `scripts/prepare_data.sh`, then a fresh data preparation.
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

## Growing past 8 GB: the optimizer's memory in the CPU's RAM

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

**When.** Only once we know we want a model bigger than 8 GB allows. Bets 2
and 5 are cheaper ways to get more from the same memory.

## Speed, in parallel

The 2026-10-14 code review (kept locally by the owner) ranks GPU speed items.
The cheapest to try:

- run the attention softmax per sequence;
- drop memsets in front of accumulations;
- Muon in bf16.

The exams cost time too. Each one runs on the CPU while the GPU waits: 5.5
minutes for 250 problems at step 2000 of night 1, so about 15 minutes per
night with the final exam. Run them on the GPU, or less often.

Do the zero-code check first: `GPU="--gpu --matmul bf16" scripts/probe.sh`. It
measures what bf16 is actually worth here.
