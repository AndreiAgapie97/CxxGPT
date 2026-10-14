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

## The bets, in the order to try them

Prerequisite: the review fixes in group A (each night must get a new data
order) and B1 (the GSM8K score must accept "1,080"). Without them, comparisons
between nights are not trustworthy.

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

**First experiment.** Run eval-reasoning on the same checkpoint with and
without the calculator. That alone measures how much of the GSM8K failure is
arithmetic. Then a curriculum A/B with notes kept versus dropped.

**Cost.** Small. **Expected effect:** the largest per hour of work.

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
- Balance: more experts are harder to keep evenly busy (night 1's load is
  around 2-2.5, against an ideal 1.0). A stronger `--bias-rate` may be needed.

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

Do the zero-code check first: `GPU="--gpu --matmul bf16" scripts/probe.sh`. It
measures what bf16 is actually worth here.
