# CxxGPT

A native C++20 language-model research engine, aiming at strong reasoning
and more efficient training and inference:

- **ternary weights** (each projection weight is −1, 0 or +1) with int8
  activations, as in BitNet b1.58;
- **sparse Mixture-of-Experts**: many expert FFNs, each token uses a few;
- **hand-written kernels**, CPU (AVX2, AVX-512, VNNI) and CUDA, with no
  Python framework anywhere in training or inference.

Everything is ours except the training data: the model, the training
engine and optimizers, the BPE tokenizer, the data pipeline, the reasoning
curriculum and the inference engine. Python appears only in one small
script that downloads public datasets.

Ternary projection inference stores two bitplanes (two bits per weight) plus
scales; `log2(3)` is the information bound. Embeddings, norms and routers retain
fp32 weights. Training retains fp32 master weights, gradients, quantized
copies and optimizer moments, so packed inference storage does not describe
training memory.

## Where things stand

| part | state |
|---|---|
| training engine (CPU) | transformer with RoPE, RMSNorm, SwiGLU, tied embeddings; ternary BitLinear; top-k MoE with shared experts and bias-based load balancing; AdamW and Muon; checkpoints and resume |
| training engine (GPU) | the same model in CUDA + cuBLAS, op for op; compiled in CI, tested against the CPU on the laptop's RTX 4070; fused attention on the tensor cores by default (`--no-fused-attn` for cuBLAS) |
| tokenizer | byte-level BPE, trained by our own code; parallel streaming encoder |
| data | synthetic reasoning curriculum; GSM8K and other public math and text sets (`scripts/get_data.py`) |
| evaluation | exact-answer accuracy on held-out problems, per task and difficulty |
| inference | ternary models packed into bitplanes, run on the int8 kernels with a KV cache; the output layer and full-precision attention (`--fp-attn`) as int8 rows |

The models trained so far are tiny (3–10 M parameters, CPU only). The
laptop's GPU is next: see [docs/LAPTOP.md](docs/LAPTOP.md).

## Build and test

```sh
make            # ternary_moe, test_engine, cxxgpt, test_training_cli (g++ -O3 -march=native)
make test       # every CPU test
make asan tsan  # the same under AddressSanitizer+UBSan and ThreadSanitizer
make gpu        # cxxgpt-gpu and test_gpu (needs the CUDA toolkit)
make gpu-test   # GPU against CPU (needs an NVIDIA GPU)
```

C++20; builds warning-free under g++ and clang++. CI builds with both
compilers on several `-march` targets, runs every test under the
sanitizers, and compiles the CUDA backend with nvcc 12.8 and 13.0.

## Quick start

Train a small model on the reasoning curriculum, then test it on held-out
problems:

```sh
./cxxgpt gen-reasoning --out data/reason          # train/val tokens and a test set
./cxxgpt train --data data/reason.train.tok --val data/reason.val.tok \
    --experts 8 --topk 2 --ternary --steps 2500 --lr 1.5e-3 --out runs/model.ckpt
./cxxgpt eval-reasoning --ckpt runs/model.ckpt --test data/reason.test.txt
./cxxgpt sample --ckpt runs/model.ckpt --temp 0 --answer --prompt $'Q: 47+85\nA: '
```

`cxxgpt <command> --help` lists a command's options and their defaults;
`cxxgpt train --help` also explains each. Useful ones: `--muon` (the Muon
optimizer), `--gpu` (with `cxxgpt-gpu`), `--time-limit MINUTES` (stop and
save in time), `--resume`, `--reason-test FILE --reason-every N` (accuracy
during training).

### Commands

| command | does |
|---|---|
| `gen-reasoning` | writes the synthetic curriculum: problems with step-by-step solutions, train/val token files and a test set one difficulty level beyond training |
| `gen-word` | writes exact word problems, worked solutions, and separate train/test sentence patterns |
| `bpe-train` | trains a BPE tokenizer on a text file |
| `text2tok` | encodes text into a token file (parallel, streaming; `--doc-sep` splits documents) |
| `prep-qa` | turns question/answer JSONL (GSM8K and the like) into token files with a loss mask and a test set |
| `train` | trains or resumes a model (CPU, or GPU with `cxxgpt-gpu --gpu`) |
| `eval-reasoning` | exact-answer accuracy per task and difficulty; greedy by default, or a vote over several sampled tries (below) |
| `sample` | generates text from a prompt |
| `bench-infer` | times the ternary inference engine against an fp32 baseline |

### Many tries and a vote

```sh
./cxxgpt eval-reasoning --ckpt model.ckpt --test data/word/test.txt --tries 8 --temp 0.7 --seed 1
```

- `--tries N` samples every problem N times at temperature `--temp`
  (above 0) and takes the most common final answer; `--effort
  low/medium/high` is a shorthand for 1/8/32 tries (`--tries` wins).
- With more than one try, the headline `overall` score is that vote
  (maj@N), and an effort report follows: per level, the first sampled
  try's accuracy, the vote's, any try's (pass@N), and the mean number of
  distinct answers.
- Only tries that finish their `#### answer` line vote; a problem where no
  try finished is wrong. With one try at `--temp 0` (the default) the score
  is exactly the greedy one.
- Every headline names its effort: `(greedy)`, `(1 sampled try, temp 0.8,
  seed 1)` or `(maj@8, temp 0.7, seed 1)`. Scores at different efforts
  are different tests.
- Each try is seeded from (`--seed`, the problem's place in the test file,
  the try's number), so a try gives the same text whatever `--batch` and
  `--limit` are, and try k is the same at every effort level. All of one
  problem's tries are generated together, so `--tries` must not exceed
  `--batch` (64 by default).

### Generated word problems

```sh
./cxxgpt gen-word --out data/word/train --split train --seed 1 --count 10000
./cxxgpt gen-word --out data/word/test --split test --seed 2 --count 1000
./cxxgpt text2tok --in data/word/train.text --out data/word/train.tok --doc-sep '<|endoftext|>'
# Add --bpe tok.bpe to text2tok when using a learned tokenizer.
./cxxgpt eval-reasoning --ckpt model.ckpt --test data/word/test.txt
```

`gen-word` writes two files from the same problems: `PREFIX.txt` is the
existing five-field problem format read by `eval-reasoning`; `PREFIX.text`
is readable Q/A text, with `####` before each final answer and
`<|endoftext|>` between problems, for `text2tok`. Tokenizing the Q/A text
trains on the whole document; it does not create a question loss mask.

Options (see `gen-word --help`):

- `--seed 1`, `--count 1000`, `--out data/word`: reproducible generation and
  output prefix. Zero count writes empty files.
- `--min-steps 2 --max-steps 8`: inclusive range of arithmetic operations.
  Set both to the same number for a fixed length. A percentage takes two
  operations: multiply by the percentage, then divide by 100.
- `--distractors on|off`: include or omit a sentence with an irrelevant
  number (default on).
- `--notes on|off`: include or omit GSM8K calculator notes such as
  `<<48/2=24>>24` (default on). This changes only the worked solution.
- `--split train|test`: 48 training sentence patterns or 24 held-out test
  patterns, grouped into 12 arithmetic situations. The opening patterns
  differ too, so the splits cannot share a complete question, even with
  different seeds. Each file also omits duplicate questions.

The situations include boxes of objects, hourly production and wages,
liquid volumes, gifts, donations, equal shares, fractions, percentages,
selling, buying, and liters converted to milliliters. Names, objects,
prices, and quantities vary. Objects must remain whole; money and volume
may use exact hundredths. No floating-point arithmetic or rounding is used.
An attempt is discarded if any intermediate value is negative, exceeds
1,000,000, needs more than two decimal places, or gives fractional objects.
The step count stays fixed while invalid attempts are retried.

This first version uses short, sequential stories and unit fractions
(`1/2`, `1/3`, `1/4`, `1/5`). It is less varied than human-written math.
Success on its held-out wording alone is not evidence of improvement on
GSM8K or SVAMP. No generated datasets are checked into the repository, and
this command does not change the night data mix.

### Real data and GPU training

```sh
scripts/prepare_data.sh              # download, train our tokenizer, write token files
scripts/probe.sh                     # GPU speed and memory per model size, and the
                                     #   night.sh command for each
TOKS=... MODEL=... scripts/night.sh  # one night of training, checked on GSM8K
scripts/report.sh night1             # the night on one screen
```

`prepare_data.sh` fetches GSM8K, TinyStories, FineWeb-Edu,
OpenMathInstruct-2 and MetaMathQA (`scripts/get_data.py`, which needs only
`pyarrow`), trains a 16k-token BPE on a sample, and encodes everything.
[docs/LAPTOP.md](docs/LAPTOP.md) has the whole setup, from a fresh Windows
laptop to a night of training.

## Reproducible research

```sh
scripts/research.sh runs/toy-01 200 3 3 500
scripts/reasoning-baseline.sh runs/reason-01 200 3 3
```

The first compares four variants on a tiny transition environment, a continual
task switch with bounded replay, and a contextual bandit. The second trains
and evaluates the reasoning curriculum. Run directories must be new; seeds,
source hashes, flags and raw results are preserved. See [the research log](docs/research.md)
for gradient contracts, numerical corrections, measured results and limits.
The research CUDA stable-loss and masked-accumulation tests require
`make gpu-test` on NVIDIA hardware before the research PR leaves draft.

## Code

| file | contents |
|---|---|
| `src/model.h` | the model: layout, init, forward, backward, KV-cache decoding |
| `src/ops.h` | every layer op, forward and backward |
| `src/gemm.h` | packed, threaded fp32 matrix multiply |
| `src/train.h` | AdamW, Muon (Newton–Schulz), clipping, schedule, token data, checkpoints |
| `src/config.h` | model shape and parameter layout, shared with the GPU code |
| `src/gpu.cu`, `src/gpu.h` | the CUDA trainer |
| `src/attention.cuh` | fused attention kernels for the GPU in fp32 (tested on a CPU emulator, `tests/cuda_emu.h`) |
| `src/attention_tc.cuh` | the same on the tensor cores, for TF32/bf16 training (tested on the GPU) |
| `src/tokenizer.h` | byte-level BPE |
| `src/reasoning.h` | the reasoning curriculum and its evaluation |
| `src/ternary.h`, `src/infer.h` | bitplane int8 kernels and the ternary inference engine |
| `src/thread_pool.h` | the thread pool everything runs on |
| `src/cxxgpt.cpp` | the command-line tool |
| `ternary_moe.cpp` | kernel tests and benchmarks for one ternary MoE layer |
| `tests/` | tests for all of the above |

## Results so far

### Architecture comparison

The following larger-model results are reported by Claude's `main` at
`9130fbe`; this research run has not reproduced them. Our smaller, independently
recorded multi-seed probes are in [docs/research.md](docs/research.md).

Four variants with the same width (dim 256, 4 layers, 4 heads), data,
steps and learning rate: 2500 steps of 4096 tokens on the reasoning
curriculum, CPU. The MoE variants have 8 experts (top-2) with the same
active parameters per token as the dense ones. Accuracy is exact-answer on
960 held-out question strings: 6 tasks × difficulties 1–4. The d=4 label
is above the trained labels, but does not uniformly mean new mathematical support.

| variant | params (active) | val loss | accuracy | d = 1–3 | d = 4 |
|---|---:|---:|---:|---:|---:|
| dense | 3.3 M | 0.1136 | 59.4% | 78.5% | 2.1% |
| dense, ternary | 3.3 M | 0.1152 | 57.0% | 74.2% | 5.4% |
| MoE | 9.8 M (3.3 M) | **0.0986** | **62.0%** | **81.0%** | 5.0% |
| MoE, ternary | 9.8 M (3.3 M) | 0.1116 | 55.2% | 72.4% | 3.8% |

**Equation limit:** eq caps its additive constant at 999 at both d=3 and
d=4. The unknown ranges from 0–30 and 0–39, respectively, so much of eq
d=4 uses the earlier operand domain. Aggregate d=4 scores are therefore
not a clean measure of extrapolation. The existing generator and recorded
results are preserved; the evaluation table already prints eq separately.

**Split limit:** separation excludes exact question strings, not mathematical
equivalents or shared templates. For example, `12+34` and `34+12`, or the
same comparison with swapped operands, can cross splits. Trained-level
accuracy may include memorization. No commutative deduplication is added
here because changing the generator would invalidate comparisons with the
recorded data. New independent protocols should declare their exclusion key.

One seed each. A binomial standard error of about 1.6 points assumes
independent problems; it does not measure seed variance or correlations
between related synthetic problems.

- **MoE helps:** at the same compute per token, its validation loss is
  13% lower than dense.
- **With AdamW, ternary costs dense little** (0.1136 → 0.1152) but costs
  MoE much more (0.0986 → 0.1116), which erases most of MoE's gain. With
  Muon, the better-trained models show a similar cost for both (below).
- **These aggregate d=4 results do not establish length generalization.**
  Equation support overlaps training; inspect the per-task evaluation
  rather than interpreting the whole column as unseen-length accuracy.
- **Chain** (multi-hop variable lookup) is the hardest task. Most
  failures are a wrong first step: the solution resolves variables from
  the given value toward the question, so the model has to pick the path
  to the queried variable before it has written anything about it.
  Writing the lookups first fixes most of that (below).

### Muon

The same four variants with Muon for the matrices inside the blocks
(attention and FFN projections, each expert on its own); embeddings,
norms and routers stay on AdamW. Same learning rate and steps.

| variant | val loss, AdamW | val loss, Muon | accuracy, AdamW | accuracy, Muon | d = 4, Muon |
|---|---:|---:|---:|---:|---:|
| dense | 0.1136 | 0.0894 | 59.4% | 64.9% | 5.0% |
| dense, ternary | 0.1152 | 0.0988 | 57.0% | 62.7% | 9.2% |
| MoE | 0.0986 | **0.0863** | 62.0% | **65.9%** | 9.2% |
| MoE, ternary | 0.1116 | 0.0974 | 55.2% | 61.1% | 5.0% |

- **Muon helps every variant:** validation loss 12–21% lower, accuracy
  4–6 points higher. The best run so far is the full-precision MoE with
  Muon.
- **Ternary weights cost 10–13% of validation loss** once training is
  good (with Muon: dense 0.0894 → 0.0988, MoE 0.0863 → 0.0974). With
  AdamW, dense seemed to pay almost nothing; the weaker training hid it.
  BitNet-style work reports that the gap shrinks as models grow (not
  tested here yet), and that training in full precision first and
  switching to ternary at the end beats ternary throughout, which holds
  here (below). A ternary MoE with Muon does match a full-precision MoE
  with AdamW (0.0974 vs 0.0986).
- **MoE's gain shrinks with Muon:** 13% lower validation loss than dense
  with AdamW, 3.5% with Muon (1.4% when both are ternary). These MoE
  models have three times the parameters for the same compute per token;
  on 10 M tokens of curriculum, the better optimizer lets dense catch up
  most of the way. Whether MoE pays off with far more data per run is a
  question for the GPU nights.
- **Its cost** is the Newton–Schulz iterations on every matrix, which
  grow with the number of parameters, not of tokens: here (4096 tokens
  per step) 10% of the step time on dense and 15% on ternary MoE, whose
  experts all go through it every step. For the laptop's d768 MoE
  at 65536 tokens per step it comes to about 13% of the arithmetic;
  `scripts/probe.sh` measures the speed with Muon on.
- On ternary MoE, Muon starts slower (validation 0.50 vs 0.43 at step
  500) and is ahead from step 1000 on (0.154 vs 0.208).
- `scripts/night.sh` uses Muon by default.

### Learning rate and weight decay

Ternary MoE, otherwise the same runs:

| optimizer | lr | val loss | accuracy | d = 1–3 | d = 4 |
|---|---:|---:|---:|---:|---:|
| AdamW | 1.5e-3 | 0.1116 | 55.2% | 72.4% | 3.8% |
| AdamW | 3e-3 | 0.1063 | 60.0% | 78.6% | 4.2% |
| Muon | 1.5e-3 | 0.0974 | 61.1% | 79.9% | 5.0% |
| Muon | 3e-3 | **0.0956** | **61.7%** | **80.7%** | 4.6% |
| AdamW, weight decay 0 (not 0.1) | 1.5e-3 | 0.1120 | 53.9% | 70.8% | 2.9% |

- **A higher learning rate helps both**, AdamW more (5% lower validation
  loss, against 2% for Muon). At the same learning rate Muon stays well
  ahead: 10% lower validation loss at 3e-3, 13% at 1.5e-3.
  `scripts/night.sh` keeps 1.5e-3 for now: the best learning rate
  usually drops as models get wider, and a diverged run costs a night.
- **Weight decay makes no measurable difference.**

### Chain: look the variables up first

`gen-reasoning --chain-plan` writes chain solutions in two passes: first
the lookups, from the queried variable back to the given value (copying
each definition), then the arithmetic forward. Same problems, same test
set (validation losses are on different text, so not comparable):

| variant | chain, plain | chain, plan first | all tasks, plain | all tasks, plan first |
|---|---:|---:|---:|---:|
| dense, Muon | 35.0% | **78.8%** | 64.9% | **70.0%** |
| MoE, ternary, Muon | 21.2% | 35.0% | 61.1% | 62.4% |

- **Planning first more than doubles chain accuracy on dense:** 100% at
  difficulties 1–2, 97.5% at 3, and 17.5% at 4 (harder than training,
  where it mostly stops one lookup short). The model no longer has to
  guess the path before writing anything; it finds it one lookup at a
  time. How a solution is written matters as much as the model.
- **The ternary MoE gains much less.** Its errors are copying errors: it
  writes the definition of another variable with the same source
  (`f=z+5` where the problem says `f=z-1`). Precise copying is the job of
  attention, but ternary attention is not the cause: the MoE in full
  precision does as badly ("Ternary FFN, full-precision attention").

### Full precision first, ternary at the end

The laptop trains in nights, and each night starts a new learning-rate
schedule from the previous night's weights (`night.sh` with `INIT=`).
These runs do the same after 1700 of 2500 steps: MoE with Muon, the last
800 steps either kept as they were or switched to ternary
(`train --init FILE --ternary`).

| first 1700 steps | last 800 steps | val loss | accuracy | d = 1–3 | d = 4 |
|---|---|---:|---:|---:|---:|
| full precision | full precision | **0.0940** | **63.1%** | **81.5%** | **7.9%** |
| full precision | ternary | 0.0982 | 60.9% | 78.9% | 7.1% |
| ternary | ternary | 0.1033 | 58.6% | 76.8% | 4.2% |

- **Switching to ternary for the last third halves its cost:** 4.5%
  higher validation loss than staying in full precision, against 9.9%
  for ternary throughout. Right after the switch the training loss jumps
  (1.9 at its 50th step), and it is back level with the others by step
  200.
- **So the laptop's first nights train in full precision**, and the last
  ones switch to ternary (`INIT=... TERNARY=1`, see
  [docs/LAPTOP.md](docs/LAPTOP.md)). Full precision also leaves the GPU
  about 1 GB more at d768: no ternary copy of the weights.
- Restarting the schedule costs something by itself (0.0940 here,
  against 0.0863 for one 2500-step schedule); every night pays that,
  ternary or not.

### Ternary FFN, full-precision attention

`train --ternary --fp-attn` makes only the FFN ternary (the experts, most
of the weights); the attention projections stay in full precision. Same
curriculum as the chain section (lookups first), Muon, 2500 steps, one
run each:

| model | val loss | accuracy | chain | d = 1–3 | d = 4 |
|---|---:|---:|---:|---:|---:|
| MoE, ternary FFN, full-precision attention | **0.0786** | **71.9%** | 77.5% | **93.6%** | **6.7%** |
| MoE, full precision | 0.0897 | 64.2% | 36.2% | 83.8% | 5.4% |
| MoE, all ternary | 0.0957 | 62.4% | 35.0% | 81.9% | 3.8% |
| dense, full precision | 0.0788 | 70.0% | **78.8%** | 91.5% | 5.4% |

- **Ternary attention is not what hurts copying:** the MoE in full
  precision fails chain as badly as the all-ternary one. The difference
  is when chain "clicks". Its accuracy at steps 500 to 2500 (the checks
  during training, on 240 of the 960 test problems):

  | model | 500 | 1000 | 1500 | 2000 | 2500 |
  |---|---:|---:|---:|---:|---:|
  | MoE, ternary FFN, fp attention | 12.5% | 52.5% | 70.0% | 75.0% | 77.5% |
  | dense, full precision | 12.5% | 5.0% | 42.5% | 77.5% | 80.0% |
  | MoE, full precision | 17.5% | 15.0% | 15.0% | 30.0% | 42.5% |
  | MoE, all ternary | 5.0% | 10.0% | 25.0% | 25.0% | 35.0% |

  The lookup skill appears suddenly, at a different step in each run, and
  the two weak MoEs are still climbing when training stops. One run per
  model cannot separate that timing from the model itself; several seeds
  each can (minutes per run on the laptop GPU, 40 on this VM).
- **For now attention stays in full precision**, the best run so far:
  `night.sh` `TERNARY=1` trains `--ternary --fp-attn`, `TERNARY=all`
  everything ternary. The seed runs decide before the ternary nights.
- **Training costs the same.** Attention is a small part of the model:
  at the laptop's night-1 size (d768, 12 layers, 8 experts, top-2), 28 M
  of 267 M weights, 28 M of the 98 M a token uses.
- **The inference engine stores these weights as int8 rows**, like the
  output layer (next sections). That costs no accuracy: evaluated as the
  engine runs it (`eval-reasoning --as-engine`), 71.7% against 71.9%,
  and 62.5% against 62.4% for the all-ternary model.
- **It costs speed and memory at inference:** at the night-1 size, 98 MB
  of weights instead of 77 MB, and 494 tokens/s instead of 682 (4
  threads, medians of 3 interleaved rounds).

### Inference kernels

`ternary_moe.cpp`, on a 4-vCPU Xeon VM with AVX-512 + VNNI (these shared
VMs are noisy; expect ±20% between runs). GEMV, M = K = 4096, 60% zero
weights; medians in ms. The baseline is a vectorized fp32 dot product.

| kernel | 1 thread | vs fp32 | 4 threads | vs fp32 |
|---|---:|---:|---:|---:|
| fp32 dense | 2.62 | 1.0× | 0.88 | 1.0× |
| int8 AVX2 | 0.85 | 3.1× | 0.28 | 3.1× |
| int8 AVX-512 | 0.57 | 4.6× | 0.13 | 6.7× |
| int8 VNNI 512-bit | 0.32 | 8.3× | 0.12 | 7.6× |

A 4096×4096 ternary matrix takes 4 MB instead of 64 MB in fp32. One MoE
layer (d_model 512, 8 experts, top-2) decodes a token in about 56 µs on 4
threads.

### Whole-model decoding

`cxxgpt bench-infer`, same VM: one sequence, 128 tokens with a KV cache,
median of 9 runs. The baseline is the same model on our vectorized fp32
decoding path, with the ternary weights as plain floats.

| model | threads | fp32 | ternary | speedup | weights |
|---|---:|---:|---:|---:|---|
| ternary MoE from the comparison (9.8 M, 3.3 M active) | 4 | 3431 tok/s | 9806 tok/s | 2.9× | 39 MB → 2.7 MB |
| | 1 | 1511 tok/s | 7601 tok/s | 5.0× | |
| MoE, d1024, 8 layers, random weights (304 M, 102 M active) | 4 | 120 tok/s | 759 tok/s | 6.3× | 1218 MB → 77 MB |
| | 1 | 30.5 tok/s | 328 tok/s | 10.7× | |

The bigger the model, the bigger the gain: fp32 decoding has to read
every active weight from memory for every token, and ternary weights are
16 times smaller. The engine picks the same next token as the model
computes in training on 98–100% of positions; the rest are int8 rounding
flips between near-equal candidates.

**The output layer is stored as int8 rows** (one scale per row; it is
also the embedding). At a 16k-token vocabulary it was the largest read
of every step: at the night-1 size (d768, 12 layers, 8 experts, top-2;
267 M weights, 98 M active), 50 MB in fp32 against about 21 MB for all
the ternary weights a token uses. In int8 it is 12.6 MB. Same VM, random
weights, 4 threads, 48 tokens, medians of 3 interleaved rounds:

| output layer | weights | ternary decoding |
|---|---:|---:|
| fp32 | 114 MB | 513 tok/s |
| int8 rows | 77 MB | 682 tok/s |

Accuracy is unchanged ("Ternary FFN, full-precision attention"). The
table above was measured with the output layer in fp32; at its 256-token
vocabulary that layer is under 1 MB.

## How the kernels work

- **Weights: two bitplanes.** Each row is stored as two bitmasks, `pos`
  (weight +1) and `neg` (weight −1), 32 weights per `uint32_t`, plus one
  scale per matrix.
- **Activations: int8,** quantized symmetrically to [−127, 127] with one
  scale per token.
- **Dot products.** The masks select `x`, `−x` or `0` per byte, then a sum:
  `maddubs` + `madd` on AVX2, one `vpdpbusd` with VNNI, spread over 2–4
  accumulators to break the dependency chain.
- **Threading.** A persistent pool with spin-then-sleep workers claiming
  row chunks from one atomic cursor. An empty `parallel_for` costs about
  3 µs on 4 threads.

## Next

- Train on the laptop GPU: language first, then worked math solutions,
  checked on the GSM8K test set ([docs/LAPTOP.md](docs/LAPTOP.md)).
- Ternary's cost at larger sizes, and how late the switch to ternary
  can come.
- Several seeds per model on the curriculum (minutes each on the laptop
  GPU): full-precision attention, and for length generalization layers
  without RoPE (`--nope-every`) and rewritten expressions
  (`gen-reasoning --expr-rewrite`).
- Faster GPU training: fused attention on the tensor cores is the default
  (1.18x); bf16 inputs to the products (`--matmul bf16`) give another
  1.45x but are opt-in. On the laptop two GPU checks still miss, both on
  a norm weight's gradient (docs/LAPTOP.md, "GPU options"),
  and `--profile` says what to speed up next (docs/research-plan.md,
  "Speed").
- Benchmark the inference engine against bitnet.cpp at equal model size.

### CPU build portability

The default `-march=native` binary targets the build machine's CPU. Rebuild
on the destination machine, or use `make MARCH=x86-64` for a portable baseline
within x86-64 on the same OS/ABI; compile-time kernels are selected accordingly. Runtime ISA
dispatch and cross-architecture binary portability are not implemented.
