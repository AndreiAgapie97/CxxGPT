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
| training engine (GPU) | the same model in CUDA + cuBLAS, op for op; CPU parity suite; research CUDA changes await NVIDIA runtime validation |
| tokenizer | byte-level BPE, trained by our own code; parallel streaming encoder |
| data | synthetic reasoning curriculum; GSM8K and other public math and text sets (`scripts/get_data.py`) |
| evaluation | exact-answer accuracy on held-out problems, per task and difficulty |
| inference | ternary models packed into bitplanes, run on the int8 kernels with a KV cache |

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
| `bpe-train` | trains a BPE tokenizer on a text file |
| `text2tok` | encodes text into a token file (parallel, streaming; `--doc-sep` splits documents) |
| `prep-qa` | turns question/answer JSONL (GSM8K and the like) into token files with a loss mask and a test set |
| `train` | trains or resumes a model (CPU, or GPU with `cxxgpt-gpu --gpu`) |
| `eval-reasoning` | exact-answer accuracy per task and difficulty |
| `sample` | generates text from a prompt |
| `bench-infer` | times the ternary inference engine against an fp32 baseline |

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
960 held-out problems: 6 tasks × difficulties 1–4, where 4 is one level
harder than anything in training.

| variant | params (active) | val loss | accuracy | d = 1–3 | d = 4 |
|---|---:|---:|---:|---:|---:|
| dense | 3.3 M | 0.1136 | 59.4% | 78.5% | 2.1% |
| dense, ternary | 3.3 M | 0.1152 | 57.0% | 74.2% | 5.4% |
| MoE | 9.8 M (3.3 M) | **0.0986** | **62.0%** | **81.0%** | 5.0% |
| MoE, ternary | 9.8 M (3.3 M) | 0.1116 | 55.2% | 72.4% | 3.8% |

One seed each. A binomial standard error of about 1.6 points assumes
independent problems; it does not measure seed variance or correlations
between related synthetic problems.

- **MoE helps:** at the same compute per token, its validation loss is
  13% lower than dense.
- **With AdamW, ternary costs dense little** (0.1136 → 0.1152) but costs
  MoE much more (0.0986 → 0.1116), which erases most of MoE's gain. With
  Muon, the better-trained models show a similar cost for both (below).
- **Nobody generalizes to longer problems (d = 4).** Small transformers
  are known to struggle with this on arithmetic; it is on the list.
- **Chain** (multi-hop variable lookup) is the hardest task. Most
  failures are a wrong first step: the solution resolves variables from
  the given value toward the question, so the model has to pick the path
  to the queried variable before it has written anything about it.

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
- Length generalization on the curriculum.
- Faster GPU training: fused attention, bf16.
- Benchmark the inference engine against bitnet.cpp at equal model size.
