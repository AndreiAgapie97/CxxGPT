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
./cxxgpt sample --ckpt runs/model.ckpt --prompt $'Q: 47+85\nA: '
```

`cxxgpt train --help` lists every option. Useful ones: `--muon` (the Muon
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
scripts/probe.sh                     # GPU speed and memory per model size
TOKS=<tokens/s> scripts/night.sh     # one night of training, checked on GSM8K
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
- **Ternary costs dense little** (0.1136 → 0.1152) but costs MoE much
  more (0.0986 → 0.1116), which erases most of MoE's gain. A likely cause:
  each expert sees only a quarter of the tokens, so its full-precision
  latent weights move slowly and rarely cross the thresholds that change
  their ternary values. Higher learning rates for ternary weights and
  Muon are the next experiments.
- **Nobody generalizes to longer problems (d = 4).** Small transformers
  are known to struggle with this on arithmetic; it is on the list.

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
- Length generalization on the curriculum.
- Faster GPU training: fused attention, bf16.
- Run an existing open ternary model (BitNet b1.58 2B4T) on our inference
  engine and compare speed with bitnet.cpp.
