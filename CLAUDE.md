# CxxGPT

## Goal

A highly intelligent AI model on a much more efficient architecture than
today's Python-based models, built our own way: our own architecture,
training engine, tokenizer and kernels, in C++ (CUDA allowed). The only
thing we take from outside is public data. Frontier quality is out of reach
on one laptop, but we aim at it anyway.

The efficiency levers are:
- ternary (1.58-bit) weights, trained with BitLinear;
- int8 activations;
- sparse Mixture-of-Experts;
- hand-written kernels, with no Python framework anywhere (small Python
  scripts for data handling are fine; no PyTorch).

Reasoning is the first capability we go after.

## Where things stand

- `src/`: the training engine (`model.h`, `ops.h`, `gemm.h`, `train.h`),
  tested against double-precision gradients. Dense or MoE FFN, optional
  ternary BitLinear projections, aux-loss-free MoE balancing. AdamW, or
  Muon (`--muon`) for the matrices inside the blocks.
- `src/ternary.h` and `src/infer.h`: inference on bitplane int8 kernels
  (AVX2, AVX-512, VNNI). A trained ternary checkpoint runs on them. The
  embedding/output layer, and attention with `--fp-attn`, are int8 rows
  (one scale per row); `eval-reasoning --as-engine` rounds a model the
  same way.
- `--ternary --fp-attn` (ternary FFN, full-precision attention) gave the
  best MoE run on the curriculum (71.9%, against 64.2% all full precision
  and 62.4% all ternary), but chain accuracy "clicks" at a different step
  in each run, so one run per model does not settle it: repeat with
  several seeds (fast on the laptop GPU) before the ternary nights. Until
  then `night.sh` `TERNARY=1` keeps attention full precision (README,
  "Ternary FFN, full-precision attention").
- `src/tokenizer.h`: byte-level BPE; digits are never merged.
- `src/reasoning.h`: a synthetic reasoning curriculum with held-out harder
  test levels (`gen-reasoning --chain-plan`: chain solutions look the
  variables up first, which more than doubles chain accuracy). `cxxgpt
  prep-qa` turns GSM8K-style JSON Lines into the same format.
- `src/gpu.cu` (interface `src/gpu.h`): the CUDA training backend, the
  same model op for op, fp32 with cuBLAS (TF32 by default; `--matmul
  bf16`), and fused attention kernels (`src/attention.cuh`, opt-in with
  `--fused-attn`; tested on a CPU emulator, `tests/cuda_emu.h`).
  `--profile` times each part of a step. `tests/test_gpu.cpp` compares it
  with the CPU engine. Build with `make gpu`, giving `cxxgpt-gpu` (`train
  --gpu`) and `test_gpu`. On the laptop's RTX 4070, everything passes
  except three opt-in checks: `dense bf16`, and fused attention at
  `heads of 64` (70 tokens, where cuBLAS also misses the bound) and
  `head of 128` (about 3x cuBLAS's error); docs/LAPTOP.md "GPU options".
- `docs/research.md`: the research log from the merged `gpt/research`
  branch (stable cross-entropy, token-weighted accumulation, optimizer
  grids, digit curricula), with raw results under `docs/results/`.
- `src/cxxgpt.cpp`: the CLI (train, sample, eval, data prep, benchmarks).
- `scripts/get_data.py`: downloads public datasets (GSM8K, TinyStories,
  FineWeb-Edu, OpenMathInstruct-2, MetaMathQA) into `data/`, as text with
  `<|endoftext|>` between documents (for `text2tok --doc-sep`) or as
  GSM8K-style JSON Lines (for `prep-qa`). Tested against a local fake of
  the Hugging Face endpoints, since huggingface.co is blocked here; CI's
  `data-sources` job checks the real files and columns (`--check`).
- `scripts/compare.sh`: the architecture comparison on the curriculum.
- For the laptop: `scripts/setup_wsl.sh` (toolchain, build, all tests),
  `scripts/probe.sh` (GPU speed and memory per model size; prints the
  night.sh command for each), `scripts/prepare_data.sh` (downloads, our
  tokenizer, token files) and `scripts/night.sh` (one night of training,
  validated on the GSM8K test set; `INIT=` starts from an earlier night's
  weights); `scripts/report.sh` sums up a night's log. `docs/LAPTOP.md`
  is the runbook.
- `train --init FILE --ternary` turns a full-precision model ternary for
  the rest of its training (same weights, used differently).
- Overnight safety: a step with a non-finite gradient is skipped, ten in a
  row stop the run without overwriting the last checkpoint, and
  checkpoints are written to a temporary file and renamed.
- `ternary_moe.cpp`: kernel tests and the kernel benchmark.

## Plan

1. CPU, tiny scale (this VM): compare dense, ternary, MoE and ternary MoE
   on the reasoning curriculum; fix what the evals expose.
2. GPU training on the owner's laptop (RTX 4070 8 GB, under WSL2):
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
  - Windows, used through WSL2 Ubuntu 24.04;
  - 8-core/16-thread AMD (Zen 4, AVX-512), 32 GB RAM;
  - RTX 4070 laptop GPU with 8 GB VRAM;
  - under 200 GB free disk;
  - available for training overnight only.

## Conventions

- C++20 (GCC 13 has no `<print>`, so not C++23 yet).
- Commit directly to `main` (owner's instruction), and push each verified
  step.
- Before committing, run all of these; CI runs the same set:
  - build with `-Wall -Wextra -Werror` under both g++ and clang++;
  - `make test`, `make asan` and `make tsan`.
- The CUDA code cannot run on this VM. CI compiles and links it with nvcc;
  here, compile-check it with clang's CUDA mode (`clang++ -x cuda
  --cuda-gpu-arch=sm_89 --cuda-device-only`, then `--cuda-host-only`)
  against the toolkit headers from the PyPI `nvidia-*` wheels. Run
  `make gpu-test` on a GPU before trusting a change to `gpu.cu`.
- Every numeric change is tested against a double-precision reference.
  Wherever the code claims two paths agree (SIMD and scalar, decode and
  forward, engine and training model), a test checks it.
- Benchmarks:
  - use honest baselines: vectorized fp32 on the same threads, and later
    real runtimes;
  - this VM is noisy, so interleave A/B runs and report medians.
- The owner is a C++ beginner without an ML background: explain decisions
  in plain words, and keep code readable.
