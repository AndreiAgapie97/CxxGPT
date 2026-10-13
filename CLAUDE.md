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
  ternary BitLinear projections, aux-loss-free MoE balancing.
- `src/ternary.h` and `src/infer.h`: inference on bitplane int8 kernels
  (AVX2, AVX-512, VNNI). A trained ternary checkpoint runs on them.
- `src/tokenizer.h`: byte-level BPE; digits are never merged.
- `src/reasoning.h`: a synthetic reasoning curriculum with held-out harder
  test levels. `cxxgpt prep-qa` turns GSM8K-style JSON Lines into the same
  format.
- `src/gpu.cu` (interface `src/gpu.h`): the CUDA training backend, the
  same model op for op, fp32 with cuBLAS (TF32 by default).
  `tests/test_gpu.cpp` compares it with the CPU engine. Build with
  `make gpu`, giving `cxxgpt-gpu` (`train --gpu`) and `test_gpu`.
- `src/cxxgpt.cpp`: the CLI (train, sample, eval, data prep, benchmarks).
- `ternary_moe.cpp`: kernel tests and the kernel benchmark.

## Plan

1. CPU, tiny scale (this VM): compare dense, ternary, MoE and ternary MoE
   on the reasoning curriculum; fix what the evals expose.
2. GPU training on the owner's laptop (RTX 4070 8 GB, under WSL2):
   - a CUDA backend for the same model, tested op by op against the CPU
     engine;
   - overnight runs on public data (general text plus math and reasoning
     sets).
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
