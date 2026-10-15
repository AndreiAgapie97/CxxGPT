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
  `gen-reasoning --mistakes P` plants one wrong statement and its
  correction in a share P of the training solutions; the wrong statement
  stays visible but gets no loss (`[[no-loss]]` spans, `src/loss_mask.h`,
  also read by `text2tok`). Validation and test stay clean.
- `src/calc.h` (`--tools calc-notes|calc-plain` on `sample` and
  `eval-reasoning`): an exact calculator (fractions, 64-bit, no rounding)
  that finishes the arithmetic the model starts (`<<48/2=` or `16 x 7 = `);
  `eval-reasoning` then prints the score without and with it, and the call
  counts. Neither mode fires on the curriculum's `a*b=c` steps
  (docs/calculator.md).
- `eval-reasoning --tries N --temp T --seed S` (or `--effort
  low|medium|high`: 1/8/32 tries): samples each problem N times and votes
  on the final answer (maj@N, also pass@N and the first try's score). Each
  try has its own seed, so results do not depend on `--batch` or
  `--limit`; every headline names its effort (`(greedy)`, `(maj@8, ...)`).
- `src/gpu.cu` (interface `src/gpu.h`): the CUDA training backend, the
  same model op for op, fp32 with cuBLAS. `train --gpu` stores the
  products' inputs in bf16 by default (`--matmul bf16`, fp32 master
  weights: 1.45x the tokens/s of `--matmul tf32`, the same loss curve);
  ternary models default to TF32, which bf16 does not take yet. Fused
  attention kernels are the default (`--no-fused-attn` for cuBLAS
  products): on the tensor cores with TF32/bf16
  (`src/attention_tc.cuh`, tested on the GPU only; 1.18x the cuBLAS
  attention's tokens/s on the night-1 model, at 8 x 16, which only fits
  with them), in exact fp32 otherwise (`src/attention.cuh`, also tested on
  a CPU emulator, `tests/cuda_emu.h`).
  `--profile` times each part of a step. `tests/test_gpu.cpp` compares it
  with the CPU engine. Training's reasoning exams run on the GPU too
  (`GpuTrainer::decode_begin/decode_step`, a KV cache kept in the gradient
  buffer between steps; tested against `Model::decode_step`): 16 s instead
  of 5 minutes for 250 problems on the night-1 model. Build with `make gpu`, giving `cxxgpt-gpu` (`train
  --gpu`) and `test_gpu`. On the laptop's RTX 4070, everything passes
  except two checks, neither on what the nights run, both on a norm
  weight's gradient: the fp32 fused kernels at `heads of 64` (70 tokens,
  where cuBLAS also misses the bound) and `head of 128` (about 3x
  cuBLAS's error); docs/LAPTOP.md "GPU options".
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
  Decisions are made on a dev set held out of GSM8K train (and out of
  MetaMath and OpenMath2), never on the GSM8K test set, which gives
  headline numbers only.
- `src/wordgen.h` (`cxxgpt gen-word`): GSM8K-style word problems with exact
  answers and worked solutions (with or without calculator notes). Six
  story shapes (a running quantity, comparisons, totals over people,
  rates, a fraction or percentage of a part, purchases and change), 2-8
  operations, distractor sentences, money spelled three ways; 120 sentence
  patterns, 40 of them held out for the test split. Besides the plan's own
  exact check, a second solver works the five newer shapes out from the
  story's quantities alone.
- GSM8K dev set: `scripts/get_data.py` holds out 500 GSM8K train problems
  (fixed seed) and removes their MetaMath and OpenMath2 rewrites;
  `DEV=1 scripts/night.sh` validates on it instead of the test set.
  `cxxgpt leak-check` finds questions sharing a 13-word run with training
  files. Night 1's weights have already seen the dev questions.
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
- `src/cxxgpt.cpp`: the CLI (train, sample, eval, data prep, benchmarks).
- `scripts/get_data.py`: downloads public datasets (GSM8K, TinyStories,
  FineWeb-Edu, OpenMathInstruct-2, MetaMathQA) into `data/`, as text with
  `<|endoftext|>` between documents (for `text2tok --doc-sep`) or as
  GSM8K-style JSON Lines (for `prep-qa`). Tested against a local fake of
  the Hugging Face endpoints, since huggingface.co is blocked here; CI's
  `data-sources` job checks the real files and columns (`--check`).
- `scripts/compare.sh`: the architecture comparison on the curriculum.
- For the laptop (native Windows since 2026-10-15; the scripts run in Git
  Bash): `scripts/build_windows.cmd` (build, kernel, engine and GPU tests;
  `scripts/setup_wsl.sh` is the Linux equivalent, kept and run by CI),
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
- Every numeric change is tested against a double-precision reference.
  Wherever the code claims two paths agree (SIMD and scalar, decode and
  forward, engine and training model), a test checks it.
- Benchmarks:
  - use honest baselines: vectorized fp32 on the same threads, and later
    real runtimes;
  - this VM is noisy, so interleave A/B runs and report medians.
- The owner is a C++ beginner without an ML background: explain decisions
  in plain words, and keep code readable.
