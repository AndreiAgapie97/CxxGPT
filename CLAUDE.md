# CxxGPT

## Goal

A highly intelligent AI model on a much more efficient architecture than
today's Python-based models. The efficiency levers are:
- ternary (1.58-bit) weights;
- int8 activations;
- sparse Mixture-of-Experts;
- hand-written C++ kernels, with no Python framework in the inference path.

## Where things stand

`ternary_moe.cpp` has the kernels, a thread pool, one MoE layer, tests and a
benchmark. The kernels use bitplane weights with int8, AVX2, AVX-512 and VNNI
paths. There is no full model yet: no weights, attention, tokenizer or
training.

Intelligence comes from training, at a scale this machine can't provide.
The near-term path is to run an existing open ternary model on this engine,
validated against its reference implementation. BitNet b1.58 2B4T is a
candidate: it uses W1.58A8, the same scheme as these kernels (ternary
weights, per-token absmax int8 activations). The engine to beat on speed is
bitnet.cpp, not a Python stack.

## Conventions

- Commit directly to `main` (owner's instruction), and push each verified step.
- Before committing, run all of these; CI runs the same set:
  - build with `-Wall -Wextra -Werror` under both g++ and clang++;
  - `make test`, `make asan` and `make tsan`.
- Every numeric change is tested against a double-precision reference.
  Wherever the code claims SIMD and scalar paths agree bit for bit, a test
  checks it.
- Benchmarks:
  - use honest baselines: vectorized fp32 now, real runtimes once a model runs;
  - this VM is noisy, so interleave A/B runs and report medians.
