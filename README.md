# CxxGPT

CPU inference kernels for a ternary-weight (BitNet-style) Mixture-of-Experts
layer, in one C++17 file: `ternary_moe.cpp`.

Every weight is −1, 0 or +1, with one fp32 scale per matrix. A 4096×4096
matrix takes 4 MB instead of 64 MB in fp32.

## Build and run

```sh
make            # g++ -O3 -march=native ...
make test       # correctness tests only (exit status 0 iff all pass)
make bench      # tests, then GEMV and MoE benchmarks
make asan tsan  # tests under AddressSanitizer+UBSan / ThreadSanitizer
```

Or directly: `g++ -O3 -march=native -std=c++17 -pthread ternary_moe.cpp -o ternary_moe`,
then `./ternary_moe [-t threads] [--tests]`.

`-march` picks the kernels: scalar (`x86-64`), AVX2 (`haswell`), AVX2 +
AVX-VNNI (`alderlake`), AVX-512 (`skylake-avx512`), AVX-512 + VNNI
(`icelake-server`; Zen 4 gets it with `native`).

## How it works

**Weights: two bitplanes.** Each row is stored as two bitmasks, `pos`
(weight is +1) and `neg` (weight is −1), with 32 weights per `uint32_t` word.

**Activations: int8.** The input vector is quantized symmetrically to
[−127, 127] with one scale per vector. −128 is excluded because it can't be
negated in int8.

**int8 kernels (B4–B6b).** These apply the masks to the activations, giving
`x`, `−x` or `0` per byte, then sum. The AVX2 kernel uses
`maddubs` + `madd`; the VNNI kernels use one `vpdpbusd`. `vpdpbusd`
sits on the accumulator's dependency chain, so the sum is split across 2
(256-bit) or 4 (512-bit) accumulators. With a single accumulator the 512-bit
kernel loses about half its gain.

**Threading.** A persistent pool with spin-then-sleep workers. Threads claim
row chunks from an atomic cursor that packs `(generation << 32 | next row)`.
A worker that wakes up late for a finished job fails its claim instead of
touching the next job. An empty `parallel_for` costs about 3 µs on 4 threads;
the earlier condition-variable handoff cost about 47 µs.

**MoE layer.** A top-k router with softmax over the selected experts, and
SwiGLU experts with hidden size 4·d_model. Kernels compute row ranges, so a
token runs as two pool jobs:
1. gate, up and SiLU for every routed expert at once;
2. every routed expert's down projection and the weighted sum.

The token's activations are quantized once and shared by all its experts.

## Performance

One run on a 4-vCPU Xeon VM with AVX-512 + VNNI. These shared VMs are noisy;
expect ±20% between runs.

GEMV, M = K = 4096, 60% zero weights. Times are medians in ms. The fp32
baseline is a vectorized dense dot product, not a scalar loop.

| kernel | 1 thread | vs fp32 | 4 threads | vs fp32 |
|---|---:|---:|---:|---:|
| fp32 dense | 2.62 | 1.0× | 0.88 | 1.0× |
| B3: bitplane AVX2, fp32 act. | 2.85 | 0.9× | 1.06 | 0.8× |
| B3b: bitplane AVX-512, fp32 act. | 1.23 | 2.1× | 0.36 | 2.4× |
| B5: int8 AVX2 | 0.85 | 3.1× | 0.28 | 3.1× |
| B5b: int8 AVX-512 | 0.57 | 4.6× | 0.13 | 6.7× |
| B6: int8 VNNI 256-bit | 0.75 | 3.5× | 0.23 | 3.8× |
| B6b: int8 VNNI 512-bit | 0.32 | 8.3× | 0.12 | 7.6× |

The int8 kernels have ~0.8% relative error against fp32. That comes from
quantizing the activations; the fp32-activation kernels are exact to
rounding.

MoE layer (d_model 512, 8 experts, top-2, int8 VNNI 512-bit): about
**50 µs per token on 4 threads**. Before the pool rewrite and job fusion it
was about 360 µs.

## Tests

`--tests` checks the following:
- **Kernels:** every kernel against a double-precision reference.
  Edge cases: all-zero, all ±1, word boundaries, every SIMD tail length,
  and row counts large enough to use the pool.
- **Full MoE layer:** routing, softmax weights and both pool jobs,
  against a double-precision reference.
- **Quantizer:** ±amax maps to ±127, nothing maps to −128, and the SIMD and
  scalar paths agree bit for bit.
- **Math helpers:** `silu_mul` within 1e-6 of double; `dot_f32`.
- **Thread pool:** every index handed out exactly once, including after the
  workers fall asleep.
- **Concurrency:** two threads calling every kernel at once.

CI runs these with GCC and clang on several `-march` targets, plus ASan/UBSan
and TSan.

## Not here yet

- **Batched (prefill) GEMM:** several tokens sharing each weight load.
- **Real model:** loading actual model weights, attention, a tokenizer.
