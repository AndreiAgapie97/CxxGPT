# Masked accumulation and decode follow-up

All training and inference paths used here are native C++. Source revisions,
compiler commands, hashes, budgets, raw timing samples and checkpoint identities
are recorded. Benchmarks do not retrain their checkpoints.

* `validation/`: GCC 14.2 and Clang 19.1 builds/tests; ASan/UBSan and TSan
  including the real CLI; scalar and AVX2 engine checks; CUDA compile logs.
  Scalar/AVX2 checks precede the data/evaluation merge. CUDA compilation uses
  Clang 19, CUDA 12.4 headers/libdevice, `sm_89`, `-Wall -Wextra -Werror`, and
  `-Wno-unknown-cuda-version`. Empty CUDA logs mean no compiler diagnostics;
  host/device output files were generated successfully. No GPU runtime ran.
* `decode/trials.csv`: every measured trial, with scenario, phase and source
  revision. Two warmup pairs precede each set of 21 recorded pairs. First-run
  order alternates. Both paths use the same forced token stream and threads.
* `decode/summary.csv`: median milliseconds for each path; speedup is the
  ratio of these medians. `paired_min/max` are the extrema of individual
  fp32/packed timing ratios, not confidence intervals.
* `decode/quota-deltas.csv`: cgroup CPU counters after minus before each
  quota-aware matrix. Provenance subdirectories retain exact model options,
  source/binary/checkpoint hashes, compiler commands, hardware and stdout.
* `storage.csv`: native audit of the two trained seed-1 checkpoints.
* `sorted-eval/`: same-checkpoint reasoning evaluations after Claude's
  prompt-length batching change, with hashes and old/new exact-count comparison.
  All twelve counts remain unchanged (0–2 correct answers out of 60).

The `affinity3` phase used one and three threads at source `e5f302c`. Although
three CPUs are in the affinity mask, this environment's cgroup limits CPU time
to two CPU equivalents (`cpu.max = 200000 100000`). Three-thread timing has
large stalls. The `quota2` phase reruns one and two threads at source `891f990`;
no other compilation or experiment ran concurrently with measurements.
The recorded checkpoint hashes are the same across phases. Both code and
weight caches are warm; each trial resets the KV cache. Packing and model
construction are outside the timer. Context length is 16, 64 or 128, batch
is one. Decode timing includes cache reset and argmax, without text generation
or sampling. This is a small warm-cache workload on a shared VM.

The fp32 baseline holds the same ternary projection weights in float, with
unquantized activations. It uses the native vectorized training-model decode
path. The packed path uses the inference engine, including its implementations
of unquantized operations. Their timing difference therefore does not isolate
the benefit of ternary arithmetic. Argmax agreement measures activation and
implementation differences on the forced stream, not task accuracy.

The trained cases use the existing 2,000-step seed-1 reasoning checkpoints
(dim 64, one layer, vocabulary 256). The 8,192-token cases are untrained,
seed-1 shape controls with otherwise matched dimensions. Their argmax
agreement of 100% is not a language-quality result. Memory counts only weight
arrays; activations, KV cache, work buffers, allocator capacity and process
RSS are excluded. CPU training array totals include master weights, gradients,
quantized copies and two full optimizer arrays. They do not describe the
compact GPU Muon second-moment layout, whose runtime behavior is pending.

Reproduce a quota-aware trained case from the original checkpoint:

```sh
BENCH_THREADS='1 2' scripts/decode-benchmark.sh runs/new-decode \
    --ckpt runs/reasoning-2000-20261013/dense-ternary-seed1.ckpt
```

See `model-options.txt` in each provenance directory for the other scenarios.
Rebuilds force the requested compiler so its recorded identity matches the
binary. Results from different phases should not be used to isolate a thread
scaling effect: wall-time noise and changed run timing remain confounders.
