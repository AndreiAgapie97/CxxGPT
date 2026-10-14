# CPU memory measurements

Source `50efa7d`, GCC 14.2, two threads, quota two CPU equivalents. Run
`ROUNDS=3 THREADS=2 scripts/memory-benchmark.sh runs/new-memory` from that
revision. All workload computation is native C++; the archived CSV aggregation
uses Python's standard library. No research compilation or other experiment
ran concurrently with measurement.

The sweep completes 216 fresh processes: 72 configurations, three rounds each.
Training covers dense/MoE, fp32/ternary, AdamW/Muon, vocabularies 256/8192,
batch two and sequence lengths 32/128. Decode covers the ternary training
model, a clone with its effective weights and fp32 activations, and packed
inference, contexts 128/512 and batches one/four (packed supports one).
Dimension is 64, layers two, heads four, dense FFN 128, or four top-2 experts
of width 64 without shared experts. Three fixed cyclic-label updates or
16 forced decode tokens exercise storage; these are not accuracy experiments.

`raw.csv.gz` contains every category at startup, loaded, ready and worked
stages, identified by process filename. `summary.csv` selects worked/total,
records exact payload/capacity and median/min/max RSS counters across rounds.
Payload and capacity agree across all three runs of every configuration.
Bytes are actual vector elements (`size`) and allocation slots (`capacity`),
not object sizes or total heap usage. Metadata, strings, container objects,
allocator overhead and worker-local GEMM/attention/Muon scratch are excluded.
Packed payload includes allocated inactive placeholder matrices; logical
weights separately include active tensors and scalar quantization scales.

RSS is Linux `/proc/self/status` VmRSS; `peak_rss_bytes` is Linux
`getrusage(RUSAGE_SELF).ru_maxrss * 1024`. These approximate kernel counters
can disagree; recorded VmRSS can exceed the reported high-water mark. Neither
counter is an exact allocation audit. Peak is cumulative and includes source
model construction, packing/cloning and initialization, even after source
arrays are freed. Launch directly from the provided Bash driver: a large
interpreter launcher can contaminate the inherited high-water mark. Three
samples provide an observed range, not a statistical confidence interval.

| Workload, vocab 256 | Owned array bytes | Median VmRSS bytes | Reported median peak bytes |
| --- | ---: | ---: | ---: |
| Dense fp32 training, B2/T128, AdamW | 5,854,232 | 8,699,904 | 8,421,376 |
| Dense ternary training, same shape | 6,248,728 | 9,097,216 | 8,830,976 |
| MoE fp32 training, B2/T128, AdamW | 7,531,688 | 10,334,208 | 10,014,720 |
| MoE ternary training, same shape | 8,124,840 | 10,928,128 | 10,555,392 |
| Dense fp32 clone decode, B1/context128 | 940,892 | 3,231,744 | 4,112,384 |
| Dense packed decode, B1/context128 | 233,872 | 2,576,384 | 3,584,000 |
| Dense packed decode, B1/context512 | 653,200 | 3,117,056 | 4,096,000 |

The fp32 model no longer stores an unused quantized copy. This removes one
float per parameter: 394,496 bytes dense and 593,152 bytes MoE in this shape,
or 20% of the former five parameter-sized training arrays. Total training
memory includes substantial activations and scratch, so the total saving is
smaller. Four before/after CLI checkpoints, dense/MoE with AdamW/Muon, are
byte-identical. The CLI also checks fp32-to-ternary warm starts.

Packed dense KV storage rises from 131,072 to 524,288 bytes as reserved context
rises from 128 to 512, despite decoding only 16 tokens. It occupies 56% then
80% of packed array payload. With vocabulary 8192 and context128, dense
packed payload is 2,297,232 bytes versus 5,035,868 for the fp32 clone; float
embeddings dominate logical weights. The clone retains gradients and backward
buffers, so this ratio also reflects different inference implementations.
It does not isolate ternary compression. CPU Muon retains full-sized m/v;
its worker scratch appears in RSS, not the optimizer-array category. GPU
memory and runtime validation remain pending.
