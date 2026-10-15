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

### Native Windows build

Install Visual Studio's C++ tools, LLVM/clang-cl, CMake and Ninja. From an
**x64 Native Tools Command Prompt for VS**, run:

```bat
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

This builds `build\cxxgpt.exe`, `build\ternary_moe.exe` and
`build\test_engine.exe`. The tests run directly on Windows; the shell CLI
tests stay on Linux.

With the CUDA toolkit installed, add `-DCXXGPT_CUDA=ON` to the first
command for `build\cxxgpt-gpu.exe` (`train --gpu`) and `build\test_gpu.exe`
(`-DCXXGPT_CUDA_ARCH=sm_89` names the GPU; the default, `native`, is the one
in the machine). nvcc compiles `src/gpu.cu` with Visual Studio's `cl.exe`,
the only host compiler it takes on Windows, and clang-cl compiles the rest;
both use the same C++ runtime (`/MD`). The laptop's command, with CUDA 13.3:

```bat
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release ^
  -DCXXGPT_MARCH=native -DCXXGPT_CUDA=ON -DCXXGPT_CUDA_ARCH=sm_89
```

CI does not build the CUDA half on Windows (its runner has no CUDA); the
Linux CUDA jobs compile the same `gpu.cu`.

`scripts\build_windows.cmd` does all of this from any terminal: it finds
Visual Studio and clang-cl, builds the laptop's way (with CUDA when `nvcc`
is installed) and runs the kernel, engine and GPU tests. The training
scripts (`night.sh`, `probe.sh`, `prepare_data.sh`, `report.sh`) run in Git
Bash and use `build\`'s programs there ([docs/LAPTOP.md](docs/LAPTOP.md)).

Windows CI builds the same targets another way: it finds Visual Studio with
`vswhere`, configures with that Visual Studio generator and its clang-cl
toolset (`-T ClangCL -A x64`), builds with `--config Release` and runs
`ctest -C Release -V`.

For timing on the laptop, add `-DCXXGPT_MARCH=native` to the first command
(in a fresh `build` folder). Everything is then compiled for that CPU, as
the Linux Makefile's `-march=native` does: the fp32 GEMM, the int8/fp32
helpers and the ternary kernels all use AVX-512. **Only this native build is
comparable with WSL timings**; the default build below keeps the GEMM at SSE
and the helpers scalar. (CMake's Release is `-O2`, the Makefile's `-O3`.) A
native binary may not run on a different CPU. CI builds the default one.

The default Windows binary includes AVX2, AVX-512 and VNNI row kernels, each
compiled with its own instruction settings. At runtime it checks both the CPU's
features and Windows' support for saving vector registers, then chooses the
widest available int8 kernel, using the same preference as Linux. The rest
of the binary uses baseline x86-64 instructions, so an older CPU can use the
scalar kernels safely. The Linux Makefile and its `-march` choices are unchanged.

One kernel differs from WSL in the default build. It compiles B6 (int8 VNNI
256-bit) for AVX-VNNI, the encoding of Intel's newer CPUs, so on a CPU that
has AVX512-VNNI but not AVX-VNNI, such as the laptop's Zen 4, it leaves B6
out. WSL's `-march=native` build, like the native Windows build, has B6
there, in its AVX-512 encoding. The MoE layer uses B6b (512-bit) on Zen 4
either way. `ternary_moe --tests` prints the kernels it tested, and the
default build also prints the CPU features it found.

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
| `eval-reasoning` | exact-answer accuracy per task and difficulty; greedy by default, or a vote over several sampled tries (below); `--tools calc-notes` or `calc-plain` also scores the same problems with an exact calculator filling in the model's arithmetic ([docs/calculator.md](docs/calculator.md)); `--gpu` (with `cxxgpt-gpu`) decodes on the GPU, with exact fp32 products by default, so the scores are the CPU's (`--matmul tf32` or `bf16` scores the model as training computes it) |
| `sample` | generates text from a prompt (`--tools` as for `eval-reasoning`) |
| `bench-infer` | times the ternary inference engine against an fp32 baseline |
| `leak-check` | finds questions (such as the GSM8K dev set) that share a run of 13 words with training files |
| `qa-context` | how often random training windows train an answer without its question in view (`--data A.tok,B.tok --seq 512`) |

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
- `--no-repeat N` (also on `sample`; off by default) never lets an answer
  write a run of N tokens a second time: small models greedily repeat a
  sentence until the token budget runs out. Only the answer counts, not
  the question it may copy, and shorter repeats ("9 * 2") stay allowed, so
  N should be long (8-16). The headline says so (`(greedy, no-repeat 12)`).
  Every evaluation also prints how many first tries ended without a
  `#### ` line. The model can dodge the block with a near-copy (another
  apostrophe in the same sentence); unlikelihood training (issue #34) is
  the training-side fix.
- `--labels FILE.jsonl` (issue #16, `src/failure_labels.h`) says why each
  first try is wrong, by rules anyone can check:
  - `correct`;
  - `unfinished`: no `#### ` line before `--max-new` ran out;
  - `format`: the right number, written another way (`$18`);
  - `arithmetic`: a written calculation that is wrong, checked by the
    exact calculator (`480 * 480 = 112,400`), before any sign of the
    right number;
  - `passed_gold`: the right number appears in the work, then another
    answer is given (18 dollars a day, then "in a week, 18 * 7 = 126");
  - `other`: wrong with no wrong calculation in sight: a wrong plan or
    goal, for a person to tell.

  The file has a line with the run's settings, then one line per problem
  with its label, every checked calculation, its flags and the text, so
  each label can be audited. The flags count loops (a line written twice,
  or a run of 8 tokens repeated) and numbers from outside the question;
  they fire on right answers too, so the summary reports them for right
  and wrong answers apart. Labels describe the run without the
  calculator.

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
- `--distractors on|off`: include or omit an irrelevant sentence (default
  on). Twenty-four patterns, some with numbers, can appear before, between,
  or after the story's arithmetic sentences. The splits use different
  distractor wording; a distractor number never appears in the calculation.
- `--notes on|off`: include or omit GSM8K calculator notes such as
  `<<48/2=24>>24` (default on). This changes only the worked solution.
- `--split train|test`: 84 training sentence patterns or 42 held-out test
  patterns, grouped into 21 arithmetic situations. The opening patterns
  differ too, so the splits cannot share a complete question, even with
  different seeds. Each file also omits duplicate questions.

Stories include a running quantity, comparisons between two people, totals
over two or three people, production per hour/day/week combined with another
person's contribution, a fraction or percentage of one part combined with
the rest, and unit prices times quantities followed by change from a payment.
In the fraction-or-percentage stories, which ask how many items are kept,
extra items are always described as kept, never as gifts.
Each shape has both training and held-out sentence patterns. Longer stories
add more contributions or purchases before the final comparison or total.
The running-quantity stories still include wages, liquid volumes, gifts,
donations, equal shares, selling, buying, and liters converted to milliliters.
Names, objects, prices, quantities, and solution phrasing vary. About a third
of the solution steps are written the GSM8K way, as a sentence that names the
person and the unit and carries the note inline (`Maya now has 84+4 =
<<84+4=88>>88 toy cars.`). Objects must remain whole; money and
volume may use exact hundredths. Questions print fractional money with two
decimal places (`$7.50`, `2.50 dollars`), and whole amounts as `$7`; notes
and answers keep the shortest exact spelling. Follow-up volumes use the
current unit, so a gift of 2.25 liters becomes 2250 milliliters after a
conversion. No floating-point arithmetic or rounding is used.
An attempt is discarded if any intermediate value is negative, exceeds
1,000,000, needs more than two decimal places, or gives fractional objects.
The step count and story shape stay fixed while invalid attempts are retried.

These generated stories are still less varied than human-written math.
Long stories outside the running-quantity shape are repetitive: at 6-8
steps, each of the five other shapes is one multiplication (or fraction, or
percentage) followed by a string of additions, and on the test split, with
only two patterns per sentence family, the same addition pattern often
appears two or three times in one story. Success on its held-out wording alone is not evidence of improvement on
GSM8K or SVAMP. No generated datasets are checked into the repository, and
this command does not change the night data mix.

### Real data and GPU training

```sh
scripts/prepare_clean.sh             # the audited mix (below), night.sh's default
scripts/prepare_data.sh              # the mix of nights 1-2: download, tokenizer, token files
scripts/probe.sh                     # GPU speed and memory per model size, and the
                                     #   night.sh command for each
TOKS=... MODEL=... scripts/night.sh  # one night of training, checked on GSM8K
scripts/report.sh night1             # the night on one screen
scripts/ledger.sh night1 "note"      # its row in docs/runs.md, the ledger of runs
```

`prepare_data.sh` fetches GSM8K, TinyStories, FineWeb-Edu,
OpenMathInstruct-2 and MetaMathQA (`scripts/get_data.py`, which needs only
`pyarrow`), trains a 16k-token BPE on a sample, and encodes everything.
[docs/LAPTOP.md](docs/LAPTOP.md) has the whole setup, from a fresh Windows
laptop to a night of training.

Settings are chosen on a dev set, not on the GSM8K test set: `get_data.py`
holds out 500 GSM8K train problems (`data/gsm8k/dev.jsonl`, the 500 with the
smallest sha256 of their question) and removes them, and the MetaMath and
OpenMath2 rows made from them, from training; it prints how many rows each
filter removed. `DEV=1 scripts/night.sh` validates on them. Data prepared
before the dev set existed needs one more `scripts/prepare_data.sh`, then
one exact-text check that no dev question is left in training:
```sh
./cxxgpt leak-check --questions data/gsm8k/dev.txt \
  --train data/gsm8k/train.txt,data/old/metamath/train.txt,data/old/openmath2/train.txt
```
[docs/LAPTOP.md](docs/LAPTOP.md) ("The dev set") gives the counts to
expect.

**The audited data (since 2026-10-15, issue #46).** A random sample of
OpenMath2 found clear mistakes in 26 of its 40 `augmented_math` solutions,
and night 2's answers carry the same habits (36% write `\boxed{}`; issue
#16). From night 3 on, a new model trains from random weights on openly
licensed text and exact math, and MetaMath, OpenMath2, FineWeb-Edu and
TinyStories are left out. `scripts/prepare_clean.sh` builds it, from the
Common Pile downloads in `data/public-20261015/` (never changed) with
`scripts/prep_public.py`:

| File | What | Tokens | In the mix |
|---|---|---|---|
| `textbooks` | LibreTexts and Pressbooks open textbooks | 240M | x2 |
| `wiki` | Wikipedia, Wikibooks, Wikisource, Wikinews articles (one shard of 35) | 342M | x1 |
| `doab-part` | academic books (DOAB), cut from 2 of 5 shards | 901M | x1 |
| `arxiv-part` | openly licensed arXiv papers, cut from 1 of 8 shards | 405M | x1 |
| `word-train` | 1M exact word problems (`gen-word`, no calculator notes) | 189M | x1 |
| `gsm8k-train` | 6,966 human-written GSM8K train problems (7 near-copies of a test or dev question dropped) | 1.2M | x8 |

That is 2.33B tokens in the mix, about 3.7 nights. Empty, short and
wrongly decoded records, non-article wiki pages, Wikiquote and repeated
texts are dropped; DOAB and arXiv are cut at a document boundary, so that
academic books and papers are a little over half of the mix instead of
nearly three quarters. No GSM8K test, GSM8K
dev or SVAMP dev question shares a 13-word run with any of it. The 500
held-out GSM8K problems are the dev set (`DEV=1`): the new model has never
seen them. `night.sh` trains on this mix unless `MIX=` names others. The
old sets are kept in `data/old/` (moved there, not deleted): for comparing
old and new data, and for questions about nights 1 and 2. `prepare_data.sh`
writes them to `data/` again if run.

**Answers with their questions.** Training windows start at random
offsets, so a window often begins in the middle of a math problem: the
answer is trained while its question is out of view. `./cxxgpt qa-context
--data data/gsm8k/train.tok,data/old/metamath/train.tok,data/old/openmath2/train.tok`
counts it exactly. `train --sampler qa` (`scripts/night.sh --sampler qa`)
moves such a window back to the start of its problem, when the whole
problem fits in a window. At `--seq 512`, the share of answer tokens that
see their whole question:

| Source | `--sampler random` (default) | `--sampler qa` |
|---|---|---|
| GSM8K | 74% | 100% |
| MetaMath | 63% | 92% |
| OpenMath2 | 40% | 54% |

OpenMath2 stays low because 37% of its problems are longer than a window;
those keep random starts, or the end of their answers would never be
trained. Windows that start with a question train 3-7% fewer answer tokens
each (GSM8K: 277 instead of 297 per window). Web text and stories are not
affected, and validation keeps random windows. Whether it helps accuracy
is not measured yet (docs/research-plan.md, "Answers with their
questions").

**Loops: unlikelihood training (issue #34).** `--no-repeat` stops an
exact loop while decoding; `train --unlikelihood ALPHA --ul-ngram N` tries
to make the model stop wanting one. Wherever the training text breaks off
a run of N - 1 tokens it already wrote, in a way that differs from how it
went on the first time, the token that would have repeated the run is
pushed down: `ALPHA * -log(1 - p)` is added to the loss. The right next
token is never pushed down, runs shorter than N may repeat ("9 * 2"),
and masked text (the question, planted mistakes) never counts, so copying
the question costs nothing. The log keeps the cross-entropy as `loss` and
adds `ul` (the term without ALPHA, per trained token) and how often it
fires. It runs on the CPU engine and on the GPU (`train --gpu`), which
`test_gpu` checks against the CPU engine.

How often it fires (share of trained tokens with a token to push down,
windows of 512):

| N | GSM8K | MetaMath | OpenMath2 | TinyStories | FineWeb-Edu |
|---|---|---|---|---|---|
| 6 | 2.2% | 4.0% | 5.6% | 0.44% | 0.84% |
| 8 | 0.60% | 1.7% | 2.8% | 0.08% | 0.27% |
| 12 | 0.08% | 0.49% | 0.98% | 0.01% | 0.07% |

A first CPU run (4 layers of 128, MetaMath, 1,500 steps, N = 6, same seed
and data): with the term at a negligible weight (ALPHA = 1e-9, so it is
still measured), the probability the model gives to repeating grows as it
learns (`ul` 0.0044 at step 250, 0.0074 at 1,500). With ALPHA = 1
it ends 6% lower (0.0070), at the same validation loss (2.1001 against
2.0993). That is one seed on a small model; whether it removes loops in
the answers is the experiment in issue #34.

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
| `src/calc.h` | the exact calculator used during generation (`--tools`) |
| `src/no_repeat.h`, `src/unlikelihood.h` | repeated runs of tokens: blocked when decoding (`--no-repeat`), pushed down in training (`--unlikelihood`) |
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

`gen-reasoning --mistakes 0.1` gives about 10% of training solutions one
masked wrong step followed by a correct step; validation and test stay clean.
The default 0 preserves previous output byte for byte. Nonzero runs also
write `OUT.train.text` for inspection. `[[no-loss]]...[[/no-loss]]` marks the
wrong statement: the generator removes these annotations and sets its token
loss mask to zero, while keeping the statement as context. `text2tok` does
the same only with `--loss-markers` (otherwise the markers are plain text,
so stray ones in web text change nothing). Use the generator's `.train.tok`
to retain its question mask. The [experiment
plan](docs/research-plan.md#b-planted-mistakes-and-corrections) documents the
marker and paired 0%, 10%, 30% laptop commands.

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
- **The same codes as training.** The engine packs each weight with the
  training forward's own quantizer (`ternary_quantize`: scale = mean |w|
  per matrix or expert, code = round(w / scale) clamped to ±1, ties to
  even) on the fp32 weights the checkpoint keeps. So every weight gets the
  code training used, also exactly at ±scale/2 (`tests/test_ternary_codes.h`,
  issue #40). Nothing on that path rounds to bf16: a bf16 copy would change
  0.02% of a random model's codes, and a weight that a fine-tuning step
  moved just past a threshold would fall back.
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
  1.45x and are the default since 2026-10-14 (TF32 for ternary models).
  On the laptop two GPU checks still miss, both on a norm weight's gradient (docs/LAPTOP.md, "GPU options"),
  and `--profile` says what to speed up next (docs/research-plan.md,
  "Speed").
- Benchmark the inference engine against bitnet.cpp at equal model size.

### CPU build portability

The default `-march=native` binary targets the build machine's CPU. Rebuild
on the destination machine, or use `make MARCH=x86-64` for a portable baseline
within x86-64 on the same OS/ABI; compile-time kernels are selected accordingly. Runtime ISA
dispatch and cross-architecture binary portability are not implemented.
