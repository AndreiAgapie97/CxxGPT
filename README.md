# CxxGPT

An experimental C++20 language-model engine for studying math problem solving
and measuring training and inference costs, using:

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
scripts (`night.sh`, `probe.sh`, `report.sh`) run in Git
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
during training), `--schedule wsd` (a flat learning rate, then a cooldown
over the last `--decay-steps`, so a run can be made longer or cooled down
from any checkpoint) and the control file, `OUT.control`: `lr-scale X`,
`save`, `eval`, `reason` or `stop` written into it act on the running
training (docs/LAPTOP.md, "Changing a run while it trains").

Add `--reason-report runs/model.reason` to `train` to save every evaluated
question and answer, with the existing heuristic failure labels, in
`runs/model.reason.step-N.jsonl`. Each file starts with evaluation settings
and ends with label counts and the original zero-based problem IDs that
became correct or regressed since the previous evaluation. Intermediate
evaluations use the current weights; they need not have a saved checkpoint.
Comparisons cover one training invocation: after `--resume`, the first
evaluation starts a new baseline. Use a distinct prefix for each run;
evaluating the same step again replaces that step's file. Labels flag
patterns for inspection and do not establish the cause of a wrong answer.

### Commands

| command | does |
|---|---|
| `gen-reasoning` | writes the synthetic curriculum: problems with step-by-step solutions, train/val token files and a test set one difficulty level beyond training |
| `gen-word` | writes exact word problems, worked solutions, and separate train/test sentence patterns |
| `gen-math` | writes math problems built from a chosen answer (equations, fractions, number theory), each solved again by a second solver |
| `bpe-train` | trains a BPE tokenizer on a text file |
| `text2tok` | encodes text into a token file (parallel, streaming; `--doc-sep` splits documents) |
| `prep-qa` | turns question/answer JSONL (GSM8K and the like) into token files with a loss mask and a test set |
| `train` | trains or resumes a model (CPU, or GPU with `cxxgpt-gpu --gpu`); `--val-sources NAME=FILE,...` scores each held-out file on its own at every validation, Q&A files also on their final answers (`--val-log FILE.csv`) |
| `eval-reasoning` | exact-answer accuracy per task and difficulty (answers compared as numbers, below); greedy by default, or a vote over several sampled tries (below); `--tools calc-notes` or `calc-plain` also scores the same problems with an exact calculator filling in the model's arithmetic ([docs/calculator.md](docs/calculator.md)); `--gpu` (with `cxxgpt-gpu`) decodes on the GPU, with exact fp32 products by default, so the scores are the CPU's (`--matmul tf32` or `bf16` scores the model as training computes it) |
| `eval-loss` | a checkpoint's loss on held-out token files, one line per file: by default on the same random windows as `train --val` (`--batch 8 --batches 20`, `night.sh`'s 160 windows; with training's `--batch` and `--eval-batches` it repeats training's last validation number), `--all` on every token once; `--gpu` (with `cxxgpt-gpu`), exact fp32 products by default |
| `self-train` | samples a model's own solutions to training problems and keeps only the ones with the right answer and right arithmetic, as a masked token file (below) |
| `sample` | generates text from a prompt (`--tools` as for `eval-reasoning`); `--gpu` (with `cxxgpt-gpu`) chooses every token on the GPU, the same text as the CPU's |
| `bench-infer` | times the ternary inference engine against an fp32 baseline |
| `leak-check` | finds questions (such as the GSM8K dev set) that share a run of 13 words with training files |
| `qa-context` | how often random training windows train an answer without its question in view (`--data A.tok,B.tok --seq 512`) |

### How answers are scored

An answer is the text after `#### ` up to the end of the line, and it is
compared with the problem's answer as a number, not as text
(`canonical_answer` in `src/reasoning.h`). One number may carry:
- spaces, and commas that group digits (`1,080`);
- a `$` before it (`$57`, `-$28.50`);
- a `%` right after it (`57%`);
- words after a space (`57 dollars`, `57 square feet`), but no word that
  makes it another number or joins it to one (`1 million`, `5 dozen`,
  `57 point 5`, `57 or`, `57 to 60`; the list is `changes_number`);
- one `.` at the very end (`57.`).

Two numbers are equal when they have the same digits once leading zeros,
zeros at the end of the decimals and `-0` are gone: `57`, `57.0`, `57.00`,
`$57` and `57 dollars` all match `57`, `0.8` matches `0.80`, `-28.5`
matches `-28.50`. It is an exact comparison, digit by digit, with no
rounding and no tolerance: `5.7`, `570` and `57.01` do not match `57`, nor
does `-57`. An answer with anything else around its number (`57 or 58`,
`about 57`, `57 (rounded)`), or that is not written as a number (`1/2`,
`.5`, `057`, `fifty-seven`), is compared as text, exactly as written; the
same goes for a gold answer that is not one number. The vote groups tries
the same way, so `$57` and `57.00` are one answer.

Until 2026-10-17 answers were compared as text (only grouping commas
removed). That changed no saved score: the gold answers of GSM8K dev and
test are plain whole numbers, the LibreTexts problems' are already written
the shortest way, and the greedy first tries of night 1, night 2 (GSM8K
test) and rp1 on the LibreTexts problems held no answer the two rules
judge differently.

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
  - `format`: the right number in the answer line, but not alone or not
    with its sign (`18 or 19`, `about 18`, `18` for -18; `$18`, `18.00`
    and `18 dollars` are right, see "How answers are scored");
  - `arithmetic`: a written calculation that is wrong, checked by the
    exact calculator (`480 * 480 = 112,400`; a chain `a = b = c` link by
    link), before any sign of the right number;
  - `passed_gold`: the right number appears in the work, then another
    answer is given (18 dollars a day, then "in a week, 18 * 7 = 126");
  - `other`: wrong with no wrong calculation in sight: a wrong plan or
    goal, for a person to tell.

  The file has a line with the run's settings, then one line per problem
  with its label, every checked calculation, its flags and the text, so
  each label can be audited. The flags count loops (a line written twice,
  or a run of 8 tokens repeated) and numbers from outside the question;
  they fire on right answers too, so the summary reports them for right
  and wrong answers apart. `FILE.jsonl` describes the run without the
  calculator. With `--tools calc-notes`, the run with it is labelled
  too, in `FILE.calc-notes.jsonl` (the same problems, line for line, to
  compare the two problem by problem); there the notes' results are the
  calculator's, so its `arithmetic` labels come from the model's other
  calculations.

### The model's own checked solutions (self-train)

```sh
build/cxxgpt-gpu self-train --gpu --ckpt runs/rp1.ckpt --bpe data/licensed-10b/tok.bpe \
    --problems data/math/fresh.txt --out data/self/rp1-fresh \
    --tries 8 --temp 0.7 --seed 1 --keep 2 --audit data/self/rp1-fresh.audit.jsonl \
    --exclude data/licensed-10b/gsm8k-test.txt,data/licensed-10b/gsm8k-dev.txt
# The same tries checked again, with other settings or newer checks, without a model:
build/cxxgpt self-train --bpe data/licensed-10b/tok.bpe --problems data/math/fresh.txt \
    --recheck data/self/rp1-fresh.audit.jsonl --out data/self/rp1-fresh-text --same text
```

The model tries every problem `--tries` times (the same sampler and seeds
as `eval-reasoning --tries`), and a try is kept only if (`src/self_train.h`):
- it finished its `#### ` line with the right answer, and wrote some work
  before it;
- every calculation in the work is right by the exact calculator, both
  `<<48/2=24>>` notes (and the number written after a note) and plain
  `16 x 7 = 112`, found as `--labels` finds them, sign included (`30 - 35
  = 5` is wrong here, though `--labels` compares numbers without their
  sign). A chain such as `6 * 4 + 2 = 24 + 2 = 26` is checked link by
  link (`6 * 4 + 2 = 24 + 2`, then `24 + 2 = 26`). A right answer through
  a wrong step (`3 + 4 = 8, so 7`) is a lucky answer, and would teach the
  mistake. The number after a note keeps the note's sign: rp1 copies the
  calculator's `<<14-21=-7>>` as `7`, which is rejected. A calculation the
  calculator cannot read (`10 / 3 = 3.33`) is counted, not rejected, but a
  note with no exact decimal (`<<10/3=`, which the calculator leaves to the
  model) must be that value rounded to its places (`3.33`, `3.3`), never a
  whole number: rp1 wrote `<<20/3=10>>` and `<<304/82=4>>` in right tries.
  GSM8K's own notes never have such a value (0 of 23,716), and neither rule
  rejects any of 10,459 gold solutions with notes (GSM8K train and dev, and
  the word, eq and frac dev sets);
- no line of 12 or more characters is written twice (`--allow-loops` keeps
  those);
- it uses no number from outside the question: not in it (in digits or
  words), not a result after `=`, and not one `+ - * /` away from numbers
  already seen (the `--labels` outside flag), apart from 1, 2, 100 and a
  few constants (7, 12, 24, 52, 60, 365, 1000: days a week, months,
  hours, minutes, weeks and days a year, metric units). Numbers the work
  writes in words (`Each box holds thirteen`) are checked too, against
  the question's numbers, the work's own and one step from them. An invented
  quantity is a wrong plan even when the answer comes out right
  (`--allow-outside` keeps those);
- it uses every number of the question (`--allow-unused` keeps those),
  but 1 (the unit in `1 hour each day`), and a percentage may be used as
  its hundredth (`25%` as `0.25`). rp1's lucky answers on the textbook
  problems ignore one: the area of a triangle with base 14 and height 7
  as `7 x 7 = 49`. On GSM8K dev (rp1 after 20 minutes of night 2's mix,
  32 tries), of the right tries that pass the other checks, it drops 20
  of the 35 that reach under half of the gold solution's intermediate
  results (likely lucky) and 8 of the 245 that reach all of them; 8% of
  GSM8K's own train solutions leave a number unused (`a 2008 study`).
  Generated problems add numbers on purpose (a distractor sentence), so
  with `--recipes` the re-draw check replaces this one;
- it is not the problem file's own solution (`--keep-gold` keeps those):
  a model that has learned its training problems, or a generator's
  templates, writes them back, and they teach it nothing new;
- it holds no `[[no-loss]]` marker and no BOS byte.

Of those, a try that is the same solution as an earlier kept one is
dropped, and at most `--keep K` per problem are kept (0, the default: every
distinct one), in try order, so the same seed writes the same files. "The
same solution", for the gold check too, means the same outline by default
(`--same outline`): the same calculations' values in the same order, in
whatever words (`12*4 = 48 pencils` and `The start is 12 x 4 = 48`). So
`2 * 5 = 10` and `5 + 5 = 10` count as the same, which matters little
past one step. Work without a readable calculation is compared as text,
and `--same text` compares only text (spaces around it aside). The files:
- `OUT.txt`: the kept solutions as a problems file (for a person,
  `leak-check` or `eval-reasoning`);
- `OUT.tok`: the same as `prep-qa` writes them, the question masked out
  and the work and `#### ` answer trained on, the answer as the problem
  file gives it (`1200`, where the model may have written `1,200`);
- `OUT.json`: the settings (checkpoint, problems, tokenizer) and counts,
  the record of where the data came from.

The summary gives the problems tried, the first try's accuracy, the
problems any try solved (pass@N) and those with a kept solution, and every
try's verdict (`kept`, `unfinished`, `wrong`, `no_work`, `markers`,
`arithmetic`, `loop`, `outside`, `gold_copy`, `ambiguous`, `unchecked`,
`redraw`, `duplicate`, `over_cap`, `too_long`); `--audit FILE.jsonl` writes
every try's text, verdict and outside numbers (and the re-draw check's
record, below). `--recheck FILE.jsonl` (with the same `--problems`) checks
an audit file's tries again without a model, for other settings or newer
checks; options that decide the sampling are refused with it.
`--tools calc-notes` lets the calculator finish the model's notes while
it writes, as in `eval-reasoning`.

The problems must be training problems: GSM8K train (as `prep-qa` writes
it, without the dev problems), or the train split of a generator, never a
dev or test set. Nothing in a problems file says which split it is, so
pass the held-out sets to `--exclude` (`gen-math`'s rule: a 13-word run,
or the same numbers and answer); excluded problems are counted. Whether
self-generated files join a night's mix, and how much, is the owner's
decision (docs/research-plan.md, "Learning from its own checked
solutions"); keep the original data in the mix too.

First runs (2026-10-17, `runs/rp1.ckpt`, step 58,784):

| problems | tries | first try | pass@N | kept | gold copies | wrong | wrong step | duplicates |
|---|---|---|---|---|---|---|---|---|
| GSM8K train, 16 (CPU) | 4 | 81% | 94% | 3 | 45 | 11 | 5 | 0 |
| `gen-word` train, new seed, 100 (GPU) | 8 | 100% | 100% | 752 | 10 | 2 | 0 | 36 |
| `gen-math` equations, new seed, 100 (GPU) | 8 | 94% | 97% | 1 | 744 | 41 | 14 | 0 |
| `gen-math` fractions, new seed, 100 (GPU) | 8 | 90% | 94% | 1 | 682 | 111 | 6 | 0 |

(`--temp 0.7 --seed 1`; the GSM8K row counted from its audit file, the
gold-copy check came after it, and one of its wrong steps was a right
chain, `6 * 4 + 2 = 24 + 2 = 26`, misread then: chains are checked link
by link since. GPU rows `--matmul bf16 --max-new 400`, about a minute
each.)
- rp1 has learned GSM8K train by heart (81% there, 3.9% on the test set),
  so it writes the gold solutions back.
- rp1's nights train on `gen-word`, `gen-math` equations and fractions, so
  on new problems it writes the generators' own templated text: word for
  word for `gen-math` (one fixed solution per story), in another of the
  sentence patterns for `gen-word` (so these count as kept, though they
  are generator text all the same).
- So, for this model, every source tried gives back text we already have
  or can generate for free, and pass@8 is only 3-4 points above the first
  try on `gen-math`. `self-train` pays where the model solves problems
  whose text is not already in its training: problems with an answer but
  no worked solution, or kinds of problem it has not been taught.

The same tries checked again (`--recheck`) with the outline and outside
checks, the defaults since 2026-10-17 (kept with `--same text` in
brackets, which reproduces the table above):

| problems | kept | gold copies | outside | wrong step |
|---|---|---|---|---|
| GSM8K train, 16 | 0 (3) | 52 | 0 | 1 |
| `gen-word`, 100 | 0 (752) | 798 | 0 | 0 |
| `gen-math` equations, 100 | 0 (1) | 745 | 0 | 14 |
| `gen-math` fractions, 100 | 0 (1) | 683 | 0 | 6 |

- The 752 `gen-word` solutions were the generator's own, in another of its
  sentence patterns: the same calculations in the same order.
- GSM8K's wrong steps fell from 5 to 1 (`30 - 35 = 5`): the others were a
  right chain, read link by link now.
- No right try used an outside number: rp1 follows its templates here. On
  GSM8K test answers of another model (`runs/night2.ckpt`, 15.3% greedy;
  its `--labels` file), outside numbers (constants
  aside) are in 1 of 202 right answers (a `0.10`) and 62 of 1,117 wrong
  ones; without the constants, 6 right answers would be rejected (12, 7,
  60).

Problems with an answer and no worked solution (2026-10-17): 186 of the
383 textbook problems (half A, by the sha256 of the question; half B is
held out to measure), rp1 after 20 minutes of night 2's notes mix,
`--tools calc-notes --tries 8 --temp 0.7 --max-new 400`, GPU, 4 minutes:
22 problems solved by some try (11.8%), 19 tries kept. Read by hand, 5
of the 19 had a wrong step the checks missed then: the minus dropped
after a note (`<<14-21=-7>>7`, 3 tries) or a whole number for a division
with no exact decimal (`<<20/3=10>>`, 2 tries). Both are rejected since
(above), and `--recheck` keeps 14. Of those 14, 8 follow a right plan,
some with a useless line in between (`20 rows * 2 = 40 rows`, then `16 *
20 = 320 tiles`), and 6 reach the book's answer by a wrong plan with
right arithmetic (the area of a triangle as `7 x 7 = 49`, where `14 x 7 /
2` is also 49). The checks cannot tell those apart without a second
version of the problem (the re-draw check, below), which these problems
do not have. None of it is in a mix.

#### How clean are the kept tries?

```sh
build/cxxgpt score-checks                  # tests/checker_labels.jsonl
build/cxxgpt score-checks --allow-unused   # the same, one check off
```

The owner's rule (2026-10-17): self-train trains only on clean solutions.
To measure how far the checks are from that, every try with the right
answer in two 32-try audits of rp1 (20 minutes into night 2's mix, `--temp
0.7 --seed 1 --tools calc-notes`) was labelled by hand, blind to the
checks' verdicts: 546 tries, 217 on 106 of the 383 textbook problems and
329 on 179 of the 500 GSM8K dev problems; tries the
checks call duplicates are left out. The labels (`tests/checker_labels.jsonl`,
written by `operations/build_checker_labels.py`, with each textbook page's
address and CC BY licence; GSM8K is MIT; the tries are our model's own):
- `clean`: every statement true, every calculation right and needed, a
  method that would give the right answer with other numbers;
- `padded`: a sound method, plus something not to train on (a calculation
  never used, a false or nonsense sentence, a number copied wrong);
  `minor` when that is one wrong word or unit (`5 x 2 = 10 miles per
  hour` for miles a day);
- `lucky`: the answer does not follow from a sound method (`7 x 7 = 49`
  for a triangle of base 14 and height 7; `24 + 12 = 36` with the 12
  never worked out).

Of the 546: 27 clean, 93 minor, 108 padded, 318 lucky. Each problem is in a
tune half or a held-out half (the first byte of the sha256 of its question,
odd: held out); new checks are tried on the tune half, and the held-out
half gives the number to believe. `score-checks` runs today's checks on
every try:

| half | kept | clean | clean or minor | clean tries dropped |
|---|---|---|---|---|
| tune | 118 | 9 (7.6%) | 28 (23.7%) | 0 |
| held-out | 111 | 3 (2.7%) | 13 (11.7%) | 0 |

- The checks never drop a clean try, but of what they keep, 2.7% is clean
  on the held-out half (and half of it is lucky: 58 of 111). Each check
  helps: with the loop, outside and unused checks off, 1.5% is clean.
- The arithmetic, loop, outside and unused checks drop 232 tries, all
  padded or lucky (114 by the unused check, 112 of them lucky).
- Most clean GSM8K tries (15 of 17) are gold copies: the same outline as
  GSM8K's own solution, which a sound short solution often is. The model
  never saw these dev solutions, so they count as clean, but self-train
  drops them on purpose (`--keep-gold` keeps them).
- One reader labelled all 546 in one day. 30 tries labelled again blind,
  in random order, got the same label, but the reader remembered many of
  them; a second reader would be a better check.

So today's checks are far from training only on clean solutions: on the
held-out half they keep 36 tries that are not clean for every clean one. Nothing
self-trained goes into a mix until a check set clears a bar fixed in
advance on the held-out half (95% clean was proposed).

#### Stricter checks

```sh
build/cxxgpt score-checks --strict                         # all five
build/cxxgpt score-checks --check-dead --show kept:lucky   # the tune half's kept lucky tries
build/cxxgpt self-train ... --strict                       # the same switches
```

Five more checks, off by default (2026-10-17), each found by reading
the tune half's kept tries (`score-checks --show VERDICT:LABEL` prints
the tune half only, never the held-out one):
- `--strict-outside`: no number one +, -, * or / away from the question's
  numbers (the outside check lets those in, which lets in almost any
  small number); a number must be in the question (a percentage or an
  amount in cents also as its hundredth), 1, 2, 100, a half, 0.01, a
  constant, a result right after "=" (with its sign), or written earlier;
  decimals without their 0 (`.731`) are read;
- `--check-dead` (verdict `dead`): every calculation's result is used
  later, or is the answer (`171 + 380 = 551` and never 551 again); a
  percentage counts as its share (`100 - 25 = 75`, then `.75`);
- `--check-mismatch` (`mismatch`): the calculation written before a note is
  the note's (`.71 x 100 = <<171*100=17100>>`, `60 / 1 = <<60/4=15>>`): the
  same value, or else no number the note leaves out;
- `--check-broken` (`broken`): every note is written
  `<<calculation=result>>` (not `<<19.45>>`, `<<60=60>>` or `45>>45>>45`);
- `--check-computed` (`uncomputed`): some calculation gives the answer,
  or it is written right after an "=" (`x = 2350`).

Each was tried on the tune half, and on GSM8K train's own 7,473 solutions
(the dev problems are not in it), which a check should keep: what it
drops there are mostly clean solutions, sometimes real slips (`6*2.5 =
<<6*3=18>>`, an answer of 66 after a work ending in 64, bare `<<12=12>>`
notes). Kept tries on the tune half, and GSM8K solutions dropped:

| checks | tune kept | clean | clean or minor | clean dropped | GSM8K solutions dropped |
|---|---|---|---|---|---|
| today's | 118 | 9 (7.6%) | 28 (23.7%) | 0 | 948 (12.7%) |
| `--strict-outside` | 99 | 9 (9.1%) | 28 (28.3%) | 0 | 1,812 (24.2%) |
| `--check-dead` | 96 | 9 (9.4%) | 28 (29.2%) | 0 | 1,021 (13.7%) |
| `--check-mismatch` | 110 | 9 (8.2%) | 28 (25.5%) | 0 | 988 (13.2%) |
| `--check-broken` | 109 | 9 (8.3%) | 27 (24.8%) | 0 | 1,210 (16.2%) |
| `--check-computed` | 115 | 9 (7.8%) | 28 (24.3%) | 0 | 989 (13.2%) |
| `--strict` (all five) | 72 | 9 (12.5%) | 27 (37.5%) | 0 | 1,940 (26.0%) |

On the held-out half, `--strict` keeps 59 tries: 3 clean (5.1%, from
2.7%), 13 clean or minor (22.0%, from 11.7%), no clean try dropped. (The
held-out counts were printed once, by `score-checks` without `--show`,
while the checks were being written, but no held-out try was read, and no
check was chosen or changed for them.) Two other checks were tried and
dropped: the same calculation twice (GSM8K writes `2 × $3` for each of two
people's eggs) and a calculation times 1 (`5*1` for one nickel).

What these checks cannot catch is most of what is left: every calculation
right, every number from the question, and a method that has nothing to
do with the story (`15 - 7 = 8`, `8 x 15 = 120`, `120 - 15 = 105` for
seven penalties of 15 yards). No check on the text tells that from a
sound solution, so the 95% bar is out of reach this way; the held-out
half has too few clean tries (3) to measure such a bar anyway. What tells
method from luck is the re-draw check below: the same story with other
numbers. The stricter checks cost a quarter of GSM8K's own solutions,
and a stronger model will write more like them than rp1's short tries do.

#### The re-draw check: right by method, not by luck

The checks above do not see a wrong plan that lands on the right number:
"6 / 2 = 3" (half) where the story needed "6 - 3". A right method stays
right when the question's numbers change; a lucky one almost never does.
So, for generated problems (`src/redraw.h`, docs/research-plan.md item 3):

```sh
build/cxxgpt gen-word --out data/self/word --seed 77 --count 100 --recipes data/self/word.recipes
build/cxxgpt-gpu self-train --gpu --ckpt runs/rp1.ckpt --bpe data/licensed-10b/tok.bpe \
    --problems data/self/word.txt --recipes data/self/word.recipes --out data/self/rp1-word --tries 8
```

1. `gen-word`/`gen-math --recipes FILE` also writes, per problem, how to
   make it again: the generator, its options and the random state its story
   was drawn from (the problems themselves do not change).
2. `self-train` makes each story again with other numbers (`--redraws 3`
   versions): the same people, sentences and steps, new numbers, and the
   generator's own exact answer. The generators tell their two kinds of
   random choices apart (`Choices`, `src/wordgen.h`): what the story is,
   recorded and played back, and the numbers, drawn again. A version
   counts only if its question and its worked solution are the original's
   once every number is masked (`Maya has # boxes of # cards`), so the plan
   is the same step for step, and, for `gen-math`, the topic's second solver
   accepts it.
3. The try's calculations become a formula over the question's numbers:
   each number in a calculation is a number of the question (`q3`), an
   earlier result (whose formula takes its place) or a constant (1, 2,
   100, a number the question gives in words, or one from nowhere else).
   For "40 * 25 / 100 = 10, 40 - 10 = 30" it is `q1 - (q1 * q2 / 100)`.
4. The formula is worked out exactly with each version's numbers, and the
   try is kept only if it gives every version's answer (`redraw` if not).

Constants stay the same in every version: a real one (100 for a
percentage) passes, and a number the model worked out without writing it
down ("12" for 3 boxes of 4) fails. (The outside check runs first and
already drops most invented numbers, `0.75` for 25% off; the re-draw check
also catches the ones it allows, such as a 12 or a 7 that only happens to
be right.) A number with two possible sources (two
question numbers alike, a question number equal to an earlier result, a
question number that is also 1, 2 or 100, or two calculations giving the
answer) leaves the try **ambiguous**: each choice of sources is a
reading, and the try passes if one reading gives every version's answer
(the default since 2026-10-17, the owner's decision: with one reading only,
a third to two thirds of right tries would be dropped).
`--strict-readings` drops every ambiguous try instead; the summary counts
the ambiguous tries that one reading passes either way. A try with no formula for
its answer (an answer stated in prose) or a problem that could not be drawn
again is `unchecked`: dropped, or kept with `--keep-unchecked`. A problem
without a recipe (GSM8K train) is not checked, and its kept tries are
counted. The audit file records, per try, the formula, its constants, each
version's question and answer and what the formula gave.

On the generators' own solutions (300 problems per topic,
`tests/test_redraw.h`): with one reading, 108 of 300 word problems, 167 of
296 equations and 202 of 300 fractions pass, the rest being ambiguous
(their questions repeat small numbers); with every reading tried, all
word problems and equations pass, and 299 of 300 fractions (one with more
combinations than are followed). Without their `<<...>>` notes (as `--notes
off` writes them, and as rp1 writes) they pass as often. Number theory
(80 problems per split) gets three versions of 80 of 80, but only 10 (train)
and 6 (test) of its gold solutions pass: most write a number no calculation
makes (see
"Generated math problems: number theory").

On rp1 (2026-10-17, `runs/rp1.night1.ckpt`, 100 new problems per topic, 8
tries, `--temp 0.7 --seed 1`, then `--recheck` with `--keep-gold`, so that
every right try reaches the check):

| problems | right tries | redraw | ambiguous, `--strict-readings` | ambiguous (default) | unchecked |
|---|---|---|---|---|---|
| `gen-word` | 793 | 0 | 534 | 6 | 0 |
| `gen-math` equations | 741 | 0 | 336 | 8 | 32 |
| `gen-math` fractions | 629 | 0 | 218 | 0 | 0 |

- No lucky try: rp1 writes the generators' own solutions (by default they
  are all `gold_copy`), so the run shows what the check costs on right
  work, not what it catches; the tests show what it catches.
- What it cost on the first run, all fixed since: 195 fraction tries
  failed because "30% of 270 is 270*30/100 = 81" was not read as a
  calculation (the labels read it now: `labels::detail::after_word`), and
  6 word tries because a result with too many readings was taken for a
  constant (it makes the try ambiguous now).
- With one reading (`--strict-readings`), 35-67% of the right tries are
  ambiguous; with every reading tried (the default), 0-1%. The 32 unchecked tries are 4 age problems whose
  plan ("3 - 2 = 1, so x + 16 = 32") holds for few other numbers: no 3
  versions in 200 draws each.

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
- `--split train|test`: 108 training sentence patterns or 54 held-out test
  patterns, grouped into 27 arithmetic situations. The opening patterns
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

An audit (2026-10-16) read the stories as a person would and added rules,
each checked by the tests on every generated problem:
- no "1 cards" or "1 dollars": no step leaves exactly 0 or 1, and added
  amounts start at 2;
- a payment is what a shopper would hand over: the next multiple of $5,
  $10, $20, $50 or $100 above the bill, at most about twice the bill (not
  $180 for two stickers at $0.75);
- extra kept items say where they come from ("A friend gives Maya 4 more
  cards, and Maya keeps them", not "Maya keeps 6 green cards");
- a running quantity is cut down (an equal share, a fraction or a
  percentage kept) at most twice and never twice in a row, and is sold or
  bought at most once;
- no wording is used twice in one story: extra items, purchases and kept
  items each take turns among three sentence families (twelve wordings for
  train, six for test), and a running quantity gets at most six gifts and
  gives some away at most twice.

These generated stories are still less varied than human-written math:
at 7-8 steps, each of the five shapes other than the running quantity is
one multiplication (or fraction, or percentage) followed by five or six
additions, each worded differently. Success on its held-out wording alone is not evidence of improvement on
GSM8K or SVAMP. No generated datasets are checked into the repository, and
this command does not change the night data mix.

### Generated math problems: equations

```sh
./cxxgpt gen-math --topic equations --out data/math/eq-train --split train --seed 1 --count 100000 \
    --exclude data/licensed-10b/gsm8k-test.txt,data/licensed-10b/gsm8k-dev.txt
./cxxgpt gen-math --topic equations --out data/math/eq-test --split test --seed 2 --count 1000 \
    --exclude data/licensed-10b/gsm8k-test.txt,data/licensed-10b/gsm8k-dev.txt
```

`gen-math` (`src/mathgen.h`) writes the same two files as `gen-word`
(`PREFIX.txt` for `eval-reasoning`, `PREFIX.text` for `text2tok`), with the
task `equations` and, as the difficulty, the number of calculator steps in
the solution (2-7). Its stories are the ones GSM8K solves with one unknown:

- a number puzzle ("thinks of a number, multiplies it by 3, adds 7 and gets
  31"), solved by undoing the steps from the end;
- two ages that add up to a total, or that keep a ratio in some years;
- two kinds of one thing: adult and child tickets, cows and chickens (their
  legs), large and small packs;
- consecutive whole, even or odd numbers with a given sum;
- a rectangle whose length depends on its width, from its perimeter (asking
  for the width, the length or the area);
- three people sharing money ("Leo gets $5 more than Maya, Zoe gets 3 times
  as much as Leo").

The solution writes the equation (`Let x be Leo's age. Then Maya is 3x + 5
years old.`) and solves it one step at a time, each number step a calculator
note: `Taking 5 from both sides: 41-5 = <<41-5=36>>36, so 4x = 36.` The
equation itself stays outside the notes, so the calculator never reads
`3x` as a multiplication.

How every problem is checked before it is written:

1. The unknown is drawn first, then the story is built around it, in whole
   numbers; a story that does not come out whole is drawn again, never
   rounded.
2. A second solver that shares no code with the generator solves the story
   again from its numbers alone, by trying every whole number the story
   allows (each possible age, ticket count, width), and must find exactly
   one that fits: the answer written. The number puzzle is checked another
   way: its steps are combined into one formula and solved backward.
3. Every calculator note is worked out again with the exact calculator
   (`src/calc.h`).

If any check fails, the program stops instead of writing the problem. The
tests also show the checker rejects broken problems (a wrong answer, a
changed number, a wrong note).

`--exclude FILES` (comma-separated problems files or one question per line)
keeps the output clear of evaluation questions. These stories are classic,
and without it a large file repeats a few GSM8K test questions almost word
for word, numbers included ("Ruby is 6 times older than Sam. In 9 years,
Ruby will be 3 times as old as Sam.", or a farm of only sheep and geese
with 20 heads and 70 legs). A problem is drawn again when its question shares a 13-word run
with an excluded question (`leak-check`'s rule), or has the same numbers and
the same answer, which also catches the same problem in other words. Of
300,000 problems, about 700 are drawn again this way; afterwards none of
GSM8K test or dev shares a 13-word run, or its numbers and answer, with the
output. Without `--exclude` the command says so. The two files above come
with the RunPod data (docs/PREPARED_DATA.md); from a fresh clone, make the
same questions from GSM8K's JSON Lines (`scripts/get_data.py`): `./cxxgpt
prep-qa --in data/gsm8k/test.jsonl --out data/gsm8k/test --task gsm8k`
writes `data/gsm8k/test.txt`, and the same for `dev.jsonl`.

Other options are `gen-word`'s: `--seed`, `--count`, `--out`, `--split
train|test` (four of six wordings per sentence for train, the other two for
test), `--distractors on|off` and `--notes on|off`; none of the last two
changes a random draw. Kinds with few different problems are drawn less
often. In 1M problems: puzzles 176k different, mixtures 131k, shares 107k,
rectangles 46k (at real sizes: a rug of a few meters, a field of up to 150),
consecutive numbers 7.2k, ages adding up 6.0k, ages in some years only 564
(realistic ages leave few), so those repeat with other names
and wordings. `operations/build_gen_math.py` builds `data/gen-math` for
training: 500,000 problems of each topic and a 1,000-problem dev set from
the held-out wordings, without calculator notes (like the word problems),
checked again in Python against GSM8K test and dev, and tokenized with the
RunPod model's tokenizer; its `manifest.json` records the seeds, counts and
generator commit.

### Generated math problems: fractions, percentages, ratios and units

```sh
./cxxgpt gen-math --topic fractions --out data/math/frac-train --split train --seed 1 --count 100000 \
    --exclude data/licensed-10b/gsm8k-test.txt,data/licensed-10b/gsm8k-dev.txt
```

`--topic fractions` (`src/mathgen_fractions.h`) has the same files,
options, splits and checks as the equations, with the task `fractions`.
Its stories:

- parts of an amount, one to three steps, as fractions or percentages:
  "has 240 apples, sells 25% of the apples, then gives 1/3 of the remaining
  apples to Leo";
- a price cut or raised by a percentage, once or twice, in exact cents;
- what percent: "18 of the 24 questions right", or a price that went from
  $40 to $50;
- an amount shared in a ratio (3:5, or 2:3:4 among three people), or a
  class of boys and girls, asking for a share or a difference;
- a ratio and one count ("cats to dogs is 2:3, and there are 18 dogs"),
  asking for the other count or the total;
- the same rate for another amount: prices for more or fewer items, a
  recipe for more or fewer cookies;
- units: minutes to hours, grams to kilograms, kilometers to meters, cents
  to dollars.

The numbers are drawn so that every step comes out exact (a part of an
amount is always whole, a price change always whole cents); every answer is
a whole number or a decimal with at most two places, as in GSM8K, and
nothing is rounded. Money and measurements are kept in hundredths inside
the program. The checker (`fraction_check`) shares no code with the
generator: it works each story out again from its numbers (the parts as one
product, prices in cents, every whole percent or ratio part size tried in
turn), and calc.h checks every note.

No generated question shares a 13-word run with GSM8K test or dev, but
simple stories land on the same numbers as some of GSM8K's easiest
questions (a $100 ticket down 20%; 20 cookies, 2/5 given away), so use
`--exclude` here too: of 300,000 problems it draws about 260 again, after
which none has a GSM8K test or dev question's numbers and answer. In 1M
problems: price changes 98k different, proportions 73k, parts 62k, ratio
splits 61k, units 16k, ratio-and-count 13k, what-percent 6.6k (drawn
least often). Prices, weights, distances and counts stay in ranges a real
thing has (a sofa $300-1,500, a lamp $20-150; a jar of honey 100-400 g; a
walk or a drive of up to 8 km a day).

### Generated math problems: number theory

```sh
./cxxgpt gen-math --topic number-theory --out data/math/nt-train --split train --seed 1 --count 100000 \
    --exclude data/licensed-10b/gsm8k-test.txt,data/licensed-10b/gsm8k-dev.txt
```

`--topic number-theory` (`src/mathgen_number_theory.h`, 2026-10-17) has
the same files, options, splits and checks as the other two topics, with
the task `number-theory`. Every answer is a whole number. Its stories:

- remainders: 50 cookies packed in boxes of 6, or 111 students in vans of
  13, asking for the full boxes, what is left over, the boxes needed for
  all, or how many more would fill one more box;
- least common multiples: two buses (lights, visits) repeating every 12 and
  18 minutes, asking when they next meet or how many times in the next few
  hours; hot dogs in packs of 8 and buns in packs of 6, asking for the
  fewest packs for equal numbers;
- greatest common divisors: identical gift bags from 24 pencils and 36
  erasers, or two ribbons cut into the longest equal pieces;
- factors: 36 chairs in equal rows, at least 2 (or 3) rows of at least 2
  (3) chairs: how many numbers of rows are possible?
- multiples: raffle tickets 1 to 200 that are multiples of 7, every 5th
  customer, every 4th and every 6th person (who gets both?), the multiples
  of 7 from 45 to 180, the even- or odd-numbered lockers from 18 to 210;
- digits: the digits needed to number a book's pages, the pages that 342
  digits number, a two-digit number from its digit sum and difference;
- repeating patterns: 3 red beads then 2 blue, or 5 work days then 2 days
  off: how many of each in the first n?

The exact calculator (`src/calc.h`) knows only `+ - * /`, so a whole-number
quotient is written as two products around the number, as a pupil checks
it: `7*28 = <<7*28=196>>196 and 7*29 = <<7*29=203>>203 is more than 200,
so 28 tickets win.` A common multiple is found by counting up by the larger
number (`Counting up by 12: 12, 24.`), a common divisor from the factors of
the smaller one. The checker (`number_theory_check`) shares no code with
the generator and works every story out by brute force: it fills boxes one
at a time, tries every time until both buses come round, every common
divisor, every number of rows, every ticket, every page, every two-digit
number, and lays the beads down one by one. calc.h checks every note.

Wordings, so that the text is less of a template:

- the story: 20 sentence families of six wordings, 120 patterns, 40 held
  out for the test split;
- the question and the main solution lines (the two products, counting
  up, the factors, the largest common factor): 37 phrases of three
  wordings each, two for train and one held out ("How many trays can Leo
  fill completely?", "How many full trays can Leo make?"; "... is more
  than 50", "..., while ... is already more than 50"). The other solution
  lines have one wording.
- fillers (irrelevant sentences): this topic has its own 24, 12 per split,
  about the story's people ("Maya is 12 years old.", "Leo has a dog named
  Biscuit."), half of them with a number the question does not need, and
  only half the problems get one. gen-word's shared fillers are not used
  here: they name numbered things ("The office door is numbered 14", "route
  9") that would read as part of a story about numbered tickets or
  lockers, and every equations or fractions problem has one, so each of
  them repeats in about 8% of problems; here no filler is in more than 4.4%.

The solutions are still written to a small set of plans (one per kind and
question), and each story is one or two sentences around one idea, where
GSM8K usually chains several: a step towards less templated data, not the
end of it.

In 1M problems: multiples 110k different stories, remainders 26k, cycles
25k, common divisors 7.9k, patterns 6.5k, digits 2.0k and factors only
about 300 (a number of chairs up to 200 has few factorings), so factors
are drawn least (1 in 33) and repeat with other names and wordings.

No generated question shares a 13-word run with GSM8K test or dev. Small
stories land on the numbers and answer of some of GSM8K's questions by
chance (20 campers in tents of 6 need 4 tents, and a GSM8K question with
the numbers 6 and 20 has the answer 4; without `--exclude`, 25 GSM8K test
and dev questions in 839 of 300,000 problems), so use `--exclude`: of
300,000 problems it draws 884 again, after which none of GSM8K test or dev
shares a 13-word run, or its numbers and answer, with the output (checked
again in Python, with the filler taken out). Not in any training mix: the
owner decides.

A hand check of 30 problems (seed 2026, train) and 15 (seed 77, test): all
45 answers and solutions right; one train wording asked "How many trays
does Leo need?" after a sentence that named nobody, in 3 of the 30. That
wording now names the person, and a test checks that a name in the
question is in the story. After the new wordings and fillers, 22 more
(seed 4242, train, and 4243, test): all right.

`--recipes FILE` works as for the other topics, and self-train's re-draw
check makes three versions of almost every problem. But most of these
solutions do not pass it (below, "The re-draw check"): a quotient found by
trying products (the 28 above), a common multiple found by counting up, or
the first even locker is written as a number no calculation makes, which
the formula keeps as a constant, so on other numbers it gives another
answer. Only solutions made of plain sums pass: the page digits, the factor
counts and some common multiples. So for this topic the re-draw check
would reject most right tries; it needs a calculator that writes whole
quotients and remainders before it can judge them.

### Real data and GPU training

```sh
scripts/probe.sh                     # GPU speed and memory per model size, and the
                                     #   night.sh command for each
TOKS=... MODEL=... scripts/night.sh  # one night of training, checked on GSM8K
python3 scripts/dashboard.py rp1     # watch it live in the browser (docs/LAPTOP.md)
scripts/report.sh rp1                # the night on one screen
scripts/ledger.sh rp1 "note"         # its row in docs/runs.md, the ledger of runs
```

The data comes from the RunPod run (below and
[docs/PREPARED_DATA.md](docs/PREPARED_DATA.md)), with its tokenizer.
[docs/LAPTOP.md](docs/LAPTOP.md) has the whole setup, from a fresh Windows
laptop to a night of training, and how a night continues the RunPod model.

Settings are chosen on a dev set, not on the GSM8K test set: `get_data.py`
holds out 500 GSM8K train problems (`data/gsm8k/dev.jsonl`, the 500 with the
smallest sha256 of their question) and removes them from training.
`DEV=1 scripts/night.sh` validates on them.

**The training data (since 2026-10-16): the RunPod model's.** The RunPod
run (docs/RUNPOD_HANDOFF_2026-10-16.md) trained a 267M-parameter MoE (97.6M
active per token) from random weights, and the laptop continues it on the
same data, with the same tokenizer. A copy of the data is on the owner's
Google Drive (docs/PREPARED_DATA.md); on the laptop it lives in two
folders:
- `data/licensed-10b/`: 10.0B tokens of openly licensed text from the
  Common Pile (`scripts/build_licensed_corpus.py`, pinned revisions in
  `configs/licensed-10b-sources.json`): arXiv papers 5.87B, PubMed papers
  3.68B, DOAB books 435M, LibreTexts/OpenStax textbooks 15M. Each shard has
  train, val and test files, and every document's licence and attribution
  is kept. Also the tokenizer, `tok.bpe` (16,384 tokens, learned from a
  sample of the training split; sha256 `6e811ced...`), and GSM8K's dev and
  test sets encoded with it (`gsm8k-dev`, `gsm8k-test`).
- `data/targeted-math/`: 973,193 exact `gen-word` problems (1M generated
  after the 2026-10-16 audit, same seeds, minus 25,646 sharing a 13-word
  run with a GSM8K dev or test question or a generated dev question, and
  1,161 with a GSM8K dev or test question's numbers and answer;
  `operations/build_targeted_math.py`, 199.6M tokens; every one of its
  4,849,161 written calculations re-checked exactly), and GSM8K train
  without the 500 dev problems and 7 near-copies (6,966 problems). The
  RunPod model trained on the set made before the audit (963,089 problems,
  `src/wordgen.h` at commit 3b6d880, with its flaws: "1 cards", payments
  of $180 for $1.50, kept items from nowhere, a sentence repeated in 44%
  of problems); it is kept as `data/targeted-math-runpod/`.

`night.sh` trains on the run's last mix unless `MIX=` names others:

| Files | What | Tokens | In the mix |
|---|---|---|---|
| `targeted-math/word-train` | exact word problems | 193M | x4 |
| `targeted-math/gsm8k-train` | human-written GSM8K train problems | 1.2M | x64 |
| 12 `licensed-10b/blocks/*/train-*` shards | textbooks, DOAB books (all of them), arXiv papers (3 of 64 files) | 750M | x1 |

That is 1.6B tokens a pass, about 53% worked math. The math share is
by sampling weight; answers are the only trained tokens of a problem. The
500 dev problems (`DEV=1`) and the 1,319 test problems are the same ones
`scripts/get_data.py` holds out (the same files, byte for byte), and the
model has seen neither. The run chose its checkpoints on the dev set only,
and scored the test set once at the end.

The RunPod model's checkpoints, in `runs/runpod-2026-10-16/`:
`fresh-eu.ckpt` (step 46,946, with the optimizer, for continuing it; 17/500
dev, 51/1319 = 3.9% test, greedy) and `best-math-inference.ckpt` (the best
on dev, from about step 45,000, weights only: 20/500 dev, 47/1319 test).

**Data provenance (2026-10-16).** What the models learn from, and why: a
model meant to be published should depend only on openly licensed text,
human-written problems, our own programs and our own models, never on text
another company's AI model (or any model under a restrictive licence)
wrote. Our own models' output is allowed (owner's decision, 2026-10-16)
when the model that wrote it trained only on clean data itself, such as
`self-train` files from the RunPod model or the nights that continue it;
each such file's `OUT.json` names the checkpoint that wrote it.

What trains the models now (`night.sh`'s default mix, above):
- **Common Pile:** openly licensed text (open textbooks, DOAB books, arXiv
  and PubMed papers), from pinned revisions, CC BY 4.0 or CC0 only, with
  each document's licence and attribution kept.
- **`gen-word` word problems:** written by our own C++ program from
  templates and seeds, with exact answers. No AI model writes them.
- **GSM8K:** human-written problems and solutions, MIT licence.

Not in the mix yet: `self-train` solutions (the model's own, checked; the
owner decides when and how much).

Which checkpoints are clean to publish: **the RunPod model** (2026-10-16)
and the nights that continue it. They trained on the data above only, from
random weights, with a tokenizer learned from the licensed corpus.

This is a record of where the data came from, not legal advice. Before a
public or commercial release of weights, check the attribution each
licence asks for (Common Pile's per-document licences, GSM8K's MIT
notice).

**Textbook word problems with answer keys (2026-10-17).**
`operations/build_libretexts_problems.py` takes word problems and their
book answers from the CC BY math textbooks in Common Pile's copy of
LibreTexts (OpenStax's Prealgebra 1e, Intermediate Algebra 1e, Precalculus
1e and others, and course remixes of them), into
`data/libretexts-problems/`: 383 problems (JSON Lines with each problem's
page, book and licence; `problems.txt` for `eval-reasoning` and
`self-train`), none sharing a 13-word run, or its numbers and answer, with
a GSM8K dev or test question.
- **Licence.** OpenStax's math books were CC BY 4.0 until OpenStax
  relicensed them CC BY-NC-SA 4.0 in 2026 (the `osbooks-*` repositories'
  LICENSE files on 2026-03-12 to 03-19, their licence lines on
  2026-04-23). LibreTexts copied them while they were
  CC BY, and a CC licence cannot be taken back from copies already made,
  so the LibreTexts copies are what the script reads, never openstax.org's
  current books. Share-alike and non-commercial pages are left out.
- **What is kept.** Only an exercise of one paragraph, followed directly by
  an answer that is one number ("58 hardback books" is 58), is kept.
  Anything unclear is dropped: two answers, fractions, a figure or table
  the text lost, two exercises sharing one answer. A wrong pairing would go
  unseen (nothing solves these problems again), and
  `tests/test_libretexts_problems.py` checks each rule. Most exercises do
  not qualify (algebra drills, graphs, several parts), hence the small
  count.
- **Audit.** 100 problems were worked out by hand, all right in the final
  set. The draw before the last fixes had 5 bad ones, 3 of them wrong
  keys in the books themselves (1161 for 1126, 14 for 34); those are
  excluded by name. Expect a few percent of such book errors among the
  rest. Some keys round without the question saying how (33.33, 2.2), and
  the scorer compares the text after `####` exactly.
- **Use.** There are answers but no worked solutions, so the set is for
  checking the model's own solutions (`self-train`) and for evaluation,
  not for the training mix as it is. LibreTexts is in the training data
  (`textbooks-libretexts`), so a score on it is not a held-out test.
- **Second build (2026-10-17), `data/libretexts-problems-2/`: 475
  problems, 93 of them new.** The builder now also reads answers written
  "- Answer:" or on the "- 340 yd" line itself, fractions and degrees in a
  question ("3 1/2 miles", "27°"), "per" in an answer ("60 miles per
  hour"), a full stop after the number ("The number is 4."), and the
  nursing books' dosage calculations (OpenStax's Pharmacology for Nurses,
  29 problems: "The client weighs 176 lb. How many milligrams ...?").
  Each problem says whether it is `new` against the first build
  (`--previous`); one old problem is gone ("find the equation of the
  line", key 9). 50 new problems were worked out by hand: 4 bad in the
  first draw (8%: a question asking for an equation, one naming
  "problem 78", two wrong book keys), fixed by two rules and excluded by
  name; all 50 right after. 0 of GSM8K test and dev share a 13-word run
  with it (`cxxgpt leak-check`).
- **Third build (2026-10-17), `data/libretexts-problems-3/`: 469
  problems, none new.** OpenStax's own 2e books
  (`operations/build_openstax_problems.py`, PR #80) give a second answer
  key for the problems both editions have. Where the keys differ, four
  LibreTexts keys are wrong: a coffee tip ($0.35 on $1.65: 17.5% for
  21.2%), tickets varying inversely with price (97 for 9), a bacteria
  decay (5,870,061 for about 47,700) and Rachael's shake (1161 for 1126;
  the first build's exclusion was written "21-ounce", the copy kept says
  "21 ounce", so exclusions now ignore punctuation and case). Two more
  have two right answers, depending on whose clock (David's 11 hours of
  driving or 12 from John's start; 1.5 or 1.83 hours): dropped as
  unclear. These six are the only change from the second build.
- **Why not thousands.** A survey of the whole LibreTexts file (18,099
  math, statistics and medicine pages): about 44,000 answer keys, of which
  some 1,000 belong to a one-paragraph word problem with a one-number
  answer (half of them copies across course remixes), and the rest are
  algebra drills, graphs, several parts or LaTeX. Worked examples with a
  final "Answer the question" step add a few dozen; chemistry and physics
  answers need constants and significant figures. The OpenStax books'
  back-of-book answer keys (odd exercises) are not in LibreTexts' copies,
  and `data/licensed-10b` holds no raw text (token files only; DOAB's math
  titles are research books). Thousands more would need another source;
  that is the owner's call on licences.
- **OpenStax's own sources (2026-10-17), `data/openstax-problems/`: 233
  problems, none in the LibreTexts sets.** OpenStax keeps its books on
  GitHub (`github.com/openstax/osbooks-*`) as CNXML, and there many
  exercises carry their answer inline ("There were 308 students.").
  `operations/build_openstax_problems.py` reads seven repositories
  (Prealgebra 2e, Elementary and Intermediate Algebra 2e, College Algebra
  2e, Algebra and Trigonometry 2e, Precalculus 2e, Contemporary Mathematics,
  Introductory and Introductory Business Statistics 2e, Statistics,
  Principles of Finance, Algebra 1), each at the last commit before its
  LICENSE file became CC BY-NC-SA (pinned in the script, which checks that
  LICENSE and every book's `md:license` say CC BY 4.0 there). The owner
  decided on 2026-10-17 that such a commit counts as CC BY for this
  project; never read a later one. `--fetch` downloads only the CNXML and
  the book lists (one commit deep, no images, about 90 MB) into
  `data/openstax-raw/`. Each problem keeps its repository, commit, module,
  books and licence, and a link to the file at that commit.
  - The same rules as the LibreTexts builder, plus: the problem and its
    answer are one paragraph each, holding only text and plain numbers in
    math (a variable, a power, a list, a figure or a link drops it); the
    instructions before a group of exercises point at no table,
    distribution or other exercise; the answer is no bound ("More than
    $42,857.14"); a rounding instruction of the group is added to the
    question; compound interest, growth, probability or trigonometry with
    a decimal answer needs a stated rounding (11.6 years for 11.58...);
    circles and cones need the question to say 3.14 or pi (Prealgebra
    keys them with 3.14). `tests/test_openstax_problems.py` checks each
    rule.
  - Of 52,250 exercises, 32,258 have an answer; 710 pass. 291 of those
    are already in the LibreTexts sets (LibreTexts holds the first
    editions; same numbers and most words, whatever the key) and are
    dropped, 166 are copies within OpenStax (one of them disagreeing on
    the answer), and 20 fail the exclusions, the radians check or the
    GSM8K dev/test filters: 233 are left, 217 of them from the Prealgebra
    bundle. Algebra 1's quiz layout ("Correct! ... The
    correct answer is 160") is not read (about 20 more, mostly copies of
    Elementary Algebra's); Contemporary Mathematics and Principles of
    Finance key only their worked examples.
  - Audit: 50 problems worked out by hand. The first draw had 4 bad (8%:
    a cone keyed with 3.14 unsaid, a probability needing the instructions'
    distribution, two "average rate of return for the values that follow",
    one of whose series keys is wrong) and 3 rounded without saying (11.6,
    0.03, 0.648); four rules and one exclusion followed. After them all 50
    keys are right, with 3 caveats: 21.57 rounded without saying, 91 in
    centimeters the question does not name, 1921 for a growth taken as
    exponential. 0 of GSM8K test and dev share a 13-word run.
  - The 291 copies in LibreTexts cross-check both: 8 have another key.
    Four LibreTexts keys are wrong (MaryAnne's tip 17.5 for 21.2,
    Madelyn's tickets 97 for 9, the bacteria decay 5870061 for about
    47,700, and the "21 ounce" Rachael shake 1161 for 1126, a wording
    EXCLUDED misses), two ask whose travel time (11 or 12 hours, 1.5 or
    1.83), one is another question (median, not mean), and one OpenStax 2e
    key is wrong (Christa's fence, 240 feet for 24).
  - Use: as the LibreTexts sets, for evaluation and self-train only, and
    not a held-out test (the first editions are in the training data).

**GSM8K train with new numbers (2026-10-17, `cxxgpt redraw-qa`).** GSM8K
train's human questions, each with its numbers swapped for new ones and an
exact worked solution and answer for them (`src/redraw_qa.h`,
[docs/redraw-qa.md](docs/redraw-qa.md)): the model has learned the gold
solutions by heart, but cannot recite the answer to a question it has seen
with other numbers. The gold solution's calculator notes become a formula
over the question's numbers (self-train's re-draw code); a problem is kept
only if every number has one clear source, and every new result keeps its
kind (whole, positive, at most two decimals). Every variant is checked
again by calc.h and by self-train's re-draw check.
- **Yield:** 3,642 of 6,966 problems (52.3%) from
  `data/gsm8k-notes/gsm8k-train.txt` (dev and test held out; the tool
  refuses a dev or test file); 10,803 variants with `--versions 3`. The
  rest are mostly arithmetic done in the head (a constant from nowhere,
  1,085) and numbers that could stand for two things (932).
- **Leak check:** 0 of GSM8K dev and test share a 13-word run with the
  variants, or the same question with numbers masked.
- **Hand checks:** three rounds of 30. Each round's findings became a rule
  ("$40 bills"; "x 5 days" for weekdays taken for Saturday's "5 hours", a
  wrong answer; "an 8 day work week"; "a gallon is 15 pints"); the last
  round had 30 of 30 answers right (docs/results/redraw-qa-check.md; 60
  variants from a last rule, 0.01 for percent, came after it, 3 of them read). Odd
  but right stories remain ("245 campers last week"), and so do GSM8K's
  own misreadings.
- **Use:** not in a mix (the owner decides). `prep-qa --keep-calc` makes
  tokens of `OUT.jsonl`.

To see whether math training costs the model its general text, score a
checkpoint on held-out text it never trained on, the corpus's validation
files (encoded with `data/licensed-10b/tok.bpe`, so only for models with
that tokenizer). The RunPod run measured eight batches of the DOAB ones;
for the textbooks, DOAB and the first arXiv and PubMed shards:

```
V=data/licensed-10b/blocks
build/cxxgpt-gpu eval-loss --gpu --all --ckpt runs/rp1.ckpt --data $V/textbooks-libretexts-0000/val-0000.tok,$V/doab-doab-0000/val-0000.tok,$V/arxiv-arxiv-papers-0000/val-0000.tok,$V/pubmed-licensed_pubmed-0000/val-0000.tok
```

Lower is better; compare checkpoints of the same model, with the same
settings.

**Answers with their questions.** Training windows start at random
offsets, so a window often begins in the middle of a math problem: the
answer is trained while its question is out of view. `./cxxgpt qa-context
--data data/targeted-math/gsm8k-train.tok,data/targeted-math/word-train.tok`
counts it exactly. `train --sampler qa` (`scripts/night.sh --sampler qa`)
moves such a window back to the start of its problem, when the whole
problem fits in a window. At `--seq 512`, the share of answer tokens that
see their whole question:

| Source | `--sampler random` (default) | `--sampler qa` |
|---|---|---|
| GSM8K | 74% | 100% |

Problems longer than a window keep random starts, or the end of their
answers would never be trained. Windows that start with a question train
fewer answer tokens each (GSM8K: 277 instead of 297 per window). Plain text
is not affected, and validation keeps random windows. Whether it helps accuracy
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
`test_gpu` checks against the model in double precision on the GPU
(`src/gpu_ref.cu`).

How often it fires (share of trained tokens with a token to push down,
windows of 512):

| N | GSM8K |
|---|---|
| 6 | 2.2% |
| 8 | 0.60% |
| 12 | 0.08% |

A first CPU run (4 layers of 128, worked math problems, 1,500 steps, N = 6, same seed
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

(Measured on the curriculum before 2026-10-16. Since then `gen-reasoning`
writes no degenerate zeros: no chain variable is 0, no add operand is 0,
eq's x is at least 1, and expr never computes a-a. About 10% of chain
answers used to be 0. Runs made after that date are not comparable with
these numbers; repeat the baselines first.)

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
