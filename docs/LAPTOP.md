# Training on the laptop

The laptop (RTX 4070 Laptop GPU with 8 GB, 16 threads, 32 GB RAM) trains
overnight natively on Windows. (Until 2026-10-15 it trained under WSL2,
the Linux inside Windows; native Windows trains the same model to the same
loss, 7% faster and without WSL's slow stretches, and leaves WSL's memory
to Windows: `review/baselines/` has both curves.) The scripts run in Git
Bash, the shell that comes with Git for Windows. Setting it up takes
about an hour, once.

## One-time setup

1. **Update the NVIDIA driver**, with the NVIDIA app or from nvidia.com.
   Then, in the NVIDIA Control Panel (Manage 3D settings), set **CUDA -
   Sysmem Fallback Policy** to **Prefer No Sysmem Fallback**. Otherwise a
   model too big for the GPU's 8 GB does not fail: the driver spills into
   system RAM and training runs several times slower (d768-moe at batch 8:
   2,900 tokens/s, against 17,800 for d768-dense, which does the same work
   per token). `probe.sh` also halves the batch of any size that fills more
   than 90% of the GPU; use the batch it prints.
2. **Install the tools** (each once; `winget` is Windows' own installer, in
   PowerShell):
   - Git for Windows, which brings Git Bash: `winget install Git.Git`;
   - Visual Studio Build Tools 2022 with the **Desktop development with
     C++** workload (MSVC, the Windows SDK, CMake and Ninja), from
     visualstudio.microsoft.com;
   - LLVM, for clang-cl: `winget install LLVM.LLVM` (or the Build Tools'
     "C++ Clang tools" component; Visual Studio's installer refuses to
     change the Build Tools while any build is running);
   - the CUDA Toolkit (13.x), from developer.nvidia.com;
   - Python 3, only for downloading data: `winget install Python.Python.3.12`.
3. **Keep the laptop awake at night.** Plugged in: sleep "Never", and
   "When I close the lid: Do nothing" (Control Panel, Power Options). Put
   it on a hard surface so the fans can breathe. In Windows Update, set
   the active hours to cover the night (or pause updates), so it does not
   restart in the middle of a run.
4. Check that the GPU is visible: `nvidia-smi` should list the RTX 4070.
5. Get the code, then build everything and run all tests, including the
   GPU against the CPU (any terminal; in Git Bash:
   `scripts/build_windows.cmd`):
   ```bat
   git clone -b main https://github.com/AndreiAgapie97/CxxGPT.git
   cd CxxGPT
   scripts\build_windows.cmd
   ```
   The programs land in `build\` (`cxxgpt.exe`, `cxxgpt-gpu.exe`,
   `test_gpu.exe`, ...). It should end with `GPU tests passed` and
   `All set`. If anything fails, keep the output; Claude can work from it.
6. For the data scripts, Python 3 (python.org's installer, `python` in
   Git Bash). They need only its standard library.
7. Install Claude Code and start it in the project
   (code.claude.com has the Windows installer), `claude` in the CxxGPT
   folder. To follow and steer that session from the Claude app (phone or
   browser), start it with `claude remote-control` instead.

The commands below are for Git Bash, in the CxxGPT folder; on Windows,
`./cxxgpt` is `build/cxxgpt.exe` and `./cxxgpt-gpu` is
`build/cxxgpt-gpu.exe` (the scripts pick these by themselves).

## Running overnight

- Start the night in a Git Bash window and leave it open: closing the
  window stops the run (run the same command again to resume from the
  last checkpoint, saved every 1000 steps).
- `--time-limit MINUTES` stops training and saves a checkpoint in time for
  the morning. Ctrl-C also saves before stopping. `--resume` continues
  from the checkpoint.
- If training diverges (the loss or gradient becomes infinite or NaN),
  that step is skipped; ten in a row stop the run without saving over
  the last checkpoint. The log then says which step the checkpoint holds;
  resume it with a lower `--lr` (night.sh: `LR=...`).
- `nvidia-smi` in another terminal shows GPU load, memory and temperature.
  `night.sh` also logs it every 10 s to `runs/$RUN.gpu.csv`: time, clock
  (MHz), temperature, power, load, memory, the slowdown reasons (0x1: idle;
  0x4: power cap; 0x8: hardware slowdown; 0x20/0x40: too hot) and the last logged step. When a stretch of the
  night runs slow, those rows say whether the GPU slowed itself down (clock
  and power drop, a thermal reason) or sat idle waiting (load drops, clock
  stays up: another program busy on the CPU, such as a build, or Windows).

### Watching a night

`python3 scripts/dashboard.py` (in a second Git Bash window, from the
repository's folder) opens http://127.0.0.1:8765/ with the run written to
last; `python3 scripts/dashboard.py rp1` picks one. It only reads the files
night.sh writes, so it can be started and stopped at any time, also halfway
through a night, and it never slows the run down (it reads only what was
added, every 10 s). The page shows:
- the state (training, ended, stalled, crashed), the step, the time left
  (the steps still to go at the recent speed, or the time limit) and the
  speed;
- warnings: no new log line for 5 minutes, skipped steps, a loss or
  gradient-norm jump, tokens/s under 80% of the recent speed, an expert
  under 1%, the GPU at 87 °C or slowing itself down, a control file in use;
- charts of the training and validation loss, the GSM8K exam score, the
  validation loss of each part of the mix and of GSM8K's final answers
  (below), the learning rate, the gradient norm, the speed, the smallest expert share
  and the GPU's temperature and power, with "compare with" laying an
  earlier run over them (dashed);
- the last wrong exam answer, each layer's expert shares, the events
  (saves, resumes, skipped steps, warnings), the settings and the log's
  last lines.

One validation number hides which part of the mix improves, so night.sh
also scores held-out files one by one (`train --val-sources`, `VAL_SOURCES`
in night.sh): at the start and with every validation, the GSM8K
validation set, the word problems' dev split (`data/targeted-math/word-dev.tok`)
and a val shard each of textbooks, DOAB books and arXiv papers. A file of
questions and answers gives two numbers: its answers' loss, and its final
answers' (the tokens after `####`, what the exam scores). The log has one
line each time (`val sources at step N`), and `runs/$RUN.val.csv` a row
per number. On the RunPod model it should take about 10 s every 500 steps
(an estimate from the forward time, not yet timed). Small
files are scored on every token, the larger DOAB and arXiv shards on
night.sh's 160 fixed windows.

Only this computer can open it (it listens on 127.0.0.1). `--port` changes
the port; `--no-open` does not open a browser.

## Changing a run while it trains

A run never has to start again from random weights to change course.

**Without stopping it: the control file.** Next to the checkpoint,
`runs/$RUN.ckpt.control` (another name with `--control FILE`). The trainer
looks at it before every step and acts on a change once the file has
stayed the same for half a second (a file read while it is being written
could be half empty), so on the night model at the next step or the one
after. From a second Git Bash window:
```sh
echo "lr-scale 0.5" > runs/rp1.ckpt.control
```
One command a line:
- `lr-scale X`: the learning rate is the schedule's times X from the next
  step. It holds while the line is in the file, after a restart too.
  Remove the line, or write `lr-scale 1`, to go back.
- `save`: a checkpoint now.
- `eval`: the validation loss now.
- `reason`: the reasoning exam now (when the run has `--reason-test`).
- `stop`: save and stop, as Ctrl-C does.

`save`, `eval`, `reason` and `stop` run once each time the file changes.
A file already there when training starts only keeps its `lr-scale`; a
`stop` left over from an earlier run does not stop the new one. Write the
whole file each time (`>`, not `>>`): a line left out is gone, and an
`lr-scale` left out goes back to 1. The trainer prints a `control:` line
for each command, and the log gets a `# control step N: ...` row.

**With a short stop: change the command and resume.** Ctrl-C (or `stop`)
saves everything: the weights, the optimizer's state, the step and the
data order. The same command with `--resume` and other options continues
from there: another data mix (`--data`), a new learning rate (`--lr`,
`--steps`, `--min-lr-frac`), unlikelihood, the sampler, the exams. Only
the model's shape is fixed by the checkpoint. This is how the RunPod run
moved to its math mix at step 32,414.

**A schedule for runs without a planned end: `--schedule wsd`.** The
default schedule (cosine) brings the learning rate down slowly towards
`--steps`, so the end has to be planned from the start: resuming with
another `--steps` makes the rate jump. With `--schedule wsd`, the rate
stays flat after the warmup and comes down only over the last
`--decay-steps` (by default a fifth of `--steps`). Before that, the rate
does not depend on `--steps`, so the run can be made longer at any
resume. To get a model to score, cool down a copy of its checkpoint:
```sh
cp runs/rp1.ckpt runs/rp1-cool.ckpt
build/cxxgpt-gpu.exe train --gpu ... --resume --out runs/rp1-cool.ckpt \
    --schedule wsd --steps $((STEP + 2000)) --decay-steps 2000
```
where `STEP` is the step `rp1.ckpt` holds and `...` stands for the
run's own options, with logs of the copy's own (`--log`, `--load-log`,
`--reason-report`). The copy starts its cooldown from the run's exact
state; `rp1.ckpt` is not touched, so the run itself resumes afterwards
at the full rate. The laptop's GPU trains one model at a time: run the
cooldown while the main run is stopped (or on another GPU). This is a
smooth continuation when the run itself trains with `--schedule wsd`.
For a run trained with cosine (such as the RunPod run), give the copy
`--lr` equal to the last rate in the run's log (`lr` column), or its
rate jumps back up to `--lr` at the first step. With 20% of
the steps as cooldown, WSD matched cosine in a published comparison
(Hagele et al., 2024). The cooldown makes most of its drop early: after a
quarter of its steps, half the drop is done.

## The plan

### First evening (about two hours, mostly downloads)

1. Setup and tests, as above.
2. **Throughput probe:** how many tokens per second each model size trains
   at, and whether it fits in the 8 GB (a few minutes):
   ```sh
   scripts/probe.sh
   ```
   It ends with the `night.sh` command for each size that fits, its
   measured speed and batch filled in. Bigger models learn more from each
   token but see fewer of them in a night: a reasonable pick is the
   largest size that still trains 0.3B tokens a night or more (the
   tokens/night column), so a week of nights gives it a couple of billion.
   The first nights train in full precision (the sizes without
   `-ternary`); the `d768-moe-ternary` row is the speed of the last
   nights, once the model is switched to ternary (below).
3. **Data and the model:** the RunPod run's data, tokenizer and
   checkpoints (README, "The training data"), from the owner's Google
   Drive (docs/PREPARED_DATA.md: about 23 GB unpacked). After unpacking,
   the clone has `data/licensed-10b/` (the corpus, `tok.bpe`, GSM8K dev
   and test), `data/targeted-math/` (word problems, GSM8K train) and
   `runs/runpod-2026-10-16/` (the checkpoints and the run's logs).
   `night.sh` names any file it cannot find. `scripts/get_data.py gsm8k`
   downloads GSM8K's JSON Lines into `data/gsm8k/` (4 MB from GitHub; the
   same dev and test questions), which `night.sh` does not need; CI checks
   on every push that GSM8K's files are still where it expects them
   (`get_data.py --check`).

### The dev set (`DEV=1`)

We choose settings on a dev set, never on the GSM8K test set (which gives
the headline numbers only). `get_data.py` holds 500 GSM8K train problems
out as `data/gsm8k/dev.jsonl` (the 500 whose question text has the smallest
sha256 digest, so every preparation picks the same ones), and drops them
from the training files.

- The RunPod data uses the same 500 (its `gsm8k-dev.txt` is the same
  file, byte for byte). Its math files leave out every training problem
  that shares a run of 13 words with a dev or test question (7 GSM8K
  problems, 36,911 word problems).
- `DEV=1 ... scripts/night.sh` then validates and takes its exams on
  `data/licensed-10b/gsm8k-dev.tok` and `gsm8k-dev.txt` instead of the
  test files. The default stays `DEV=0` (the test set).

### Night 1

```sh
TOKS=... BATCH=... ACCUM=... MODEL="..." scripts/night.sh   # the line the probe printed
```

`night.sh` sizes the run so its learning-rate schedule ends when the
night does (7.5 hours by default), trains on the data mix (README, "The
training data"), and checks itself on the GSM8K test set, which training never sees
(or on the dev set with `DEV=1`, above): validation loss and exact answers
every 500 steps. The exact-answer exams run on the GPU (16 s for
250 problems on the night-1 model, against 5 minutes on the CPU, which
left the GPU idle): `reasoning eval on the GPU, 28 answers at a time` in
the log. They score the model as training computes it (bf16 products), so
`cxxgpt eval-reasoning` (on the CPU, or with `--gpu` and its exact
products, below) can differ by a problem or two. If the
GPU has no room for the answers' caches, the log says so and the exam
runs on the CPU as before. Its settings
(model size, optimizer, batch, data mix) are at the top of the script.
Run the same command again to resume after an interruption. The log is
`runs/night1.log`; in the morning, `scripts/report.sh night1` sums it up on
one screen (validation loss and exact answers over the night, speed, how
it ended, and one answer the model got wrong). Then `scripts/ledger.sh
night1 "note"` adds the night's row to [runs.md](runs.md), the ledger of
every run (tokens, hours, energy, scores), and you commit it: `runs/` stays
on the laptop, the ledger goes to GitHub. Run it again after scoring the
full test set (`eval-reasoning --labels runs/night1.labels.jsonl`).

For MoE models, `night.sh` also writes `runs/night1.load.csv` (or
`runs/$RUN.load.csv`): `step,layer,expert_0,...,max_load_ratio`. You can
request this on any CPU or GPU run with `train --load-log FILE`.
Layer and expert numbers start at zero. Each row covers the successful
training steps since the previous row, every `--log-every` steps, plus a
final partial interval when the run stops. Validation and skipped updates
do not count. Shares divide each expert's assignments by all assignments
in that layer, so they sum to one even with top-2 routing. The last column
is the largest share times the number of experts: 1 means equal use.
The existing console `load` still describes the last step's worst layer.
With `--load-log` (only then), the console also gets one warning line per
logging interval that lists every expert below 1% of its layer, in layer
order, e.g. `under 1%: layer 2 expert 5 (0.0%), layer 7 expert 3 (0.2%)`; after 12
experts it ends with "and N more".
`report.sh` shows each layer's final shares and the first logged step
with a share below 1%. Resuming appends rows to the same file, after a
`# resumed at step N` line; use a new file for a different run. As with
`--log`, training restarts from the last checkpoint, so the steps after it
are run again and their rows appear twice: the rows after the `# resumed`
line are the ones the model kept. Nothing is written for dense models: they have
no experts, so `--load-log` creates no file.
If an expert stays under 1% for a night, `cxxgpt clone-experts --ckpt
runs/night2.ckpt --out runs/night2-repaired.ckpt --auto --load-log
runs/night2.load.csv` copies its layer's busiest expert into it (and
prints every layer's shares; nothing changes where no expert is under
1%). Start the next night from the repaired file with `INIT=`
(docs/research-plan.md, "Reviving dead experts").

To ask the model a question
yourself (`--temp 0`: its most likely answer; `--answer`: stop after the
`#### answer` line):
```sh
./cxxgpt sample --ckpt runs/rp1.ckpt --bpe data/licensed-10b/tok.bpe --temp 0 --answer \
  --prompt $'Q: A pen costs $3. How much do 7 pens cost?\nA: '
```

To score a checkpoint on the whole GSM8K test (1319 problems) in a
minute or two, on the GPU (`--gpu`, with `cxxgpt-gpu`):
```sh
T=data/licensed-10b
build/cxxgpt-gpu.exe eval-reasoning --gpu --ckpt runs/rp1.ckpt --test $T/gsm8k-test.txt --bpe $T/tok.bpe
build/cxxgpt-gpu.exe eval-reasoning --gpu --ckpt runs/rp1.ckpt --test $T/gsm8k-test.txt --bpe $T/tok.bpe \
  --tries 8 --temp 0.7
```
The products are exact fp32 by default, so the scores are the CPU's
(night 1: 20 / 250 with `--limit 250` on both, 14 s on the GPU against
about 5 minutes on the CPU). On night 1: greedy 90 / 1319 (6.8%) in 71 s,
64 answers at a time; 8 tries and a vote (maj@8) 103 / 1319 (7.8%) in
13.5 minutes (one sampled try alone: 4.8%; any of the 8 right: 20.7%).
`--batch` caps the answers decoded together (64 by default; each takes
75 MB of GPU memory on the night-1 model, and the GPU fits about 75).
`--matmul tf32` or `bf16` scores the model as training computes it, which
can differ by a problem or two. Another program on the GPU (a training
run, a second eval) slows it down several times: a greedy run alongside a
training run took 447 s.

### Continuing the RunPod model

The laptop continues the RunPod model (README, "The training data") with
its optimizer state, on its last data mix (`night.sh`'s default). Its
checkpoint holds step 46,946, and its learning rate had come down to
1.2e-4 at the end of its cosine schedule. Copy the checkpoint under the
new run's name (the original stays as it is), then:
```sh
cp runs/runpod-2026-10-16/fresh-eu.ckpt runs/rp1.ckpt
RUN=rp1 STEP0=46946 LR=1.2e-4 DEV=1 HOURS=30 TOKS=28000 scripts/night.sh --sampler qa --time-limit 450
```
- `STEP0` says where the checkpoint is: the run's steps come after it, at
  a flat `LR`, with a cooldown over the last fifth (`--schedule wsd`).
  Without `LR`, the rate would jump to the default 1.5e-3, twelve times
  where the run left it.
- `HOURS=30` plans four nights; `--time-limit 450` stops each one after
  7.5 hours with a checkpoint, and the same command the next evening
  resumes it, optimizer and data order included.
- `TOKS` is the speed `scripts/probe.sh` measures for this model (28,000
  tokens/s for `d768-moe` at the last measurement); `BATCH=8 ACCUM=16`
  gives the same 65,536 tokens a step as the RunPod run's 16 x 8.
- `--sampler qa` is the RunPod run's: a window that starts inside a
  problem starts at its question.

Not yet run on the laptop: the first night's log must start with
`resumed runs/rp1.ckpt at step 46946`, and its first logged rate must be
1.2e-4.

### Following nights

What each night trains on depends on what the last one shows; we decide
it together each morning. A run that keeps going reruns its own command
(with a larger `HOURS` to make it longer; with `--schedule wsd`, the rate
does not depend on it before the cooldown). Starting a new run from an
earlier one's weights looks like this (the model comes from the
checkpoint; the optimizer and the learning-rate schedule start over):
```sh
RUN=rp2 INIT=runs/rp1.ckpt MIX="..." TOKS=... BATCH=... ACCUM=... scripts/night.sh
```
The model trains in full precision first, and the last night or two
switch it to ternary, which is what our kernels run: `TERNARY=1` with
`INIT`, and the ternary row's `TOKS`, `BATCH` and `ACCUM` from the probe.
`TERNARY=1` keeps attention in full precision (`--fp-attn`, the best
curriculum run so far, though not yet a settled result: README "Ternary
FFN, full-precision attention"); `TERNARY=all` makes it ternary too.
`TERNARY=all` passes `--fp-attn 0`, which clears a saved `--fp-attn`, so it
also works when `INIT` is a checkpoint from a `TERNARY=1` night.
On the curriculum, switching for the last third of training cost half as
much as training ternary throughout (README, "Full precision first,
ternary at the end").
```sh
RUN=night5 INIT=runs/night4.ckpt TERNARY=1 TOKS=... BATCH=... ACCUM=... scripts/night.sh
```
CPU-scale experiments and engine work continue on the cloud VM during the
day.

### GPU options

Two GPU speed options; `test_gpu`, which `build_windows.cmd` runs, checks
both against the model in double precision on the GPU (`src/gpu_ref.cu`):
- fused attention, on by default since 2026-10-14 (`--no-fused-attn`
  turns it off; `--fused-attn` insists on it). Its kernels never store the
  attention weights, which saves about 1.2 GB of GPU memory at the night-1
  size and leaves room for a larger batch or model. With bf16 (the
  default) or TF32 matmuls they multiply on the tensor cores
  (`src/attention_tc.cuh`): on the night-1 model, attention takes 380 ms
  of a step instead of 930 ms, and training runs at 16,900 tokens/s at
  4 x 32 (cuBLAS: 15,000) and 17,800 at 8 x 16, which only fits with them
  (2026-10-14; since 2026-10-16, with everything else since then, the
  night model's attention takes 240 ms of a 2.2 s step).
  With `--matmul fp32` they run the exact fp32 kernels
  (`src/attention.cuh`). Heads larger than 128 use the cuBLAS attention;
- `--matmul bf16`, the default since 2026-10-14 (`--matmul tf32` goes
  back): the model's matrix products take
  inputs stored in bf16 (8 significant bits against TF32's 11), which the
  tensor cores multiply at twice the TF32 rate. Weights, gradients and the
  optimizer stay fp32 (the products use a bf16 copy of the weights), and
  so do the router and attention; Muon's products take bf16 inputs too.
  On the night-1 model at 8 x 16: 25,600 tokens/s against 17,600, with
  7.0 GB of GPU memory against 6.7.
  Ternary models too, since 2026-10-16: their products multiply the
  codes themselves (-1/0/+1 weights, int8 activations, exact in bf16)
  and apply the scales after the sum. On the ternary night model: 2486
  ms a step against TF32's 3600 (26,400 tokens/s against 18,200), 7.4 GB
  against 7.7, the same losses over 100-200 steps (`GPU="--gpu --matmul
  tf32"` goes back).

Until 2026-10-16, `test_gpu` compared the GPU with the CPU engine in float,
and two checks failed from the first run on the laptop's RTX 4070
(2026-10-14), under WSL and natively alike. Against the fp64 reference both
pass, at 1.6e-6 and 7.4e-7 against a bound of 1e-5 (#27): the errors below
were the CPU float engine's, not the fused kernels'. The old record:
- `heads of 64` (70 tokens, fused fp32): 4.1e-4 against 1e-4. The cuBLAS
  attention misses that bound too there (1.8e-4); at 24, 128, 256 and 512
  tokens both are within it and equal (6.0e-5 fused, 6.9e-5 cuBLAS at
  512, the night-1 head size);
- `head of 128` (fused fp32): 1.3e-4, and 1.9e-4 at 128 tokens, about
  three times the cuBLAS attention's error (5e-5 to 6e-5) at every length
  tried. No night size uses head 128 (all have head 64).
Both missed on a norm weight's gradient, the tensor most sensitive to
rounding. On 2026-10-15 (main at 4e9499b) they measured 1.5e-4 (`l0.ln1`)
and 1.1e-4, the latter now worst on an attention output matrix
(`l1.wproj`): the sizes move a little from build to build, the worst tensor
with them. Neither is what the nights run (TF32 with the tensor-core
kernels). Those kernels pass their own checks: alone, within 2e-4 to 1e-3
of an exact reference (TF32 keeps 11 significant bits), and in a TF32
model as close to the CPU as the cuBLAS attention is (gradients 7.0e-2
against cuBLAS's 6.1e-2, both worst on a norm weight). A third, `dense
bf16`, failed until its bounds were set from what bf16 inputs measure
(gradients 1.3e-1, a norm weight's; the MoE runs' worst is the
router's, 4.4e-1 against TF32's 1.0e-1 on the same model).

Where a training step spends its GPU time (if it runs out of memory, use
`--batch 4 --accum 32`):
```sh
./cxxgpt-gpu train --gpu --profile --data data/reason.train.tok --vocab 16384 --seq 512 \
  --batch 8 --accum 16 --steps 10 --warmup 5 --eval-every 0 --save-every 0 --out build/profile.ckpt \
  --muon --dim 768 --layers 12 --heads 12 --experts 8 --topk 2 2>&1 | tail -25
```
(`data/reason.train.tok` is the file the probe made.)
