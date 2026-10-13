# Training on the laptop

The laptop (RTX 4070 Laptop GPU with 8 GB, 16 threads, 32 GB RAM) trains
overnight under WSL2, the Linux that runs inside Windows. Setting it up
takes about an hour, once. After step 8, Claude Code runs on the laptop
and can do everything else from there.

## One-time setup

### On Windows

1. **Update the NVIDIA driver**, with the NVIDIA app or from nvidia.com.
   WSL uses the Windows driver; nothing GPU-related gets installed inside
   Linux except the compiler.
2. **Give Linux enough memory.** By default WSL gets half the RAM. Create
   the file `C:\Users\<you>\.wslconfig` containing:
   ```ini
   [wsl2]
   memory=24GB
   processors=16
   swap=8GB
   ```
3. **Install Ubuntu.** Open PowerShell as administrator and run
   `wsl --install -d Ubuntu-24.04`. Reboot if asked, then pick a Linux user
   name and password when Ubuntu first opens.
4. **Keep the laptop awake at night.** Plugged in: sleep "Never", and
   "When I close the lid: Do nothing" (Control Panel, Power Options). Put
   it on a hard surface so the fans can breathe.

### In Ubuntu

5. Check that the GPU is visible: `nvidia-smi` should list the RTX 4070.
6. Get the code, in the Linux home folder (much faster than `/mnt/c`).
   `-b main` picks the branch the work is on; without it, git takes the
   repository's default branch, which may be an older one:
   ```sh
   git clone -b main https://github.com/AndreiAgapie97/CxxGPT.git ~/CxxGPT
   cd ~/CxxGPT
   ```
7. Run the setup script. It installs the compilers, the CUDA compiler and
   cuBLAS, and Python with pyarrow (for data files). Then it builds
   everything and runs all tests, including the GPU against the CPU:
   ```sh
   scripts/setup_wsl.sh
   ```
   It should end with `all passed` and `All set`. If anything fails, keep
   the output; Claude can work from it.
8. Install Claude Code and start it in the project:
   ```sh
   curl -fsSL https://claude.ai/install.sh | bash
   cd ~/CxxGPT && claude
   ```
   To let it push to GitHub from the laptop, also run
   `sudo apt install gh && gh auth login` once.

## Running overnight

- Start long runs inside `tmux`, so they survive closing the terminal:
  `tmux new -s train`, start the run, then press Ctrl-b and then d to
  detach. `tmux attach -t train` brings it back.
- `--time-limit MINUTES` stops training and saves a checkpoint in time for
  the morning. Ctrl-C also saves before stopping. `--resume` continues
  from the checkpoint.
- If training diverges (the loss or gradient becomes infinite or NaN),
  that step is skipped; ten in a row stop the run without saving over
  the last checkpoint. The log then says which step the checkpoint holds;
  resume it with a lower `--lr` (night.sh: `LR=...`).
- `nvidia-smi` in another terminal shows GPU load, memory and temperature.

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
3. **Data:** downloads the public datasets (GSM8K, TinyStories,
   FineWeb-Edu, OpenMathInstruct-2, MetaMathQA), trains our tokenizer on a
   sample of them, and writes the token files. Safe to rerun if
   interrupted:
   ```sh
   scripts/prepare_data.sh
   ```
   About 10 GB of downloads; everything fits well under 50 GB.

### Night 1

```sh
tmux new -s train
TOKS=... BATCH=... ACCUM=... MODEL="..." scripts/night.sh   # the line the probe printed
```

`night.sh` sizes the run so its learning-rate schedule ends when the
night does (7.5 hours by default), trains on web text, stories and worked
math, and checks itself on the GSM8K test set, which training never sees:
validation loss every 500 steps, exact answers every 2000. Its settings
(model size, optimizer, batch, data mix) are at the top of the script.
Run the same command again to resume after an interruption. The log is
`runs/night1.log`; in the morning, `scripts/report.sh night1` sums it up on
one screen (validation loss and exact answers over the night, speed, how
it ended, and one answer the model got wrong).

### Following nights

Worked math solutions take a larger share (OpenMathInstruct-2, MetaMathQA,
GSM8K train), with the checkpoint from night 1 as the starting point. What
exactly depends on what night 1 shows; we decide it together each morning.
Starting from night 1 looks like this (the model comes from the
checkpoint; the learning-rate schedule starts over):
```sh
RUN=night2 INIT=runs/night1.ckpt MIX="..." TOKS=... BATCH=... ACCUM=... scripts/night.sh
```
CPU-scale experiments and engine work continue on the cloud VM during the
day.
