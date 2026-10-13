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
6. Get the code, in the Linux home folder (much faster than `/mnt/c`):
   ```sh
   git clone https://github.com/AndreiAgapie97/CxxGPT.git ~/CxxGPT
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
- `nvidia-smi` in another terminal shows GPU load, memory and temperature.

## The plan

1. **First evening:** setup and tests, then a short throughput probe (how
   many tokens per second each model size trains at), then the data
   downloads and our tokenizer:
   ```sh
   .venv/bin/python scripts/get_data.py gsm8k tinystories fineweb-edu openmath2 metamath
   ```
   Each source is a few GB at most; everything fits in well under 100 GB.
2. **Night 1:** general text plus math, so the model learns language.
3. **Following nights:** worked math solutions (OpenMathInstruct-2,
   MetaMathQA, GSM8K train), checked every few hundred steps on the
   GSM8K test set, which training never sees.
4. **In between:** CPU-scale experiments and engine work continue on the
   cloud VM during the day; the laptop trains at night.
