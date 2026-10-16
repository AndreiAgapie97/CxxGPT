#!/usr/bin/env bash
# One-time setup on Ubuntu 24.04 (under WSL2 or not) with an NVIDIA GPU:
# compilers, the CUDA compiler and cuBLAS, Python for the data scripts; then
# builds everything and runs every test, including GPU against CPU.
#   scripts/setup_wsl.sh            (CUDA_VER=12-8 picks another toolkit)
# Safe to run again: finished steps are skipped. NO_GPU=1 skips everything
# that needs a GPU present (CI uses it to test this script).
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_VER=${CUDA_VER:-13-0}
CUDA_HOME=/usr/local/cuda-${CUDA_VER/-/.}

NO_GPU=${NO_GPU:-0}

echo "== GPU"
# Under WSL the driver, and nvidia-smi, come from Windows.
if [ -d /usr/lib/wsl/lib ]; then export PATH=$PATH:/usr/lib/wsl/lib; fi
if [ "$NO_GPU" = 1 ]; then
    echo "skipped (NO_GPU=1)"
elif ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
    echo "No GPU visible. On Windows, install the latest NVIDIA driver, then run 'wsl --shutdown' and retry."
    exit 1
fi

echo "== compilers and tools"
sudo apt-get update -q
sudo apt-get install -y -q build-essential clang git tmux wget python3-venv

echo "== CUDA $CUDA_VER: the compiler and cuBLAS (the driver stays the Windows one)"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    repo=ubuntu2404
    if grep -qi microsoft /proc/version; then repo=wsl-ubuntu; fi  # NVIDIA's repo without a Linux driver
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb" \
        -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -q
    sudo apt-get install -y -q --no-install-recommends \
        "cuda-nvcc-$CUDA_VER" "cuda-cudart-dev-$CUDA_VER" "libcublas-dev-$CUDA_VER"
fi
"$CUDA_HOME/bin/nvcc" --version | tail -2

echo "== build and test (CPU)"
make -j"$(nproc)" all
make test

echo "== build and test (GPU against CPU)"
if [ "$NO_GPU" = 1 ]; then
    make -j"$(nproc)" gpu CUDA_HOME="$CUDA_HOME" CUDA_ARCH=sm_89
else
    make -j"$(nproc)" gpu CUDA_HOME="$CUDA_HOME"
    ./test_gpu
fi

echo
echo "All set. Data:     scripts/prepare_clean.sh (README, \"The training data\")"
echo "         Training: ./cxxgpt-gpu train --gpu --help"
