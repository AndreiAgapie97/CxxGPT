#!/usr/bin/env bash
# Preserve raw results and enough provenance to reproduce a toy-world run.
# scripts/research.sh RUN_DIRECTORY [steps=200] [seeds=3] [threads=3] [rl_steps=500]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?provide a new run directory, e.g. runs/research-01}
STEPS=${2:-200}
SEEDS=${3:-3}
THREADS=${4:-3}
RL_STEPS=${5:-500}
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"  # fail rather than overwrite an existing experiment
make -B "CXX=${CXX:-g++}" EXTRA=-Werror research_experiment > "$RUN_DIR/build.log" 2>&1
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
git status --short > "$RUN_DIR/source-status.txt"
git diff HEAD --binary > "$RUN_DIR/source.patch"
sha256sum experiments/toy_world.cpp src/*.h > "$RUN_DIR/source-sha256.txt"
sha256sum research_experiment > "$RUN_DIR/binary-sha256.txt"
"${CXX:-g++}" --version > "$RUN_DIR/compiler.txt"
uname -a > "$RUN_DIR/platform.txt"
lscpu > "$RUN_DIR/cpu.txt"
printf '%s\n' "steps=$STEPS seeds=$SEEDS threads=$THREADS rl_steps=$RL_STEPS" > "$RUN_DIR/budget.txt"
./research_experiment --steps "$STEPS" --seeds "$SEEDS" --threads "$THREADS" --continual \
    --rl-steps "$RL_STEPS" --out "$RUN_DIR/results.csv" | tee "$RUN_DIR/output.log"
