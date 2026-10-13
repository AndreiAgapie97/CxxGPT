#!/usr/bin/env bash
# Exploratory fixed-budget warm starts, with a new strictly question-disjoint
# probe. SOURCE_DIR must contain the recorded native byte-token baselines.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?provide a new directory}
STEPS=${2:-200}
THREADS=${3:-2}
SOURCE_DIR=${SOURCE_DIR:-runs/reasoning-2000-20261013}
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
make -B -j"$THREADS" "CXX=${CXX:-g++}" EXTRA=-Werror cxxgpt reasoning_audit reasoning_targets test_training_cli > "$RUN_DIR/build.log" 2>&1
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
git status --short > "$RUN_DIR/source-status.txt"
git diff HEAD --binary > "$RUN_DIR/source.patch"
sha256sum src/*.h src/cxxgpt.cpp experiments/reasoning_*.cpp > "$RUN_DIR/source-sha256.txt"
sha256sum cxxgpt reasoning_audit reasoning_targets > "$RUN_DIR/binary-sha256.txt"
"${CXX:-g++}" --version > "$RUN_DIR/compiler.txt"
printf 'steps=%s threads=%s lr=0.0005 warmup=0 clip=1 wd=0.1 optimizer=AdamW probe_seed=20261014 per_cell=100\n' "$STEPS" "$THREADS" > "$RUN_DIR/protocol.txt"
./reasoning_targets --train "$SOURCE_DIR/reason.train.tok" --val "$SOURCE_DIR/reason.val.tok" \
    --exclude-test "$SOURCE_DIR/reason.test.txt" --out "$RUN_DIR/targets" > "$RUN_DIR/targets.log"
sha256sum "$SOURCE_DIR"/reason.* "$RUN_DIR"/targets/* > "$RUN_DIR/data-sha256.txt"
for seed in 1 2 3; do
  for variant in dense moe-ternary; do
    initial="$SOURCE_DIR/$variant-seed$seed.ckpt"
    sha256sum "$initial" >> "$RUN_DIR/initial-checkpoints-sha256.txt"
    ./reasoning_audit --ckpt "$initial" --problems "$RUN_DIR/targets/probe.txt" \
        --val-data "$SOURCE_DIR/reason.val.tok" --threads "$THREADS" > "$RUN_DIR/$variant-seed$seed-initial.audit.csv"
    ARMS=(all final)
    if ((seed % 2 == 0)); then ARMS=(final all); fi
    for arm in "${ARMS[@]}"; do
      data="$SOURCE_DIR/reason.train.tok"
      [[ "$arm" != final ]] || data="$RUN_DIR/targets/train-final.tok"
      prefix="$RUN_DIR/$variant-seed$seed-$arm"
      ./cxxgpt train --init "$initial" --data "$data" --val "$SOURCE_DIR/reason.val.tok" \
          --batch 2 --steps "$STEPS" --lr 5e-4 --warmup 0 --clip 1 --seed "$seed" --threads "$THREADS" \
          --eval-every "$STEPS" --eval-batches 5 --save-every 0 --log-every 50 \
          --out "$prefix.ckpt" --log "$prefix.csv" > "$prefix.log"
      ./reasoning_audit --ckpt "$prefix.ckpt" --problems "$RUN_DIR/targets/probe.txt" \
          --val-data "$SOURCE_DIR/reason.val.tok" --threads "$THREADS" > "$prefix.audit.csv"
      ./cxxgpt eval-reasoning --ckpt "$prefix.ckpt" --test "$RUN_DIR/targets/probe.txt" \
          --threads "$THREADS" --show-failures > "$prefix.eval"
    done
    ./test_training_cli same-sampler "$RUN_DIR/$variant-seed$seed-all.ckpt" "$RUN_DIR/$variant-seed$seed-final.ckpt"
  done
done
sha256sum "$RUN_DIR"/*.ckpt > "$RUN_DIR/checkpoints-sha256.txt"
