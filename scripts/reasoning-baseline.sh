#!/usr/bin/env bash
# A small reasoning comparison using Claude's existing native data/train/eval
# path. Every variant has the same active FFN width, steps, and data samples.
# scripts/reasoning-baseline.sh RUN_DIRECTORY [steps=200] [seeds=3] [threads=3]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?provide a new run directory}
STEPS=${2:-200}
SEEDS=${3:-3}
THREADS=${4:-3}
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
make -B "CXX=${CXX:-g++}" EXTRA=-Werror cxxgpt > "$RUN_DIR/build.log" 2>&1
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
git status --short > "$RUN_DIR/source-status.txt"
sha256sum src/*.h src/cxxgpt.cpp > "$RUN_DIR/source-sha256.txt"
sha256sum cxxgpt > "$RUN_DIR/binary-sha256.txt"
"${CXX:-g++}" --version > "$RUN_DIR/compiler.txt"
lscpu > "$RUN_DIR/cpu.txt"
printf '%s\n' "steps=$STEPS seeds=$SEEDS threads=$THREADS" > "$RUN_DIR/budget.txt"
./cxxgpt gen-reasoning --out "$RUN_DIR/reason" --train 2000 --val 500 --tasks add,cmp \
    --min-diff 1 --max-diff 2 --test-max-diff 3 --test-per-cell 10 --seed 20261013 \
    > "$RUN_DIR/data.log"
sha256sum "$RUN_DIR"/reason.* > "$RUN_DIR/data-sha256.txt"
NAMES=(dense dense-ternary moe moe-ternary)
for ((seed=1; seed<=SEEDS; ++seed)); do
    for ((index=0; index<4; ++index)); do
        variant=$(((index + seed - 1) % 4))
        name=${NAMES[$variant]}
        flags=()
        if ((variant % 2)); then flags+=(--ternary); fi
        if ((variant >= 2)); then flags+=(--experts 4 --topk 2 --ehidden 64); fi
        prefix="$RUN_DIR/$name-seed$seed"
        ./cxxgpt train --data "$RUN_DIR/reason.train.tok" --val "$RUN_DIR/reason.val.tok" \
            --seq 128 --dim 64 --layers 1 --heads 4 --ffn 128 --batch 2 --steps "$STEPS" \
            --lr 3e-3 --warmup 20 --seed "$seed" --threads "$THREADS" --eval-every "$STEPS" \
            --eval-batches 5 --log-every 50 --save-every 0 "${flags[@]}" \
            --out "$prefix.ckpt" --log "$prefix.csv" > "$prefix.log"
        ./cxxgpt eval-reasoning --ckpt "$prefix.ckpt" --test "$RUN_DIR/reason.test.txt" \
            --threads "$THREADS" --show-failures > "$prefix.eval"
        printf '%s seed=%d\n' "$name" "$seed"
        sed -n '/overall/p' "$prefix.eval"
    done
done
