#!/usr/bin/env bash
# Fixed capacity bundle comparison, with a fresh disjoint question probe.
# Small controls reuse the verified 2000-step baselines; no tuning on probe.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?provide a new directory}
SOURCE_DIR=${SOURCE_DIR:-runs/reasoning-2000-20261013}
PREVIOUS_PROBE=${PREVIOUS_PROBE:-runs/reasoning-focus-20261013/targets/probe.txt}
THREADS=${THREADS:-2}
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
make -B -j"$THREADS" "CXX=${CXX:-g++}" EXTRA=-Werror cxxgpt reasoning_audit reasoning_targets test_training_cli > "$RUN_DIR/build.log" 2>&1
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
git status --short > "$RUN_DIR/source-status.txt"
git diff HEAD --binary > "$RUN_DIR/source.patch"
sha256sum src/*.h src/cxxgpt.cpp experiments/reasoning_*.cpp scripts/reasoning-capacity.sh > "$RUN_DIR/source-sha256.txt"
sha256sum cxxgpt reasoning_audit reasoning_targets > "$RUN_DIR/binary-sha256.txt"
"${CXX:-g++}" --version > "$RUN_DIR/compiler.txt"
for f in cpu.max cpu.stat; do
    [[ ! -r "/sys/fs/cgroup/$f" ]] || cat "/sys/fs/cgroup/$f" > "$RUN_DIR/$f.before.txt"
done
cat "$SOURCE_DIR/reason.test.txt" "$PREVIOUS_PROBE" > "$RUN_DIR/excluded.txt"
./reasoning_targets --train "$SOURCE_DIR/reason.train.tok" --val "$SOURCE_DIR/reason.val.tok" \
    --exclude-test "$RUN_DIR/excluded.txt" --seed 20261015 --out "$RUN_DIR/targets" > "$RUN_DIR/targets.log"
sha256sum "$SOURCE_DIR"/reason.* "$PREVIOUS_PROBE" "$RUN_DIR"/targets/probe.txt > "$RUN_DIR/data-sha256.txt"
printf 'steps=2000 threads=%s batch=2 sequence=128 AdamW lr=0.003 warmup=20 clip=1 wd=0.1 probe_seed=20261015 cells=add/cmp-difficulty1/2/3 per_cell=100\nsmall=dim64 layers1 dense_ffn128 expert_ffn64; large=dim128 layers2 dense_ffn256 expert_ffn128; heads4 topk2 experts4 shared0; fresh initialization/data seeds1/2/3\n' "$THREADS" > "$RUN_DIR/protocol.txt"
for seed in 1 2 3; do
    VARIANTS=(dense moe-ternary)
    ((seed % 2 == 1)) || VARIANTS=(moe-ternary dense)
    for variant in "${VARIANTS[@]}"; do
        small="$SOURCE_DIR/$variant-seed$seed.ckpt"
        sha256sum "$small" >> "$RUN_DIR/small-checkpoints-sha256.txt"
        ./reasoning_audit --ckpt "$small" --problems "$RUN_DIR/targets/probe.txt" \
            --val-data "$SOURCE_DIR/reason.val.tok" --threads "$THREADS" > "$RUN_DIR/$variant-seed$seed-small.audit.csv"
        ./cxxgpt eval-reasoning --ckpt "$small" --test "$RUN_DIR/targets/probe.txt" \
            --threads "$THREADS" --show-failures > "$RUN_DIR/$variant-seed$seed-small.eval"
        ARCH=()
        [[ "$variant" != moe-ternary ]] || ARCH+=(--ternary --experts 4 --topk 2 --ehidden 128)
        prefix="$RUN_DIR/$variant-seed$seed-large"
        ./cxxgpt train --data "$SOURCE_DIR/reason.train.tok" --val "$SOURCE_DIR/reason.val.tok" \
            --seq 128 --dim 128 --layers 2 --heads 4 --ffn 256 --batch 2 --steps 2000 \
            --lr 3e-3 --warmup 20 --clip 1 --seed "$seed" --threads "$THREADS" --eval-every 2000 \
            --eval-batches 5 --log-every 100 --save-every 0 "${ARCH[@]}" \
            --out "$prefix.ckpt" --log "$prefix.csv" > "$prefix.log"
        ./test_training_cli same-sampler "$small" "$prefix.ckpt"
        ./reasoning_audit --ckpt "$prefix.ckpt" --problems "$RUN_DIR/targets/probe.txt" \
            --val-data "$SOURCE_DIR/reason.val.tok" --threads "$THREADS" > "$prefix.audit.csv"
        ./cxxgpt eval-reasoning --ckpt "$prefix.ckpt" --test "$RUN_DIR/targets/probe.txt" \
            --threads "$THREADS" --show-failures > "$prefix.eval"
        printf '%s seed=%s\n' "$variant" "$seed"
        sed -n '/overall/p' "$prefix.eval"
    done
done
sha256sum "$RUN_DIR"/*.ckpt > "$RUN_DIR/checkpoints-sha256.txt"
[[ ! -r /sys/fs/cgroup/cpu.stat ]] || cat /sys/fs/cgroup/cpu.stat > "$RUN_DIR/cpu.stat.after.txt"
