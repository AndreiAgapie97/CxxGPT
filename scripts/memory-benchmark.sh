#!/usr/bin/env bash
# Fresh native process per configuration; no concurrent research workloads.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?provide a new directory}
ROUNDS=${ROUNDS:-3}
THREADS=${THREADS:-2}
[[ "$ROUNDS" =~ ^[1-9][0-9]*$ && "$THREADS" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
make -B "CXX=${CXX:-g++}" EXTRA=-Werror memory_probe > "$RUN_DIR/build.log" 2>&1
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
git status --short > "$RUN_DIR/source-status.txt"
git diff HEAD --binary > "$RUN_DIR/source.patch"
sha256sum src/*.h experiments/memory_probe.cpp scripts/memory-benchmark.sh > "$RUN_DIR/source-sha256.txt"
sha256sum memory_probe > "$RUN_DIR/binary-sha256.txt"
"${CXX:-g++}" --version > "$RUN_DIR/compiler.txt"
uname -a > "$RUN_DIR/platform.txt"
awk '/Cpus_allowed_list/ {print}' /proc/self/status > "$RUN_DIR/affinity.txt"
for f in cpu.max cpu.stat; do
    [[ ! -r "/sys/fs/cgroup/$f" ]] || cat "/sys/fs/cgroup/$f" > "$RUN_DIR/$f.before.txt"
done
printf 'rounds=%s threads=%s dim=64 layers=2 heads=4 dense_ffn=128 expert_ffn=64 experts=4 topk=2 shared=0 seed=1 train_steps=3 train_batch=2 decode_tokens=16\n' "$ROUNDS" "$THREADS" > "$RUN_DIR/protocol.txt"
for ((round=0; round<ROUNDS; ++round)); do
    VARIANTS=(dense dense-ternary moe moe-ternary)
    for ((i=0; i<4; ++i)); do
        variant=${VARIANTS[$(((i+round)%4))]}
        ARCH=()
        [[ "$variant" != *ternary* ]] || ARCH+=(--ternary)
        [[ "$variant" != moe* ]] || ARCH+=(--experts 4)
        for vocab in 256 8192; do
            for seq in 32 128; do
                OPTS=(adamw muon)
                ((round % 2 == 0)) || OPTS=(muon adamw)
                for opt in "${OPTS[@]}"; do
                    OPT=()
                    [[ "$opt" != muon ]] || OPT+=(--muon)
                    ./memory_probe --mode train "${ARCH[@]}" "${OPT[@]}" --vocab "$vocab" \
                        --context 128 --batch 2 --seq "$seq" --threads "$THREADS" \
                        > "$RUN_DIR/r$round-$variant-v$vocab-s$seq-$opt.csv"
                done
            done
        done
    done
    for variant in dense moe; do
        ARCH=(--ternary)
        [[ "$variant" != moe ]] || ARCH+=(--experts 4)
        for vocab in 256 8192; do
            for context in 128 512; do
                MODES=(decode fp32 packed)
                ((round % 2 == 0)) || MODES=(packed fp32 decode)
                for mode in "${MODES[@]}"; do
                    BATCHES=(1 4)
                    [[ "$mode" != packed ]] || BATCHES=(1)
                    for batch in "${BATCHES[@]}"; do
                        ./memory_probe --mode "$mode" "${ARCH[@]}" --vocab "$vocab" \
                            --context "$context" --batch "$batch" --threads "$THREADS" \
                            > "$RUN_DIR/r$round-$variant-v$vocab-c$context-$mode-b$batch.csv"
                    done
                done
            done
        done
    done
done
[[ ! -r /sys/fs/cgroup/cpu.stat ]] || cat /sys/fs/cgroup/cpu.stat > "$RUN_DIR/cpu.stat.after.txt"
