#!/usr/bin/env bash
# New run directory, followed by bench-infer --ckpt or model options.
# BENCH_THREADS, BENCH_TOKENS and BENCH_REPS can override the small matrix.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?provide a new run directory followed by bench-infer model options}
shift
MODEL_ARGS=("$@")
BENCH_THREADS=${BENCH_THREADS:-'1 3'}
BENCH_TOKENS=${BENCH_TOKENS:-'16 64 128'}
BENCH_REPS=${BENCH_REPS:-21}
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"  # retain previous measurements
make "CXX=${CXX:-g++}" EXTRA=-Werror cxxgpt > "$RUN_DIR/build.log" 2>&1
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
git status --short > "$RUN_DIR/source-status.txt"
git diff HEAD --binary > "$RUN_DIR/source.patch"
sha256sum src/cxxgpt.cpp src/*.h > "$RUN_DIR/source-sha256.txt"
sha256sum cxxgpt > "$RUN_DIR/binary-sha256.txt"
"${CXX:-g++}" --version > "$RUN_DIR/compiler.txt"
uname -a > "$RUN_DIR/platform.txt"
lscpu > "$RUN_DIR/cpu.txt"
printf '%q ' "${MODEL_ARGS[@]}" > "$RUN_DIR/model-options.txt"
printf '\nthreads=%s tokens=%s reps=%s warmup=2\n' "$BENCH_THREADS" "$BENCH_TOKENS" "$BENCH_REPS" > "$RUN_DIR/budget.txt"
for ((i=0; i<${#MODEL_ARGS[@]}; ++i)); do
    if [[ "${MODEL_ARGS[i]}" == --ckpt ]]; then
        sha256sum -- "${MODEL_ARGS[i+1]}" > "$RUN_DIR/checkpoint-sha256.txt"
    fi
done
# Run serially with no other research builds or experiments in flight.
# Both paths use the same threads and stream. Warmups are omitted from CSV.
for THREADS in $BENCH_THREADS; do
    for TOKENS in $BENCH_TOKENS; do
        ./cxxgpt bench-infer "${MODEL_ARGS[@]}" --threads "$THREADS" --tokens "$TOKENS" \
            --reps "$BENCH_REPS" --warmup 2 --log "$RUN_DIR/t${THREADS}-n${TOKENS}.csv" \
            > "$RUN_DIR/t${THREADS}-n${TOKENS}.log"
    done
done
