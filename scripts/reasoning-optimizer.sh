#!/usr/bin/env bash
# Native 2x2 optimizer/LR comparison. Select on development only; freeze
# both selections before inspecting a new final probe. No training changes.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?provide a new directory}
SOURCE_DIR=${SOURCE_DIR:-runs/reasoning-2000-20261013}
REFERENCE_DIR=${REFERENCE_DIR:-runs/reasoning-capacity-20261014}
THREADS=${THREADS:-2}
if [[ "${RESUME_RUN:-0}" == 1 ]]; then
    [[ -d "$RUN_DIR" && -f "$RUN_DIR/protocol.txt" && -f "$RUN_DIR/targets/probe.txt" ]]
    grep -q "threads=$THREADS;" "$RUN_DIR/protocol.txt"
    sha256sum -c "$RUN_DIR/binary-sha256.txt" > "$RUN_DIR/recovery-binaries-check.txt"
    sha256sum -c "$RUN_DIR/data-sha256.txt" > "$RUN_DIR/recovery-data-check.txt"
    awk '$2 ~ /^(src|experiments)\// {print}' "$RUN_DIR/source-sha256.txt" | sha256sum -c > "$RUN_DIR/recovery-source-check.txt"
    sha256sum scripts/reasoning-optimizer.sh >> "$RUN_DIR/recovery-driver-sha256.txt"
    if [[ -f "$RUN_DIR/selection.json" ]]; then
        sha256sum -c "$RUN_DIR/selection-sha256.txt"
        sha256sum -c "$RUN_DIR/checkpoints-sha256.txt"
    fi
else
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
make -B -j"$THREADS" "CXX=${CXX:-g++}" EXTRA=-Werror cxxgpt reasoning_audit reasoning_targets test_training_cli > "$RUN_DIR/build.log" 2>&1
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
git status --short > "$RUN_DIR/source-status.txt"
git diff HEAD --binary > "$RUN_DIR/source.patch"
sha256sum src/*.h src/cxxgpt.cpp experiments/reasoning_*.cpp scripts/reasoning-optimizer.sh scripts/select-reasoning-optimizer.py > "$RUN_DIR/source-sha256.txt"
sha256sum cxxgpt reasoning_audit reasoning_targets > "$RUN_DIR/binary-sha256.txt"
"${CXX:-g++}" --version > "$RUN_DIR/compiler.txt"
for f in cpu.max cpu.stat; do
    [[ ! -r "/sys/fs/cgroup/$f" ]] || cat "/sys/fs/cgroup/$f" > "$RUN_DIR/$f.before.txt"
done
cp docs/results/2026-10-14/capacity/probe.txt "$RUN_DIR/dev-probe.txt"
cat "$SOURCE_DIR/reason.test.txt" docs/results/2026-10-14/focus/probe.txt \
    "$RUN_DIR/dev-probe.txt" > "$RUN_DIR/excluded.txt"
./reasoning_targets --train "$SOURCE_DIR/reason.train.tok" --val "$SOURCE_DIR/reason.val.tok" \
    --exclude-test "$RUN_DIR/excluded.txt" --seed 20261016 --out "$RUN_DIR/targets" > "$RUN_DIR/targets.log"
sha256sum "$SOURCE_DIR"/reason.* "$RUN_DIR/dev-probe.txt" "$RUN_DIR/targets/probe.txt" > "$RUN_DIR/data-sha256.txt"
printf '2 optimizers x lr0.0015/0.003 x dense/moe-ternary x seeds1/2/3; 2000 updates, B2/T128, dim128/layers2/heads4/FFN256/expert128/topk2/E4/shared0, full labels; warmup20 clip1 wd0.1 cosine minimum0.1; threads=%s; reuse six AdamW0.003 reference checkpoints; fresh final probe seed20261016, 600 questions\nSelection: maximum median dev exact, then sum dev exact, then minimum median sampled validation loss, then arm ID. Freeze both selections before any final scoring; final compares selected recipe and original AdamW0.003.\n' "$THREADS" > "$RUN_DIR/protocol.txt"
fi
ARMS=(adamw-lr0015 adamw-lr003 muon-lr0015 muon-lr003)
[[ -f "$RUN_DIR/training-time.csv" ]] || printf 'variant,seed,arm,wall_nanoseconds,status\n' > "$RUN_DIR/training-time.csv"
for seed in 1 2 3; do
    VARIANTS=(dense moe-ternary)
    ((seed % 2 == 1)) || VARIANTS=(moe-ternary dense)
    for variant in "${VARIANTS[@]}"; do
        for ((i=0; i<4; ++i)); do
            arm=${ARMS[$(((i+seed-1)%4))]}
            prefix="$RUN_DIR/$variant-seed$seed-$arm"
            if [[ -f "$prefix.ckpt" && -f "$prefix.dev.audit.csv" ]]; then
                ./test_training_cli same-sampler "$REFERENCE_DIR/$variant-seed$seed-large.ckpt" "$prefix.ckpt"
                printf '%s preserved completed arm\n' "$prefix"
                continue
            fi
            if [[ "$arm" == adamw-lr003 ]]; then
                cp "$REFERENCE_DIR/$variant-seed$seed-large.ckpt" "$prefix.ckpt"
                printf 'reused %s\n' "$REFERENCE_DIR/$variant-seed$seed-large.ckpt" > "$prefix.log"
                printf '%s,%s,%s,,reused\n' "$variant" "$seed" "$arm" >> "$RUN_DIR/training-time.csv"
            else
                ARCH=(); OPT=(); lr=0.0015
                [[ "$variant" != moe-ternary ]] || ARCH+=(--ternary --experts 4 --topk 2 --ehidden 128)
                [[ "$arm" != muon* ]] || OPT+=(--muon)
                [[ "$arm" != *lr003 ]] || lr=0.003
                CONTINUE=()
                [[ ! -f "$prefix.ckpt" ]] || CONTINUE+=(--resume)
                if [[ -f "$prefix.log" ]]; then
                    mv "$prefix.log" "$prefix.interrupted.$(date +%s%N).log"
                fi
                train_started_ns=$(date +%s%N)
                ./cxxgpt train --data "$SOURCE_DIR/reason.train.tok" --val "$SOURCE_DIR/reason.val.tok" \
                    --seq 128 --dim 128 --layers 2 --heads 4 --ffn 256 --batch 2 --steps 2000 \
                    --lr "$lr" --warmup 20 --clip 1 --seed "$seed" --threads "$THREADS" --eval-every 2000 \
                    --eval-batches 5 --log-every 100 --save-every 0 "${ARCH[@]}" "${OPT[@]}" \
                    --out "$prefix.ckpt" --log "$prefix.csv" "${CONTINUE[@]}" > "$prefix.log"
                train_finished_ns=$(date +%s%N)
                printf '%s,%s,%s,%s,trained\n' "$variant" "$seed" "$arm" "$((train_finished_ns-train_started_ns))" >> "$RUN_DIR/training-time.csv"
            fi
            ./test_training_cli same-sampler "$REFERENCE_DIR/$variant-seed$seed-large.ckpt" "$prefix.ckpt"
            printf '%s seed%s %s matched reference data RNG\n' "$variant" "$seed" "$arm" >> "$RUN_DIR/sampler-checks.txt"
            ./reasoning_audit --ckpt "$prefix.ckpt" --problems "$RUN_DIR/dev-probe.txt" \
                --val-data "$SOURCE_DIR/reason.val.tok" --threads "$THREADS" > "$prefix.dev.audit.csv.tmp"
            mv "$prefix.dev.audit.csv.tmp" "$prefix.dev.audit.csv"
            printf '%s seed%s %s complete\n' "$variant" "$seed" "$arm"
        done
    done
done
if [[ ! -f "$RUN_DIR/selection.json" ]]; then
    python3 scripts/select-reasoning-optimizer.py "$RUN_DIR" > "$RUN_DIR/selected.txt"
    sha256sum "$RUN_DIR/selection.json" > "$RUN_DIR/selection-sha256.txt"
    sha256sum "$RUN_DIR"/*.ckpt > "$RUN_DIR/checkpoints-sha256.txt"
fi
while read -r variant selected; do
    TEST_ARMS=(adamw-lr003)
    [[ "$selected" == adamw-lr003 ]] || TEST_ARMS+=("$selected")
    for arm in "${TEST_ARMS[@]}"; do
        for seed in 1 2 3; do
            prefix="$RUN_DIR/$variant-seed$seed-$arm"
            if [[ ! -f "$prefix.final.audit.csv" ]]; then
                ./reasoning_audit --ckpt "$prefix.ckpt" --problems "$RUN_DIR/targets/probe.txt" \
                --val-data "$SOURCE_DIR/reason.val.tok" --threads "$THREADS" > "$prefix.final.audit.csv.tmp"
                mv "$prefix.final.audit.csv.tmp" "$prefix.final.audit.csv"
            fi
            if [[ ! -f "$prefix.final.eval" ]]; then
                ./cxxgpt eval-reasoning --ckpt "$prefix.ckpt" --test "$RUN_DIR/targets/probe.txt" \
                --threads "$THREADS" --show-failures > "$prefix.final.eval.tmp"
                mv "$prefix.final.eval.tmp" "$prefix.final.eval"
            fi
        done
    done
done < "$RUN_DIR/selected.txt"
sha256sum -c "$RUN_DIR/selection-sha256.txt" > "$RUN_DIR/selection-check.txt"
sha256sum -c "$RUN_DIR/checkpoints-sha256.txt" > "$RUN_DIR/checkpoints-check.txt"
[[ ! -r /sys/fs/cgroup/cpu.stat ]] || cat /sys/fs/cgroup/cpu.stat > "$RUN_DIR/cpu.stat.after.txt"
