#!/usr/bin/env bash
# Fixed continued-training comparison: old digit lengths vs expanded support.
# All recipes are declared before scoring; no final probe selects a model.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${1:?provide a new run directory}
SOURCE_DIR=${SOURCE_DIR:-runs/reasoning-optimizer-20261014}
THREADS=${THREADS:-2}
STEPS=${STEPS:-1000}
if [[ ${RESUME_RUN:-0} == 1 ]]; then
    sha256sum -c "$RUN_DIR/binary-sha256.txt"
    sha256sum -c "$RUN_DIR/source-sha256.txt"
    sha256sum -c "$RUN_DIR/data-sha256.txt"
    sha256sum -c "$RUN_DIR/initial-sha256.txt"
    grep -qx "steps=$STEPS threads=$THREADS" "$RUN_DIR/settings.txt"
else
    mkdir -p "$(dirname "$RUN_DIR")"
    mkdir "$RUN_DIR"
    make -B -j"$THREADS" EXTRA=-Werror cxxgpt reasoning_audit reasoning_traces reasoning_curriculum > "$RUN_DIR/build.log" 2>&1
    git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
    git status --short > "$RUN_DIR/source-status.txt"
    git diff HEAD --binary > "$RUN_DIR/source.patch"
    sha256sum src/*.h src/cxxgpt.cpp experiments/reasoning_*.cpp scripts/reasoning-curriculum.sh scripts/check-reasoning-curriculum.py > "$RUN_DIR/source-sha256.txt"
    sha256sum cxxgpt reasoning_audit reasoning_traces reasoning_curriculum > "$RUN_DIR/binary-sha256.txt"
    g++ --version > "$RUN_DIR/compiler.txt"
    cat /sys/fs/cgroup/cpu.max > "$RUN_DIR/cpu.max.txt"
    cat /sys/fs/cgroup/cpu.stat > "$RUN_DIR/cpu.stat.before.txt"
    mapfile -t CORPORA < <(rg --files --no-ignore runs -g '*.tok' | sort)
    mapfile -t PROBLEMS < <(rg --files --no-ignore runs -g '*test*.txt' -g '*probe*.txt' | sort)
    PROBLEMS+=(docs/results/2026-10-14/focus/probe.txt docs/results/2026-10-14/capacity/probe.txt docs/results/2026-10-14/optimizer/probe.txt)
    printf '%s\n' "${CORPORA[@]}" "${PROBLEMS[@]}" > "$RUN_DIR/exclusion-inputs.txt"
    corpora=$(IFS=,; echo "${CORPORA[*]}")
    problems=$(IFS=,; echo "${PROBLEMS[*]}")
    ./reasoning_curriculum --exclude-corpora "$corpora" --exclude-problems "$problems" \
        --out "$RUN_DIR/data" --seed 20261017 --train 2400 --per-cell 100 --val-per-cell 60 > "$RUN_DIR/data.log"
    python3 scripts/check-reasoning-curriculum.py "$RUN_DIR" --data-only > "$RUN_DIR/independent-data-check.txt"
    sha256sum "${CORPORA[@]}" "${PROBLEMS[@]}" "$RUN_DIR"/data/* > "$RUN_DIR/data-sha256.txt"
    for variant in dense moe-ternary; do
        arm=muon-lr0015
        [[ $variant != moe-ternary ]] || arm=muon-lr003
        for seed in 1 2 3; do
            cp "$SOURCE_DIR/$variant-seed$seed-$arm.ckpt" "$RUN_DIR/$variant-seed$seed-initial.ckpt"
        done
    done
    sha256sum "$RUN_DIR"/*-initial.ckpt > "$RUN_DIR/initial-sha256.txt"
    printf 'steps=%s threads=%s\n' "$STEPS" "$THREADS" > "$RUN_DIR/settings.txt"
    cat > "$RUN_DIR/protocol.txt" <<EOF
Paired continued training, two curricula: control difficulty1/2 balanced; expanded difficulty1/2/3 balanced.
2400 unique problems per arm, 1600 shared at the same problem positions. Replacement of one third changes support and the relative frequency of old lengths.
Full completion labels; fresh Muon state; lr0.0005, warmup20, cosine minimum0.1, wd0.1, clip1, B2/T128, $STEPS successful updates and $((STEPS*256)) sampled tokens per tail; $THREADS threads.
Start from frozen dense Muon0.0015 / ternary MoE Muon0.003 2000-update checkpoints, three seeds. Tail sampler seeds101/102/103; both arms use the same sampler draws, but streams/lengths differ. Valid supervised token budgets are replayed and reported separately.
Fresh final800 questions: add/cmp difficulties1/2/3/4, 100 per cell. Exclude every archived prior probe/test/corpus and all new training/validation questions. Question-string exclusion does not isolate semantic templates.
New validation360 is development only and is not used for selection or early stopping. Train all 12 tails before scoring any final models; freeze all six initial and twelve final checkpoint hashes first. Evaluate all planned models, not a winning seed or recipe.
Compare exact answers, strict generated column/comparison grounding, and gold byte-role loss. Expanded difficulty3 is trained support; only difficulty4 remains a new digit-length test. This is not an architecture ranking or matched compute between architectures.
EOF
    sha256sum "$RUN_DIR/protocol.txt" "$RUN_DIR/settings.txt" > "$RUN_DIR/protocol-sha256.txt"
fi
sha256sum -c "$RUN_DIR/protocol-sha256.txt"
if [[ ${PREPARE_ONLY:-0} == 1 ]]; then exit 0; fi
[[ -f "$RUN_DIR/training-time.csv" ]] || printf 'variant,seed,curriculum,wall_nanoseconds\n' > "$RUN_DIR/training-time.csv"
for seed in 1 2 3; do
    VARIANTS=(dense moe-ternary)
    ((seed % 2 == 1)) || VARIANTS=(moe-ternary dense)
    for variant in "${VARIANTS[@]}"; do
        ARMS=(control expanded)
        ((seed % 2 == 1)) || ARMS=(expanded control)
        for arm in "${ARMS[@]}"; do
            prefix="$RUN_DIR/$variant-seed$seed-$arm"
            budget=(--mode budget --data "$RUN_DIR/data/$arm.tok" --seq 128 --batch 2 --steps "$STEPS" --seed "$((100+seed))")
            if [[ -f "$prefix.ckpt" ]] && ./reasoning_curriculum "${budget[@]}" --ckpt "$prefix.ckpt" --log "$prefix.log" > "$prefix.budget.csv.tmp" 2> "$prefix.budget-check.log"; then
                mv "$prefix.budget.csv.tmp" "$prefix.budget.csv"
                continue
            fi
            CONTINUE=()
            [[ ! -f "$prefix.ckpt" ]] || CONTINUE+=(--resume)
            if [[ -f "$prefix.log" ]]; then mv "$prefix.log" "$prefix.interrupted.$(date +%s%N).log"; fi
            started=$(date +%s%N)
            ./cxxgpt train --init "$RUN_DIR/$variant-seed$seed-initial.ckpt" --out "$prefix.ckpt" \
                --data "$RUN_DIR/data/$arm.tok" --val "$RUN_DIR/data/validation.tok" --batch 2 --seq 128 \
                --steps "$STEPS" --muon --lr 0.0005 --warmup 20 --min-lr-frac 0.1 --wd 0.1 --clip 1 \
                --seed "$((100+seed))" --threads "$THREADS" --eval-every "$STEPS" --eval-batches 5 \
                --save-every 200 --log-every 100 --log "$prefix.csv" "${CONTINUE[@]}" > "$prefix.log"
            finished=$(date +%s%N)
            printf '%s,%s,%s,%s\n' "$variant" "$seed" "$arm" "$((finished-started))" >> "$RUN_DIR/training-time.csv"
            ./reasoning_curriculum "${budget[@]}" --ckpt "$prefix.ckpt" --log "$prefix.log" > "$prefix.budget.csv"
        done
    done
done
if [[ -f "$RUN_DIR/final-checkpoints-sha256.txt" ]]; then
    sha256sum -c "$RUN_DIR/final-checkpoints-sha256.txt"
else
    # Never change a frozen model set after any final evidence exists.
    if compgen -G "$RUN_DIR/*.trace.csv" > /dev/null || compgen -G "$RUN_DIR/*.audit.csv" > /dev/null; then
        echo 'final scores exist without frozen models' >&2; exit 1
    fi
    sha256sum "$RUN_DIR"/*.ckpt > "$RUN_DIR/final-checkpoints-sha256.txt.tmp"
    mv "$RUN_DIR/final-checkpoints-sha256.txt.tmp" "$RUN_DIR/final-checkpoints-sha256.txt"
fi
for variant in dense moe-ternary; do
    for seed in 1 2 3; do
        for arm in initial control expanded; do
            prefix="$RUN_DIR/$variant-seed$seed-$arm"
            if [[ ! -f "$prefix.trace.csv" ]]; then
                ./reasoning_traces --ckpt "$prefix.ckpt" --problems "$RUN_DIR/data/probe.txt" --threads "$THREADS" \
                    --transcripts "$prefix.transcripts.tsv.tmp" > "$prefix.trace.csv.tmp"
                mv "$prefix.transcripts.tsv.tmp" "$prefix.transcripts.tsv"
                mv "$prefix.trace.csv.tmp" "$prefix.trace.csv"
            fi
            if [[ ! -f "$prefix.audit.csv" ]]; then
                ./reasoning_audit --ckpt "$prefix.ckpt" --problems "$RUN_DIR/data/probe.txt" --threads "$THREADS" \
                    > "$prefix.audit.csv.tmp"
                mv "$prefix.audit.csv.tmp" "$prefix.audit.csv"
            fi
        done
    done
done
sha256sum -c "$RUN_DIR/final-checkpoints-sha256.txt" > "$RUN_DIR/checkpoints-check.txt"
sha256sum -c "$RUN_DIR/protocol-sha256.txt" > "$RUN_DIR/protocol-check.txt"
cat /sys/fs/cgroup/cpu.stat > "$RUN_DIR/cpu.stat.after.txt"
python3 scripts/check-reasoning-curriculum.py "$RUN_DIR" > "$RUN_DIR/independent-check.txt.tmp"
mv "$RUN_DIR/independent-check.txt.tmp" "$RUN_DIR/independent-check.txt"
