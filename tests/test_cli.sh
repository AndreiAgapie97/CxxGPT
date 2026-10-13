#!/usr/bin/env bash
# Exercise the real CLI after config parsing and backend selection.
set -euo pipefail
CXXGPT=${CXXGPT:-./cxxgpt}
CXXGPT_AUDIT=${CXXGPT_AUDIT:-./reasoning_audit}
CXXGPT_TARGETS=${CXXGPT_TARGETS:-./reasoning_targets}
cd "$(dirname "$0")/.."
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT
printf 'abcdefghi' > "$TEST_DIR/text.txt"
"$CXXGPT" text2tok --in "$TEST_DIR/text.txt" --out "$TEST_DIR/data.tok" > "$TEST_DIR/data.log"
COMMON=(--data "$TEST_DIR/data.tok" --val "$TEST_DIR/data.tok" --seq 8 --dim 8 --heads 2
        --ffn 12 --layers 1 --batch 1 --steps 1 --warmup 0 --eval-every 1 --eval-batches 1
        --save-every 0 --threads 2 --out "$TEST_DIR/model.ckpt")
"$CXXGPT" train "${COMMON[@]}" > "$TEST_DIR/train.log"
grep -q '9 train tokens, 9 val tokens' "$TEST_DIR/train.log"
grep -q '| val ' "$TEST_DIR/train.log"
for option in dim heads experts shared ffn ehidden topk batch accum log-every eval-batches; do
    value=-1
    if [[ "$option" == dim || "$option" == heads || "$option" == topk || "$option" == batch || "$option" == accum || "$option" == log-every || "$option" == eval-batches ]]; then value=0; fi
    status=0
    "$CXXGPT" train "${COMMON[@]}" "--$option" "$value" > "$TEST_DIR/invalid.log" 2>&1 || status=$?
    if [[ "$status" != 1 ]] || ! grep -q 'check failed' "$TEST_DIR/invalid.log"; then
        cat "$TEST_DIR/invalid.log"
        printf 'invalid --%s %s returned %s instead of a checked error\n' "$option" "$value" "$status"
        exit 1
    fi
done
printf 'CLI: single-window train/validation and eleven invalid configurations passed\n'

./test_training_cli generate "$TEST_DIR"
for variant in dense ternary moe ternary-moe; do
    ARCH=()
    if [[ "$variant" == *ternary* ]]; then ARCH+=(--ternary); fi
    if [[ "$variant" == *moe* ]]; then ARCH+=(--experts 4 --topk 2 --ehidden 6 --shared 1); fi
    for optimizer in adamw muon; do
      OPT=()
      if [[ "$optimizer" == muon ]]; then OPT+=(--muon); fi
      for dataset in masked sparse; do
        TRAIN=(--data "$TEST_DIR/$dataset.tok" --val "$TEST_DIR/$dataset.tok"
               --seq 8 --dim 8 --heads 2 --ffn 12 --layers 1 --steps 4 --warmup 0
               --seed 3 --lr 0.001 --clip 0.1 --eval-every 1 --log-every 1 --save-every 0 --threads 2)
        "$CXXGPT" train "${TRAIN[@]}" "${ARCH[@]}" "${OPT[@]}" --batch 4 --accum 1 --eval-batches 1 \
            --out "$TEST_DIR/macro.ckpt" --log "$TEST_DIR/macro.csv" > "$TEST_DIR/macro.log"
        "$CXXGPT" train "${TRAIN[@]}" "${ARCH[@]}" "${OPT[@]}" --batch 1 --accum 4 --eval-batches 4 \
            --out "$TEST_DIR/micro.ckpt" --log "$TEST_DIR/micro.csv" > "$TEST_DIR/micro.log"
        ./test_training_cli compare "$TEST_DIR/macro.ckpt" "$TEST_DIR/micro.ckpt" "${OPT[@]}"
        # Identical eval samples must give the same token-weighted metric,
        # including all-ignored eval batches. Ignore timing columns.
        awk -F, 'NR>1 {print $1 "," $2 "," $3}' "$TEST_DIR/macro.csv" > "$TEST_DIR/macro.loss"
        awk -F, 'NR>1 {print $1 "," $2 "," $3}' "$TEST_DIR/micro.csv" > "$TEST_DIR/micro.loss"
        diff -u "$TEST_DIR/macro.loss" "$TEST_DIR/micro.loss"
        if [[ "$dataset" == sparse ]]; then grep -q 'skipped .* empty sampled groups' "$TEST_DIR/micro.log"; fi
        rm "$TEST_DIR/macro.csv" "$TEST_DIR/micro.csv"
      done
    done
done
status=0
"$CXXGPT" train "${COMMON[@]}" --data "$TEST_DIR/ignored.tok" > "$TEST_DIR/ignored.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'no predictable targets' "$TEST_DIR/ignored.log"
printf 'CLI: AdamW/Muon masked/sparse accumulation, weighted validation, empty-group skips, and fully masked rejection passed\n'

"$CXXGPT" bench-infer --vocab 13 --seq 8 --dim 32 --heads 2 --layers 1 --ffn 64 \
    --tokens 8 --reps 4 --warmup 1 --threads 2 --log "$TEST_DIR/decode.csv" > "$TEST_DIR/decode.log"
[[ $(wc -l < "$TEST_DIR/decode.csv") == 5 ]]
awk -F, 'NR>1 {if ($1 != NR-2 || $2 != (NR%2==0) || $5 <= 0 || $6 <= 0 || $7 < 0 || $7 > 1) exit 1}' "$TEST_DIR/decode.csv"
status=0
"$CXXGPT" bench-infer --vocab 13 --seq 8 --dim 32 --heads 2 --layers 1 --ffn 64 \
    --tokens 8 --reps 0 --threads 2 > "$TEST_DIR/bad-reps.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'reps must be positive' "$TEST_DIR/bad-reps.log"
printf 'CLI: benchmark emits alternating raw trials and rejects zero repeats\n'

"$CXXGPT" train "${COMMON[@]}" --steps 0 --init "$TEST_DIR/model.ckpt" --out "$TEST_DIR/fresh.ckpt" > "$TEST_DIR/init.log"
./test_training_cli fresh "$TEST_DIR/model.ckpt" "$TEST_DIR/fresh.ckpt"
status=0
./test_training_cli same-sampler "$TEST_DIR/model.ckpt" "$TEST_DIR/fresh.ckpt" > "$TEST_DIR/rng-different.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'different samples' "$TEST_DIR/rng-different.log"
cp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
"$CXXGPT" train "${COMMON[@]}" --resume --init "$TEST_DIR/does-not-exist.ckpt" > "$TEST_DIR/resume.log"
cmp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
# A skipped seek must still detect a truncated moment array in weight-only loads.
cp "$TEST_DIR/model.ckpt" "$TEST_DIR/truncated.ckpt"
./test_training_cli truncate "$TEST_DIR/truncated.ckpt"
status=0
"$CXXGPT" train "${COMMON[@]}" --steps 0 --init "$TEST_DIR/truncated.ckpt" > "$TEST_DIR/truncated.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'truncated optimizer state' "$TEST_DIR/truncated.log"
for kind in params m v short-v; do
    status=0
    ./test_training_cli bad-save "$TEST_DIR/protected.ckpt" "$kind" > "$TEST_DIR/bad-save.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'check failed' "$TEST_DIR/bad-save.log"
    cmp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
done
status=0
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/unstable.ckpt" --out "$TEST_DIR/protected.ckpt" \
    --steps 20 --save-every 1 > "$TEST_DIR/diverged.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q '10 diverged steps' "$TEST_DIR/diverged.log"
[[ $(grep -c 'step 1: .*update skipped' "$TEST_DIR/diverged.log") == 10 ]]
cmp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
printf 'CLI: warm start/reset, resume precedence, truncated moments and checkpoint protection passed\n'

"$CXXGPT_AUDIT" --ckpt "$TEST_DIR/audit.ckpt" --problems "$TEST_DIR/audit.txt" --threads 2 > "$TEST_DIR/audit.csv"
awk -F, '$2=="all_completion" {if ($3!=13 || $4<=0) exit 1; all=$4*13}
    $2=="step_digit" || $2=="step_other" || $2=="delimiter" || $2=="answer_digit" || $2=="answer_other" {sum+=$3*$4; n+=$3}
    $2=="teacher_coverage" {if ($5!=1 || $6!=1) exit 1}
    END {if (n!=13 || (sum-all)>0.000001 || (all-sum)>0.000001) exit 1}' "$TEST_DIR/audit.csv"
printf 'CLI: reasoning audit covers the gold completion and preserves weighted segment loss\n'

"$CXXGPT_TARGETS" --train "$TEST_DIR/targets.tok" --val "$TEST_DIR/targets.tok" \
    --exclude-test "$TEST_DIR/audit.txt" --out "$TEST_DIR/targets" --per-cell 3 > "$TEST_DIR/targets.log"
./test_training_cli targets "$TEST_DIR/targets.tok" "$TEST_DIR/targets/train-final.tok" "$TEST_DIR/targets/probe.txt"
printf 'CLI: final-digit masks preserve tokens and fresh probes exclude known questions\n'
