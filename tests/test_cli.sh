#!/usr/bin/env bash
# Exercise the real CLI after config parsing and backend selection.
set -euo pipefail
CXXGPT=${CXXGPT:-./cxxgpt}
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
