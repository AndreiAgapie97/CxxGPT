#!/usr/bin/env bash
# Exercise the real CLI after config parsing and backend selection.
set -euo pipefail
cd "$(dirname "$0")/.."
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT
printf 'abcdefghi' > "$TEST_DIR/text.txt"
./cxxgpt text2tok --in "$TEST_DIR/text.txt" --out "$TEST_DIR/data.tok" > "$TEST_DIR/data.log"
COMMON=(--data "$TEST_DIR/data.tok" --val "$TEST_DIR/data.tok" --seq 8 --dim 8 --heads 2
        --ffn 12 --layers 1 --batch 1 --steps 1 --warmup 0 --eval-every 1 --eval-batches 1
        --save-every 0 --threads 2 --out "$TEST_DIR/model.ckpt")
./cxxgpt train "${COMMON[@]}" > "$TEST_DIR/train.log"
grep -q '9 train tokens, 9 val tokens' "$TEST_DIR/train.log"
grep -q '| val ' "$TEST_DIR/train.log"
for option in dim heads experts shared ffn ehidden topk; do
    value=-1
    if [[ "$option" == dim || "$option" == heads || "$option" == topk ]]; then value=0; fi
    status=0
    ./cxxgpt train "${COMMON[@]}" "--$option" "$value" > "$TEST_DIR/invalid.log" 2>&1 || status=$?
    if [[ "$status" != 1 ]] || ! grep -q 'check failed' "$TEST_DIR/invalid.log"; then
        cat "$TEST_DIR/invalid.log"
        printf 'invalid --%s %s returned %s instead of a checked error\n' "$option" "$value" "$status"
        exit 1
    fi
done
printf 'CLI: single-window train/validation and seven invalid configurations passed\n'
