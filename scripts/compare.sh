#!/usr/bin/env bash
# Trains the four architecture variants on the reasoning curriculum with the
# same budget and settings, then evaluates each on the full held-out test set.
#   scripts/compare.sh [steps] [extra train args...]
# Needs data/reason.{train,val}.tok and data/reason.test.txt
# (cxxgpt gen-reasoning --out data/reason). Logs go to runs/.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-2500}
shift || true
mkdir -p runs
COMMON=(--data data/reason.train.tok --val data/reason.val.tok --seq 256 --dim 256 --layers 4
        --heads 4 --batch 16 --steps "$STEPS" --lr 1.5e-3 --warmup 150 --eval-every 500
        --log-every 50 --reason-test data/reason.test.txt --reason-every 500 --reason-limit 240
        --bias-rate 2e-3 --time-limit 40 "$@")
declare -A VARIANTS=(
  [dense]=""
  [dense-ternary]="--ternary"
  [moe]="--experts 8 --topk 2"
  [moe-ternary]="--experts 8 --topk 2 --ternary"
)
for name in dense dense-ternary moe moe-ternary; do
  echo "=== $name ($(date -u +%H:%M:%S))"
  # shellcheck disable=SC2086
  ./cxxgpt train "${COMMON[@]}" ${VARIANTS[$name]} --out "runs/$name.ckpt" --log "runs/$name.csv" \
    > "runs/$name.log" 2>&1
  ./cxxgpt eval-reasoning --ckpt "runs/$name.ckpt" --test data/reason.test.txt --show-failures \
    > "runs/$name.eval" 2>&1
  grep -E "overall" "runs/$name.eval"
done
echo "=== done ($(date -u +%H:%M:%S))"
