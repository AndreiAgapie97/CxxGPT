#!/usr/bin/env bash
# GPU throughput probe: trains each candidate model size for a few steps and
# prints its speed, its GPU memory, and how many tokens a night would train.
#   scripts/probe.sh [steps]          (default 30; needs cxxgpt-gpu: make gpu)
# BATCH=8 SEQ=512 VOCAB=8192 set the shape of a step; extra sizes can be
# given as SIZES="name:--dim 640 --layers 10 --heads 10;...".
# The tokens are synthetic (the reasoning curriculum); speed does not depend
# on what they say, only on the vocabulary size, which VOCAB sets.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-30}
BATCH=${BATCH:-8}
SEQ=${SEQ:-512}
VOCAB=${VOCAB:-8192}
NIGHT_HOURS=${NIGHT_HOURS:-8}
[ -x ./cxxgpt-gpu ] || { echo "build it first: make gpu"; exit 1; }
[ -f data/reason.train.tok ] || ./cxxgpt-gpu gen-reasoning --out data/reason > /dev/null

DEFAULT_SIZES="d512-dense:--dim 512 --layers 8 --heads 8
d512-moe-ternary:--dim 512 --layers 8 --heads 8 --experts 8 --topk 2 --ternary
d768-dense:--dim 768 --layers 12 --heads 12
d768-moe-ternary:--dim 768 --layers 12 --heads 12 --experts 8 --topk 2 --ternary
d1024-moe-ternary:--dim 1024 --layers 16 --heads 16 --experts 8 --topk 2 --ternary"
SIZES=${SIZES:-$DEFAULT_SIZES}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf "%-20s %10s %10s %9s %9s %8s %12s\n" size params active "ms/step" "tok/s" "GPU MB" "tokens/night"
while IFS= read -r line; do
  [ -n "$line" ] || continue
  name=${line%%:*}
  args=${line#*:}
  # shellcheck disable=SC2086
  if ./cxxgpt-gpu train --gpu --data data/reason.train.tok --vocab "$VOCAB" --seq "$SEQ" --batch "$BATCH" \
      --steps "$STEPS" --warmup 5 --eval-every 0 --log-every 10 --save-every 0 --out "$tmp/probe.ckpt" \
      $args > "$tmp/log" 2>&1; then
    params=$(sed -n 's/^params: \([0-9.]* M\) total, \([0-9.]* M\) active.*/\1/p' "$tmp/log")
    active=$(sed -n 's/^params: \([0-9.]* M\) total, \([0-9.]* M\) active.*/\2/p' "$tmp/log")
    # The last log line averages the final 10 steps, after the warm-up ones.
    ms=$(grep "^step" "$tmp/log" | tail -1 | sed 's/.*| *\([0-9.]*\) ms.*/\1/')
    toks=$(grep "^step" "$tmp/log" | tail -1 | sed 's/.*| *\([0-9.]*\) tok\/s.*/\1/')
    mb=$(sed -n 's/^gpu memory: \([0-9]*\) MB.*/\1/p' "$tmp/log")
    night=$(awk -v t="$toks" -v h="$NIGHT_HOURS" 'BEGIN { printf "%.2fB", t * h * 3600 / 1e9 }')
    printf "%-20s %10s %10s %9s %9s %8s %12s\n" "$name" "$params" "$active" "$ms" "$toks" "$mb" "$night"
  elif grep -q "out of memory" "$tmp/log"; then
    printf "%-20s out of GPU memory at batch %d x %d (try BATCH=%d)\n" "$name" "$BATCH" "$SEQ" $((BATCH / 2))
  else
    printf "%-20s failed:\n" "$name"
    tail -5 "$tmp/log"
  fi
done <<< "${SIZES//;/$'\n'}"
echo "tokens/night: at this speed for $NIGHT_HOURS hours. Larger batches come from --accum."
