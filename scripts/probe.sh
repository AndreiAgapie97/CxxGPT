#!/usr/bin/env bash
# GPU throughput probe: trains each candidate model size for a few steps and
# prints its speed, its GPU memory, and how many tokens a night would train;
# then the scripts/night.sh command for each size that fits.
#   scripts/probe.sh [steps]          (default 20; needs cxxgpt-gpu: make gpu)
# The defaults match scripts/night.sh: BATCH=8 SEQ=512 ACCUM=16 VOCAB=16384,
# OPT=--muon ("" for AdamW). Other sizes:
# SIZES="name:--dim 640 --layers 10 --heads 10;...".
# The tokens are synthetic (the reasoning curriculum); speed does not depend
# on what they say, only on the vocabulary size, which VOCAB sets.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-20}
BATCH=${BATCH:-8}
SEQ=${SEQ:-512}
ACCUM=${ACCUM:-16}
OPT=${OPT---muon}
VOCAB=${VOCAB:-16384}
NIGHT_HOURS=${NIGHT_HOURS:-7.5}
CX=${CX:-./cxxgpt-gpu}
GPU=${GPU---gpu}
[ -x "$CX" ] || { echo "build it first: make gpu"; exit 1; }
[ -f data/reason.train.tok ] || "$CX" gen-reasoning --out data/reason > /dev/null

DEFAULT_SIZES="d512-dense:--dim 512 --layers 8 --heads 8
d512-moe-ternary:--dim 512 --layers 8 --heads 8 --experts 8 --topk 2 --ternary
d640-moe-ternary:--dim 640 --layers 10 --heads 10 --experts 8 --topk 2 --ternary
d768-dense:--dim 768 --layers 12 --heads 12
d768-moe-ternary:--dim 768 --layers 12 --heads 12 --experts 8 --topk 2 --ternary"
SIZES=${SIZES:-$DEFAULT_SIZES}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf "%-20s %10s %10s %8s %9s %9s %8s %12s\n" size params active "batch" "ms/step" "tok/s" "GPU MB" "tokens/night"
cmds=()
# One size at batch b x accumulation a; output in $tmp/log.
run() {
  # shellcheck disable=SC2086
  "$CX" train $GPU --data data/reason.train.tok --vocab "$VOCAB" --seq "$SEQ" --batch "$1" \
    --accum "$2" --steps "$STEPS" --warmup 5 --eval-every 0 --log-every 10 --save-every 0 --out "$tmp/probe.ckpt" \
    $3 $OPT > "$tmp/log" 2>&1
}
while IFS= read -r line; do
  [ -n "$line" ] || continue
  name=${line%%:*}
  args=${line#*:}
  # Out of memory: half the batch, twice the accumulation, same tokens per step.
  b=$BATCH a=$ACCUM ok=0
  while :; do
    if run "$b" "$a" "$args"; then ok=1; break; fi
    if ! grep -qiE "out of memory|ALLOC_FAILED" "$tmp/log" || [ "$b" -le 1 ]; then break; fi
    b=$((b / 2)) a=$((a * 2))
  done
  if [ $ok = 1 ]; then
    params=$(sed -n 's/^params: \([0-9.]* M\) total, \([0-9.]* M\) active.*/\1/p' "$tmp/log")
    active=$(sed -n 's/^params: \([0-9.]* M\) total, \([0-9.]* M\) active.*/\2/p' "$tmp/log")
    # The last log line averages the final 10 steps, after the warm-up ones.
    ms=$(grep "^step" "$tmp/log" | tail -1 | sed 's/.*| *\([0-9.]*\) ms.*/\1/')
    toks=$(grep "^step" "$tmp/log" | tail -1 | sed 's/.*| *\([0-9.]*\) tok\/s.*/\1/')
    mb=$(sed -n 's/^gpu memory: \([0-9]*\) MB.*/\1/p' "$tmp/log")
    night=$(awk -v t="$toks" -v h="$NIGHT_HOURS" 'BEGIN { printf "%.2fB", t * h * 3600 / 1e9 }')
    printf "%-20s %10s %10s %8s %9s %9s %8s %12s\n" "$name" "$params" "$active" "${b}x$a" "$ms" "$toks" "$mb" "$night"
    extra=
    [ "$SEQ" = 512 ] || extra+=" SEQ=$SEQ"
    [ "$OPT" = --muon ] || extra+=" OPT=\"$OPT\""
    cmds+=("$name:"$'\n'"  TOKS=$toks BATCH=$b ACCUM=$a$extra MODEL=\"$args\" scripts/night.sh")
  elif grep -qiE "out of memory|ALLOC_FAILED" "$tmp/log"; then
    printf "%-20s does not fit, even at batch 1 x %d\n" "$name" "$SEQ"
  else
    printf "%-20s failed:\n" "$name"
    tail -5 "$tmp/log"
  fi
done <<< "${SIZES//;/$'\n'}"
echo "batch: sequences per micro-batch x accumulation (BATCH and ACCUM for night.sh); $((BATCH * SEQ * ACCUM)) tokens per step."
echo "tokens/night: at this speed for $NIGHT_HOURS hours; optimizer ${OPT:-AdamW}."
if [ ${#cmds[@]} -gt 0 ]; then
  echo
  echo "To train one of them for a night:"
  printf "%s\n" "${cmds[@]}"
fi
