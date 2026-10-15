#!/usr/bin/env bash
# One night of training on the GPU, on the files scripts/prepare_data.sh made.
#   TOKS=<tokens per second> scripts/night.sh [extra train args]
# TOKS is what scripts/probe.sh measured for the chosen MODEL, at the same
# BATCH, ACCUM and SEQ; the probe prints the whole command for each size
# that fits. The step count follows from it, so that the learning rate
# schedule ends when the night does (HOURS). --time-limit stops the run in
# time regardless, with a checkpoint; running the script again resumes it.
#
# Settings (environment), with their defaults:
#   MODEL="--dim 768 --layers 12 --heads 12 --experts 8 --topk 2"
#     (full precision: the first nights train that way, and the last ones
#     switch to ternary with INIT and TERNARY=1; see docs/LAPTOP.md)
#   OPT=--muon (or "" for AdamW everywhere)   LR=1.5e-3
#   SEQ=512 BATCH=8 ACCUM=16 (tokens per step: SEQ x BATCH x ACCUM)
#   HOURS=7.5   RUN=night1 (outputs runs/$RUN.*)   DATA=data
#   INIT: a checkpoint to start from (for example runs/night1.ckpt), with a
#         fresh optimizer and schedule; its model replaces MODEL.
#   TERNARY=1 (with INIT): turn the checkpoint's FFN weights ternary from
#         this night on, attention staying full precision (--fp-attn: the
#         best curriculum run so far, README "Ternary FFN, ...");
#         TERNARY=all turns attention ternary too. Full-precision nights
#         first, ternary ones at the end.
#   MIX: the training files, comma-separated, "*N" repeats one N times;
#        by default web text, stories, OpenMath, MetaMath and GSM8K train x4.
#   DEV=1: use the held-out GSM8K dev files for both validation and exams.
# Validation defaults to the GSM8K test set, which training never sees: the loss
# on its worked solutions every 500 steps, and exact answers every 500 too
# (on the GPU an exam of 250 problems takes about 16 s).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
RUN=${RUN:-night1}
case "${DEV:-0}" in
  0) validation="test" ;;
  1) validation="dev" ;;
  *) echo "DEV must be 0 or 1"; exit 1 ;;
esac
# Keep night1's seed/order unchanged. Later INIT nights get another order;
# resuming still restores the checkpoint RNG. An explicit --seed overrides this.
seed=1
if [[ "$RUN" =~ ([0-9]+) ]]; then seed=${BASH_REMATCH[1]}; fi
HOURS=${HOURS:-7.5}
TOKS=${TOKS:?set TOKS to the tokens per second scripts/probe.sh measured for this MODEL}
MODEL=${MODEL:---dim 768 --layers 12 --heads 12 --experts 8 --topk 2}
OPT=${OPT---muon}
LR=${LR:-1.5e-3}
SEQ=${SEQ:-512}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-16}
# On native Windows (Git Bash) the programs are the CMake build's (README).
case "$(uname -s)" in
  MINGW* | MSYS*) CX=${CX:-build/cxxgpt-gpu.exe} ;;
  *) CX=${CX:-./cxxgpt-gpu} ;;
esac
GPU=${GPU---gpu}
if [ -z "${MIX:-}" ]; then
  web=$(find "$DATA/fineweb-edu" -name '[0-9][0-9][0-9].tok' | sort | paste -sd,)
  MIX="$web,$DATA/tinystories/train.tok,$DATA/openmath2/train.tok,$DATA/metamath/train.tok,$DATA/gsm8k/train.tok*4"
fi

per_step=$((SEQ * BATCH * ACCUM))
# 90% of the night: evaluations and checkpoints take the rest.
steps=$(awk -v t="$TOKS" -v h="$HOURS" -v s="$per_step" 'BEGIN { printf "%d", 0.9 * t * h * 3600 / s }')
minutes=$(awk -v h="$HOURS" 'BEGIN { printf "%d", h * 60 + 0.5 }')
# Validation reads --eval-batches batches of BATCH sequences, drawn in a fixed
# order: 160 sequences whatever BATCH is, so every night checks the same ones.
eval_batches=$(((160 + BATCH - 1) / BATCH))
mkdir -p runs
resume=()
[ -f "runs/$RUN.ckpt" ] && resume=(--resume)
# Once runs/$RUN.ckpt exists, --resume wins over --init.
[ -n "${INIT:-}" ] && resume+=(--init "$INIT")
# A new model is built from MODEL; with INIT, the model is the checkpoint's
# (a --ternary in MODEL must not switch it by accident: that is TERNARY=1).
if [ -n "${INIT:-}" ]; then
  model=()
  case "${TERNARY:-0}" in
    1) model=(--ternary --fp-attn) ;;
    all) model=(--ternary --fp-attn 0) ;;
    0) ;;
    *) echo "TERNARY must be 0, 1 or all"; exit 1 ;;
  esac
else
  read -ra model <<< "$MODEL"
fi
echo "$RUN: $steps steps of $per_step tokens; stops after $minutes minutes at the latest" | tee -a "runs/$RUN.log"
# Every 10 s, the GPU's clock, temperature, power, load, memory and slowdown
# reasons, with the last logged step, into runs/$RUN.gpu.csv: when a stretch
# of the night runs slow, this says what the GPU was doing then.
# (Windows has it on the PATH; under WSL it lives in /usr/lib/wsl/lib, not
# always on the PATH.)
smi=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
if [ -x "$smi" ]; then
  (
    # A failed query (or no step logged yet) must not stop the logger.
    set +e +o pipefail
    while sleep 10; do
      gpu=$("$smi" --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active \
        --format=csv,noheader 2>&1 | head -1)
      last=$(tail -n 100 "runs/$RUN.log" 2> /dev/null | grep '^step' | tail -1 | cut -d'|' -f1)
      echo "$gpu, ${last:-start}"
    done
  ) >> "runs/$RUN.gpu.csv" &
  gpu_log=$!
  trap 'kill "$gpu_log" 2> /dev/null || true' EXIT
fi
# tee -i: Ctrl-C reaches only the trainer, which saves before it stops.
# shellcheck disable=SC2086
"$CX" train $GPU --data "$MIX" --val "$DATA/gsm8k/$validation.tok" --bpe "$DATA/tok.bpe" \
  --reason-test "$DATA/gsm8k/$validation.txt" --reason-every 500 --reason-limit 250 \
  "${model[@]}" $OPT --seq "$SEQ" --batch "$BATCH" --accum "$ACCUM" --steps "$steps" --warmup 500 --lr "$LR" \
  --eval-every 500 --eval-batches "$eval_batches" --save-every 1000 --log-every 50 --time-limit "$minutes" \
  --seed "$seed" --out "runs/$RUN.ckpt" --log "runs/$RUN.csv" --load-log "runs/$RUN.load.csv" "${resume[@]}" "$@" 2>&1 | tee -ia "runs/$RUN.log"
