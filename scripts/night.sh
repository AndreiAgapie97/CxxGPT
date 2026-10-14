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
#   MODEL="--dim 768 --layers 12 --heads 12 --experts 8 --topk 2 --ternary"
#   OPT=--muon (or "" for AdamW everywhere)   LR=1.5e-3
#   SEQ=512 BATCH=8 ACCUM=16 (tokens per step: SEQ x BATCH x ACCUM)
#   HOURS=7.5   RUN=night1 (outputs runs/$RUN.*)   DATA=data
#   INIT: a checkpoint to start from (for example runs/night1.ckpt), with a
#         fresh optimizer and schedule; its model replaces MODEL.
#   TERNARY=1 (with INIT): turn the checkpoint's weights ternary from this
#         night on (full-precision nights first, ternary ones at the end).
#   MIX: the training files, comma-separated, "*N" repeats one N times;
#        by default web text, stories, OpenMath, MetaMath and GSM8K train x4.
# Validation is on the GSM8K test set, which training never sees: the loss
# on its worked solutions every 500 steps, and exact answers every 2000.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
RUN=${RUN:-night1}
HOURS=${HOURS:-7.5}
TOKS=${TOKS:?set TOKS to the tokens per second scripts/probe.sh measured for this MODEL}
MODEL=${MODEL:---dim 768 --layers 12 --heads 12 --experts 8 --topk 2 --ternary}
OPT=${OPT---muon}
LR=${LR:-1.5e-3}
SEQ=${SEQ:-512}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-16}
CX=${CX:-./cxxgpt-gpu}
GPU=${GPU---gpu}
if [ -z "${MIX:-}" ]; then
  web=$(find "$DATA/fineweb-edu" -name '[0-9][0-9][0-9].tok' | sort | paste -sd,)
  MIX="$web,$DATA/tinystories/train.tok,$DATA/openmath2/train.tok,$DATA/metamath/train.tok,$DATA/gsm8k/train.tok*4"
fi

per_step=$((SEQ * BATCH * ACCUM))
# 90% of the night: evaluations and checkpoints take the rest.
steps=$(awk -v t="$TOKS" -v h="$HOURS" -v s="$per_step" 'BEGIN { printf "%d", 0.9 * t * h * 3600 / s }')
minutes=$(awk -v h="$HOURS" 'BEGIN { printf "%d", h * 60 + 0.5 }')
mkdir -p runs
resume=()
[ -f "runs/$RUN.ckpt" ] && resume=(--resume)
# Once runs/$RUN.ckpt exists, --resume wins over --init.
[ -n "${INIT:-}" ] && resume+=(--init "$INIT")
# A new model is built from MODEL; with INIT, the model is the checkpoint's
# (MODEL's --ternary must not switch it by accident: that is TERNARY=1).
if [ -n "${INIT:-}" ]; then
  model=()
  [ "${TERNARY:-0}" = 1 ] && model=(--ternary)
else
  read -ra model <<< "$MODEL"
fi
echo "$RUN: $steps steps of $per_step tokens; stops after $minutes minutes at the latest" | tee -a "runs/$RUN.log"
# tee -i: Ctrl-C reaches only the trainer, which saves before it stops.
# shellcheck disable=SC2086
"$CX" train $GPU --data "$MIX" --val "$DATA/gsm8k/test.tok" --bpe "$DATA/tok.bpe" \
  --reason-test "$DATA/gsm8k/test.txt" --reason-every 2000 --reason-limit 250 \
  "${model[@]}" $OPT --seq "$SEQ" --batch "$BATCH" --accum "$ACCUM" --steps "$steps" --warmup 500 --lr "$LR" \
  --eval-every 500 --eval-batches 20 --save-every 1000 --log-every 50 --time-limit "$minutes" \
  --out "runs/$RUN.ckpt" --log "runs/$RUN.csv" "${resume[@]}" "$@" 2>&1 | tee -ia "runs/$RUN.log"
