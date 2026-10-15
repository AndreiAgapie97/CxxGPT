#!/usr/bin/env bash
# Adds a run's row to the ledger of training runs, docs/runs.md (or updates
# it): what it started from, the model and settings, how long it trained,
# how many tokens, and how it scored.
#   scripts/ledger.sh RUN ["note"]
# Reads runs/$RUN.log and runs/$RUN.csv, and if they exist runs/$RUN.gpu.csv
# (GPU hours and energy) and runs/$RUN.labels.jsonl (the full test set,
# from eval-reasoning --labels). Without a note, a row being updated keeps
# its note. LEDGER=file writes another ledger (the CLI tests use this).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?usage: scripts/ledger.sh RUN [note]}
LEDGER=${LEDGER:-docs/runs.md}
log=runs/$RUN.log
csv=runs/$RUN.csv
[ -f "$log" ] || { echo "no $log"; exit 1; }
[ -f "$csv" ] || { echo "no $csv"; exit 1; }
[ -f "$LEDGER" ] || { echo "no $LEDGER"; exit 1; }

# A ledger cell: the column'th cell (1 = run name) of the row for a run.
cell() {
  awk -F'|' -v run="$1" -v col="$2" '
    { name = $2; gsub(/ /, "", name) }
    name == run { v = $(col + 1); gsub(/^ +| +$/, "", v); print v; exit }' "$LEDGER"
}

ended=$(date -r "$log" +%F)
from=$(grep -m1 '^starting from the weights in ' "$log" | sed 's/^starting from the weights in //; s|.*/||; s/\.ckpt$//' || true)
resumes=$(grep -c '^resumed ' "$log" || true)
start=${from:-new}
[ "$resumes" -gt 0 ] && start="$start, resumed $resumes x"

# The model: size, MoE or dense, ternary or not.
model=$(awk '
  /^params: / && !p { total = $2; active = $5; p = 1 }
  /^model: / && !m {
    m = 1
    if (match($0, /MoE [0-9]+ experts \(top-[0-9]+/)) { split(substr($0, RSTART, RLENGTH), w, /[ -]/); moe = "MoE " w[2] " x top-" w[5] }
    else moe = "dense"
    if (index($0, "full-precision attention")) tern = ", ternary FFN"
    else if (index($0, "ternary")) tern = ", ternary"
  }
  END { printf "%.0fM (%.0fM active), %s%s", total, active, moe, tern }' "$log")

# The settings the log records: peak learning rate (from the CSV), the GPU
# products and attention, the window sampler and unlikelihood training.
peak=$(awk -F, 'NR > 1 && $1 !~ /^#/ && $4 + 0 > max { max = $4 + 0 } END { printf "%.2e", max }' "$csv")
settings="lr $peak"
gpu=$(grep -m1 '^gpu: ' "$log" || true)
if [ -n "$gpu" ]; then
  matmul=$(sed -E 's/.*, ([a-z0-9]+) matmuls.*/\1/' <<< "$gpu")
  attn=$(sed -E 's/.*matmuls, (.*) attention$/\1/' <<< "$gpu")
  settings="$settings, $matmul, $attn attention"
else
  settings="$settings, CPU"
fi
grep -q '^sampler: qa' "$log" && settings="$settings, sampler qa"
ul=$(grep -m1 '^unlikelihood: alpha ' "$log" | awk '{ print "unlikelihood " $3 " (n " $12 ")" }' || true)
[ -n "$ul" ] && settings="$settings, $ul"

# Steps done (the last checkpoint, or the last logged step) of the planned
# ones, and the tokens they trained.
planned=$(grep -m1 -E "^$RUN: [0-9]+ steps of " "$log" | awk '{ print $2 }' || true)
per_step=$(grep -m1 '^data: ' "$log" | sed -E 's/.* = ([0-9]+) tokens per step$/\1/')
saved=$(grep -E '^saved .* at step [0-9]+$' "$log" | tail -1 | awk '{ print $NF }' || true)
logged=$(awk -F, 'NR > 1 && $1 !~ /^#/ { s = $1 } END { print s + 0 }' "$csv")
steps=$logged
[ -n "$saved" ] && [ "$saved" -gt "$steps" ] && steps=$saved
steps_cell=$steps
[ -n "$planned" ] && steps_cell="$steps of $planned"
tokens=$(awk -v s="$steps" -v t="$per_step" 'BEGIN { printf "%.3f", s * t / 1e9 }')
before=0
if [ -n "$from" ]; then
  before=$(cell "$from" 8)
  [ -n "$before" ] || { echo "$LEDGER has no row for $from, which $RUN started from: add it first"; exit 1; }
fi
total=$(awk -v a="$tokens" -v b="$before" 'BEGIN { printf "%.3f", a + b }')

# Training time: each logged interval's steps times its time per step. A
# resume restarts the count at the checkpoint's step, so steps redone after
# a crash count again: the GPU did them twice.
train_h=$(awk -F, '
  /^# resumed at step / { prev = $1; sub(/.* /, "", prev); next }
  NR > 1 && $1 !~ /^#/ { ms += ($1 - prev) * $6; prev = $1 }
  END { printf "%.2f", ms / 3.6e6 }' "$csv")

# GPU hours and energy, from the 10-second GPU log night.sh keeps (the
# whole night, exams and checkpoints included). A laptop RTX 4070 draws at
# most ~140 W: a reading above 250 W is a glitch (the first one sometimes).
gpu_h="-"
kwh="-"
if [ -s "runs/$RUN.gpu.csv" ]; then
  read -r gpu_h kwh < <(awk -F', ' '
    { n++; w = $4 + 0; if (w > 0 && w < 250) j += w * 10 }
    END { printf "%.2f %.2f\n", n * 10 / 3600, j / 3.6e6 }' "runs/$RUN.gpu.csv")
fi

# The last validation loss; the exam every 500 steps (250 GSM8K test
# questions, greedy), last and best.
val=$(awk -F, 'NR > 1 && $1 !~ /^#/ && $3 + 0 >= 0 { v = $3 } END { if (v == "") print "-"; else printf "%.4f", v }' "$csv")
exam=$(awk '
  /^overall [0-9.]+% \(/ { gsub(/\(/, ""); n = $3 + 0; last = n; if (n > best) best = n; seen = 1 }
  END { if (seen) printf "%d (best %d)", last, best; else print "-" }' "$log")

# The full test set, when eval-reasoning --labels has scored this run.
full="-"
labels=runs/$RUN.labels.jsonl
if [ -f "$labels" ]; then
  answered=$(grep -c '"problem":' "$labels" || true)
  right=$(grep -c '"label":"correct"' "$labels" || true)
  full=$(awk -v r="$right" -v n="$answered" 'BEGIN { printf "%d / %d (%.1f%%)", r, n, n ? 100 * r / n : 0 }')
fi

note=${2-}
[ $# -ge 2 ] || note=$(cell "$RUN" 15)
row="| $RUN | $ended | $start | $model | $settings | $steps_cell | $tokens | $total | $train_h | $gpu_h | $kwh | $val | $exam | $full | $note |"

# Replace the run's row, or add it after the table's last row.
tmp=$LEDGER.tmp
awk -v run="$RUN" -v row="$row" '
  { lines[NR] = $0 }
  /^\|/ { last = NR; split($0, f, "|"); name = f[2]; gsub(/ /, "", name); if (name == run) at = NR }
  END {
    if (!last) { print "no table in the ledger" > "/dev/stderr"; exit 1 }
    for (i = 1; i <= NR; i++) {
      if (i == at) print row
      else print lines[i]
      if (!at && i == last) print row
    }
  }' "$LEDGER" > "$tmp"
mv "$tmp" "$LEDGER"
echo "$row"
