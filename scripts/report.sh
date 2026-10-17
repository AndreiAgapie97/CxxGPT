#!/usr/bin/env bash
# A night's run on one screen: the model, the validation loss and exact
# answers over the run, the speed, how it ended, and the last wrong answer.
#   scripts/report.sh [RUN]           (default night1: reads runs/$RUN.log)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-night1}
log=runs/$RUN.log
[ -f "$log" ] || { echo "no $log"; exit 1; }
echo "== $log"
# The settings, once per start or resume.
grep -E "^($RUN: |model: |params: |gpu: |gpu memory: |resumed |starting from )" "$log" | sed 's/^/  /' || true
echo
awk '
  /^step +[0-9]+ \|/ {
    n = $2 + 0
    if (match($0, /val [0-9.]+/)) val[n] = substr($0, RSTART + 4, RLENGTH - 4)
    if (match($0, /[0-9.]+ tok\/s/)) speed = substr($0, RSTART, RLENGTH)
    if (match($0, /loss [0-9.]+/)) train = substr($0, RSTART + 5, RLENGTH - 5)
    last = n
    seen[n] = 1
  }
  /^reasoning eval at step / { at = $5 + 0 }
  /^overall / && at != "" { acc[at] = $2 " " $3 " " $4 " " $5; seen[at] = 1; at = "" }
  # train --reason-tools: the same exam with the calculator, "... at step N: X% (c / t), calls ..."
  /^reasoning with calculator / { n = $7 + 0; c = $8 " " $9 " " $10 " " $11; sub(/,$/, "", c); calc[n] = c; seen[n] = 1 }
  END {
    printf "  %7s  %9s  %-28s  %s\n", "step", "val loss", "exact answers", "with calculator"
    for (n in seen) if (n in val || n in acc || n in calc) steps[++k] = n + 0
    # insertion sort: the awk in every distro, no asort
    for (i = 2; i <= k; i++) { x = steps[i]; for (j = i - 1; j > 0 && steps[j] > x; j--) steps[j + 1] = steps[j]; steps[j + 1] = x }
    for (i = 1; i <= k; i++) printf "  %7d  %9s  %-28s  %s\n", steps[i], val[steps[i]], acc[steps[i]], calc[steps[i]]
    if (last != "") printf "\n  last logged: step %d, training loss %s, %s\n", last, train, speed
  }' "$log"
load_log=runs/$RUN.load.csv
if [ -f "$load_log" ]; then
  awk -F, '
    $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {
      layer = $2 + 0
      shares[layer] = ""
      for (e = 3; e < NF; e++) {
        shares[layer] = shares[layer] sprintf("%s%.4f", e == 3 ? "" : " ", $e)
        if ($e + 0 < 0.01 && !(layer in first)) first[layer] = $1
      }
      seen[layer] = 1
      if (layer > last_layer) last_layer = layer
    }
    END {
      for (layer = 0; layer <= last_layer; layer++) if (layer in seen)
        printf "  layer %d: final shares [%s]; first under 1%%: %s\n", layer, shares[layer],
               (layer in first) ? "step " first[layer] : "never"
    }' "$load_log"
fi
echo
# How it ended (the last start or resume only).
tail -n 50 "$log" | grep -E "^(time limit reached|stopping: |not saved: |saved |step [0-9]+: loss .* update skipped)" | tail -n 3 | sed 's/^/  /' || true
wrong=$(awk '/^--- wrong/ { block = ""; on = 1 } on && /^(--- wrong|    )/ { block = block $0 "\n"; next } { on = 0 }
             END { printf "%s", block }' "$log")
if [ -n "$wrong" ]; then
  echo
  echo "  last wrong answer:"
  printf "%s\n" "$wrong" | sed 's/^/  /'
fi
