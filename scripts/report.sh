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
  END {
    printf "  %7s  %9s  %s\n", "step", "val loss", "exact answers"
    for (n in seen) if (n in val || n in acc) steps[++k] = n + 0
    # insertion sort: the awk in every distro, no asort
    for (i = 2; i <= k; i++) { x = steps[i]; for (j = i - 1; j > 0 && steps[j] > x; j--) steps[j + 1] = steps[j]; steps[j + 1] = x }
    for (i = 1; i <= k; i++) printf "  %7d  %9s  %s\n", steps[i], val[steps[i]], acc[steps[i]]
    if (last != "") printf "\n  last logged: step %d, training loss %s, %s\n", last, train, speed
  }' "$log"
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
