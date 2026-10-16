#!/usr/bin/env bash
# GSM8K, ready to train and test on: downloads it (train, the 500 dev
# problems held out of it, test) and encodes it with our tokenizer,
# data/tok.bpe. The rest of the training mix, and the tokenizer when there
# is none yet, come from scripts/prepare_clean.sh, which runs this script
# itself when GSM8K is missing.
#   scripts/prepare_data.sh
# Settings (environment): DATA=data (where everything goes).
# Each step is skipped when its output exists, so it is safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
case "$(uname -s)" in
  MINGW* | MSYS*) PY=${PY:-python} CX=${CX:-build/cxxgpt.exe} build="cmake (README)" ;;
  *) PY=${PY:-python3} CX=${CX:-./cxxgpt} build="make" ;;
esac
[ -x "$CX" ] || { echo "build it first: $build"; exit 1; }

echo "== download (resumes where it stopped)"
"$PY" scripts/get_data.py --out "$DATA" gsm8k

tok="$DATA/tok.bpe"
# Models are tied to their tokenizer's token ids: this script never makes or
# changes one. prepare_clean.sh learns it from the clean mix's text.
[ -f "$tok" ] || { echo "no $tok yet: scripts/prepare_clean.sh learns it (and runs this script)"; exit 1; }

echo "== token files"
qa() {  # question/answer JSON Lines -> tokens trained only on the answers
  if [ -f "$2.tok" ] && [ -f "$2.txt" ] && [ "$2.tok" -nt "$1" ] && [ "$2.txt" -nt "$1" ]; then
    echo "have $2.tok"; return
  fi
  "$CX" prep-qa --in "$1" --out "$2.part" --bpe "$tok" --task "$3"
  mv "$2.part.txt" "$2.txt"
  mv "$2.part.tok" "$2.tok"
}
qa "$DATA/gsm8k/train.jsonl" "$DATA/gsm8k/train" gsm8k
qa "$DATA/gsm8k/dev.jsonl" "$DATA/gsm8k/dev" gsm8k
qa "$DATA/gsm8k/test.jsonl" "$DATA/gsm8k/test" gsm8k

echo "== ready"
for f in "$DATA"/gsm8k/*.tok; do
  # The header is 24 bytes; then 2 bytes per token, plus 1 per token if masked.
  printf "  %-36s %8.1f MB\n" "$f" "$(awk -v b="$(stat -c %s "$f")" 'BEGIN { print b / 1e6 }')"
done
echo "Tokenizer: $tok. GSM8K test (never trained on): $DATA/gsm8k/test.txt"
