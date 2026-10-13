#!/usr/bin/env bash
# From nothing to training-ready token files: downloads the public datasets,
# trains our tokenizer on a sample of them, and encodes everything.
#   scripts/prepare_data.sh
# Settings (environment): VOCAB=16384 (tokenizer size), FINEWEB_FILES=1
# (about 2 GB of web text each), BPE_SAMPLE_MB=300 (text the tokenizer
# learns from), DATA=data (where everything goes).
# Each step is skipped when its output exists, so it is safe to rerun after
# an interruption. Takes an hour or more, mostly downloading.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
VOCAB=${VOCAB:-16384}
FINEWEB_FILES=${FINEWEB_FILES:-1}
BPE_SAMPLE_MB=${BPE_SAMPLE_MB:-300}
PY=${PY:-.venv/bin/python}
CX=${CX:-./cxxgpt}
[ -x "$CX" ] || { echo "build it first: make"; exit 1; }

echo "== download (resumes where it stopped)"
"$PY" scripts/get_data.py --out "$DATA" gsm8k tinystories metamath openmath2
"$PY" scripts/get_data.py --out "$DATA" fineweb-edu --files "$FINEWEB_FILES"

tok="$DATA/tok.bpe"
echo "== tokenizer: $VOCAB tokens"
if [ ! -f "$tok" ]; then
  # Mostly web text, some stories and worked math, without the document
  # separators (they are not text the model reads).
  mb=$BPE_SAMPLE_MB
  {
    head -c "$((mb * 6 / 10))M" "$DATA/fineweb-edu/000.txt"
    head -c "$((mb * 2 / 10))M" "$DATA/tinystories/train.txt"
    { "$PY" -c 'import json, sys
for line in sys.stdin:
    j = json.loads(line)
    print(j["question"]); print(j["answer"])' < "$DATA/openmath2/train_1M.jsonl" 2> /dev/null || true; } |
      head -c "$((mb * 2 / 10))M"
  } | grep -v -x '<|endoftext|>' > "$DATA/bpe-sample.txt"
  "$CX" bpe-train --in "$DATA/bpe-sample.txt" --vocab "$VOCAB" --out "$tok.part"
  mv "$tok.part" "$tok"
  rm "$DATA/bpe-sample.txt"
else
  echo "have $tok"
fi

echo "== token files"
text() {  # text with <|endoftext|> between documents -> tokens
  if [ -f "$2" ]; then echo "have $2"; return; fi
  "$CX" text2tok --in "$1" --out "$2.part" --bpe "$tok" --doc-sep '<|endoftext|>'
  mv "$2.part" "$2"
}
qa() {  # question/answer JSON Lines -> tokens trained only on the answers
  if [ -f "$2.tok" ]; then echo "have $2.tok"; return; fi
  "$CX" prep-qa --in "$1" --out "$2.part" --bpe "$tok" --task "$3"
  mv "$2.part.txt" "$2.txt"
  mv "$2.part.tok" "$2.tok"
}
for f in "$DATA"/fineweb-edu/[0-9][0-9][0-9].txt; do text "$f" "${f%.txt}.tok"; done
text "$DATA/tinystories/train.txt" "$DATA/tinystories/train.tok"
qa "$DATA/gsm8k/train.jsonl" "$DATA/gsm8k/train" gsm8k
qa "$DATA/gsm8k/test.jsonl" "$DATA/gsm8k/test" gsm8k
qa "$DATA/openmath2/train_1M.jsonl" "$DATA/openmath2/train" openmath
qa "$DATA/metamath/train.jsonl" "$DATA/metamath/train" metamath

echo "== ready"
for f in "$DATA"/*/*.tok; do
  # The header is 24 bytes; then 2 bytes per token, plus 1 per token if masked.
  printf "  %-36s %8.1f MB\n" "$f" "$(awk -v b="$(stat -c %s "$f")" 'BEGIN { print b / 1e6 }')"
done
echo "Tokenizer: $tok. GSM8K test (never trained on): $DATA/gsm8k/test.txt"
