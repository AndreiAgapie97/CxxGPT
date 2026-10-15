#!/usr/bin/env bash
# The audited training data for the fresh model (issue #46), from the
# downloads in data/public-20261015/ and the GSM8K split in
# data/math-pilot-20261015/ (neither is changed):
#   scripts/prepare_clean.sh            (OUT=data/clean-20261015)
# Writes into $OUT:
#   textbooks, wiki, doab, arxiv (.txt and .tok): scripts/prep_public.py,
#     DOAB's first 2 shards and arXiv's first one;
#   doab-part, arxiv-part: the first documents of doab and arxiv, cut at a
#     document boundary to about 900M and 400M tokens, so that research
#     books and papers do not crowd out the rest of the mix (the full
#     files stay);
#   word-train, word-dev: exact word problems (cxxgpt gen-word, no
#     calculator notes; dev uses the held-out sentence patterns), as masked
#     token files (prep-qa: the question is not trained on, as GSM8K's);
#   gsm8k-dev (.txt and .tok): the 500 GSM8K train problems held out of
#     training, for DEV=1 scripts/night.sh (copied to data/gsm8k/dev.*);
#   gsm8k-train (.txt and .tok): the other GSM8K train problems, without the
#     few near-copies of a test or dev question;
#   manifest.json: what prep_public.py read, kept and dropped, and hashes.
# Then it checks that no GSM8K test, GSM8K dev or SVAMP dev question shares
# a 13-word run with any training file (cxxgpt leak-check). Steps whose
# output exists are skipped. The tokenizer is data/tok.bpe, unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-data/clean-20261015}
SRC=${SRC:-data/public-20261015/common-pile}
PILOT=${PILOT:-data/math-pilot-20261015}
tok=data/tok.bpe
case "$(uname -s)" in
  MINGW* | MSYS*) CX=${CX:-build/cxxgpt.exe}; PY=${PY:-python} ;;
  *) CX=${CX:-./cxxgpt}; PY=${PY:-python3} ;;
esac
mkdir -p "$OUT"

echo "== text"
# One run, so that a text kept in one source is dropped from the later ones.
if [ ! -f "$OUT/arxiv.txt" ]; then
  "$PY" scripts/prep_public.py --src "$SRC" --out "$OUT" --only textbooks,wiki,doab,arxiv --shards doab=2,arxiv=1
fi

echo "== cut doab and arxiv"
cut_docs() {  # SRC DEST SHARE: the documents in the first SHARE of SRC's bytes
  "$PY" - "$@" << 'EOF'
import sys
src, dest, share = sys.argv[1], sys.argv[2], float(sys.argv[3])
sep = b"\n<|endoftext|>\n"
data = open(src, "rb").read()
cut = data.rfind(sep, 0, int(len(data) * share))
open(dest, "wb").write(data[:cut])
print(f"{dest}: the first {data[:cut].count(sep) + 1} documents of {src}, {cut} bytes")
EOF
}
[ -f "$OUT/doab-part.txt" ] || cut_docs "$OUT/doab.txt" "$OUT/doab-part.txt" "$(awk 'BEGIN { print 900 / 1511.5 }')"
[ -f "$OUT/arxiv-part.txt" ] || cut_docs "$OUT/arxiv.txt" "$OUT/arxiv-part.txt" "$(awk 'BEGIN { print 400 / 946.0 }')"

echo "== tokens"
for name in textbooks wiki doab arxiv doab-part arxiv-part; do
  [ -f "$OUT/$name.tok" ] || "$CX" text2tok --in "$OUT/$name.txt" --out "$OUT/$name.tok" --bpe "$tok" --doc-sep '<|endoftext|>'
done

echo "== word problems"
# gen-word's problems file to prep-qa's JSON Lines: the same question,
# solution and final answer, so prep-qa writes the same problems (checked
# byte for byte, apart from the difficulty field) and masks the question.
to_jsonl() {
  "$PY" - "$1" "$2" << 'EOF'
import json, sys
esc = {"n": "\n", "t": "\t", "\\": "\\"}
def unescape(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(esc[s[i + 1]])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)
with open(sys.argv[1], encoding="utf-8", newline="") as f, open(sys.argv[2], "w", encoding="utf-8", newline="\n") as o:
    for line in f:
        task, difficulty, q, steps, answer = line.rstrip("\n").split("\t")
        o.write(json.dumps({"question": unescape(q), "answer": unescape(steps) + "\n#### " + unescape(answer)},
                           ensure_ascii=False) + "\n")
EOF
}
word() {  # NAME SPLIT SEED COUNT
  [ -f "$OUT/word-$1.tok" ] && return
  "$CX" gen-word --out "$OUT/word-$1-gen" --split "$2" --seed "$3" --count "$4" --notes off
  to_jsonl "$OUT/word-$1-gen.txt" "$OUT/word-$1.jsonl"
  "$CX" prep-qa --in "$OUT/word-$1.jsonl" --out "$OUT/word-$1" --bpe "$tok" --task word
  cmp <(cut -f1,3- "$OUT/word-$1.txt") <(cut -f1,3- "$OUT/word-$1-gen.txt")
  # eval-reasoning reads gen-word's own file: its difficulty is the number of steps.
  mv "$OUT/word-$1-gen.txt" "$OUT/word-$1.txt"
  rm "$OUT/word-$1-gen.text" "$OUT/word-$1.jsonl"
}
word train train 101 1000000
word dev test 102 1000

echo "== GSM8K dev"
if [ ! -f "$OUT/gsm8k-dev.tok" ]; then
  "$CX" prep-qa --in "$PILOT/gsm-known-seen-dev.jsonl" --out "$OUT/gsm8k-dev" --bpe "$tok" --task gsm8k
fi
cp "$OUT/gsm8k-dev.txt" data/gsm8k/dev.txt
cp "$OUT/gsm8k-dev.tok" data/gsm8k/dev.tok

echo "== GSM8K train"
# The pilot's 6,973 train problems, without the few that share a 13-word
# run (leak-check's rule: ASCII letters and digits, any case) with a GSM8K
# test or dev question: GSM8K has near-copies of a story with other numbers
# ("A train travels 270 miles in 3 hours..." in train, "A plane travels
# 1200 miles in 3 hours..." in test).
if [ ! -f "$OUT/gsm8k-train.tok" ]; then
  "$PY" - "$PILOT/candidate.jsonl" data/gsm8k/test.txt "$OUT/gsm8k-dev.txt" "$OUT/gsm8k-train.jsonl" << 'EOF'
import json, re, sys
src, test, dev, dest = sys.argv[1:]
def runs(text, n=13):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
held = set()
for path in (test, dev):
    for line in open(path, encoding="utf-8"):
        held |= runs(line.split("\t")[2].replace("\\n", " "))
kept = dropped = 0
with open(dest, "w", encoding="utf-8", newline="\n") as out:
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        if runs(r["question"] + " " + r["answer"]) & held:
            dropped += 1
            print("dropped:", r["question"][:100])
            continue
        out.write(line)
        kept += 1
print(f"{dest}: {kept} problems kept, {dropped} dropped")
EOF
  "$CX" prep-qa --in "$OUT/gsm8k-train.jsonl" --out "$OUT/gsm8k-train" --bpe "$tok" --task gsm8k
  rm "$OUT/gsm8k-train.jsonl"
fi

echo "== leak check"
train="$OUT/textbooks.txt,$OUT/wiki.txt,$OUT/doab-part.txt,$OUT/arxiv-part.txt,$OUT/word-train.txt,$OUT/gsm8k-train.txt"
for q in data/gsm8k/test.txt "$OUT/gsm8k-dev.txt" "$PILOT/svamp-dev.txt"; do
  echo "-- $q"
  "$CX" leak-check --questions "$q" --train "$train"
done
echo "MIX=$OUT/textbooks.tok*2,$OUT/wiki.tok,$OUT/doab-part.tok,$OUT/arxiv-part.tok,$OUT/word-train.tok,$OUT/gsm8k-train.tok*8"
