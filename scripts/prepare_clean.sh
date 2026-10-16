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
#   textbooks-train, wiki-train and textbooks-val, wiki-val, doab-val,
#     arxiv-val: the mix's text and the held-out text it never trains on;
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
# output exists are skipped. The tokenizer is data/tok.bpe: an existing one
# is never changed (models are tied to its token ids); when there is none,
# it is learned from the mix's training text (TOK_VOCAB=16384 tokens, from
# a BPE_SAMPLE_MB=200 sample). GSM8K comes from scripts/prepare_data.sh,
# run from here when data/gsm8k/test.txt is missing.
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

echo "== held-out text"
# Text the model never trains on, for a general-language check besides
# GSM8K: every 100th document of textbooks and wiki (the rest is
# NAME-train), and about 8 MB of the DOAB and arXiv documents that follow
# the cut (never read by the mix).
split_every() {  # SRC TRAIN VAL N
  "$PY" - "$@" << 'EOF'
import sys
src, train, val, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sep = "\n<|endoftext|>\n"
docs = open(src, encoding="utf-8", newline="").read().split(sep)
open(train, "w", encoding="utf-8", newline="").write(sep.join(d for i, d in enumerate(docs) if i % n != n - 1))
open(val, "w", encoding="utf-8", newline="").write(sep.join(d for i, d in enumerate(docs) if i % n == n - 1))
print(f"{src}: {len(docs) - len(docs) // n} documents -> {train}, {len(docs) // n} -> {val}")
EOF
}
after_cut() {  # FULL PART VAL BYTES: whole documents of FULL after PART, about BYTES
  "$PY" - "$@" << 'EOF'
import os, sys
full, part, val, size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sep = b"\n<|endoftext|>\n"
start = os.path.getsize(part) + len(sep)
with open(full, "rb") as f:
    f.seek(start - len(sep))
    assert f.read(len(sep)) == sep, "the part does not end at a document boundary"
    data = f.read(size + 10_000_000)
end = data.find(sep, size)
open(val, "wb").write(data[:end])
print(f"{val}: {data[:end].count(sep) + 1} documents of {full} after {part}, {end} bytes")
EOF
}
for name in textbooks wiki; do
  [ -f "$OUT/$name-val.txt" ] || split_every "$OUT/$name.txt" "$OUT/$name-train.txt" "$OUT/$name-val.txt" 100
done
[ -f "$OUT/doab-val.txt" ] || after_cut "$OUT/doab.txt" "$OUT/doab-part.txt" "$OUT/doab-val.txt" 8000000
[ -f "$OUT/arxiv-val.txt" ] || after_cut "$OUT/arxiv.txt" "$OUT/arxiv-part.txt" "$OUT/arxiv-val.txt" 8000000

echo "== tokenizer"
if [ ! -f "$tok" ]; then
  # A sample of the training text in the shares the mix trains on it (never
  # the held-out text), without the document separators.
  mb=${BPE_SAMPLE_MB:-200}
  "$PY" scripts/bpe_sample.py --out data/bpe-sample.txt --mb "$mb" "$OUT/textbooks-train.txt:0.22" \
    "$OUT/wiki-train.txt:0.16" "$OUT/doab-part.txt:0.43" "$OUT/arxiv-part.txt:0.19"
  # --max-bytes: all of the sample (bpe-train would otherwise keep only its first 200 MB).
  "$CX" bpe-train --in data/bpe-sample.txt --vocab "${TOK_VOCAB:-16384}" --max-bytes "$((mb * 1024 * 1024))" \
    --out "$tok.part"
  mv "$tok.part" "$tok"
  rm data/bpe-sample.txt
fi

echo "== tokens"
for name in doab-part arxiv-part textbooks-train wiki-train textbooks-val wiki-val doab-val arxiv-val; do
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
[ -f data/gsm8k/test.txt ] || CX="$CX" PY="$PY" scripts/prepare_data.sh
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
train="$OUT/textbooks-train.txt,$OUT/wiki-train.txt,$OUT/doab-part.txt,$OUT/arxiv-part.txt,$OUT/word-train.txt,$OUT/gsm8k-train.txt"
for q in data/gsm8k/test.txt "$OUT/gsm8k-dev.txt" "$PILOT/svamp-dev.txt"; do
  echo "-- $q"
  "$CX" leak-check --questions "$q" --train "$train"
done
echo "MIX=$OUT/textbooks-train.tok*2,$OUT/wiki-train.tok,$OUT/doab-part.tok,$OUT/arxiv-part.tok,$OUT/word-train.tok,$OUT/gsm8k-train.tok*8"
