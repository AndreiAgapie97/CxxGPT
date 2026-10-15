#!/usr/bin/env python3
"""bpe_sample.py: a tokenizer-fitting sample drawn from the training mix.

    python scripts/bpe_sample.py --out data/tok2/bpe-sample.txt --mb 200 \
        data/clean-20261015/textbooks.txt:0.21 data/clean-20261015/wiki.txt:0.15 ...

Each FILE:SHARE gives SHARE of the sample's bytes, from whole documents of
FILE picked at random (fixed --seed). Text files are documents separated by
"<|endoftext|>" lines; a problems file (*.txt with five tab-separated fields,
as prep-qa and gen-word write) gives each problem's question and worked
solution. The separators are not written: they are not text the model
reads. Use only training files, never a dev or test set.
"""
import argparse
import random

SEP = "\n<|endoftext|>\n"


def documents(path):
    with open(path, encoding="utf-8", newline="") as f:
        first = f.readline()
        f.seek(0)
        if first.count("\t") == 4:  # a problems file
            for line in f:
                _, _, q, steps, answer = line.rstrip("\n").split("\t")
                text = q + "\n" + steps + "\n#### " + answer
                yield text.replace("\\n", "\n")
        else:
            yield from f.read().split(SEP)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("sources", nargs="+", help="FILE:SHARE")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for item in args.sources:
            path, share = item.rsplit(":", 1)
            budget = int(float(share) * args.mb * 1e6)
            docs = list(documents(path))
            rng.shuffle(docs)
            written = n = 0
            for d in docs:
                if written >= budget:
                    break
                out.write(d.strip() + "\n")
                written += len(d.encode("utf-8")) + 1
                n += 1
            print(f"{path}: {n} of {len(docs)} documents, {written / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
