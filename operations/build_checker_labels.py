"""The hand-labelled self-train tries, for measuring self-train's checks.

self-train keeps a model's try only if its answer is right and its work
passes a list of checks. How often is a kept try really a clean solution,
and how often does a check drop one? This script writes the labelled set
the scorer (`cxxgpt score-checks`) reads: every try with the right answer
from two 32-try self-train audits of rp1 (2026-10-17, runs/rp1today.ckpt,
temp 0.7, seed 1, --tools calc-notes), on the LibreTexts textbook problems
and the GSM8K dev problems, with a label given by hand, blind to the
checks' verdicts:

  clean   every statement true, every calculation right and needed; the
          same steps would give the right answer with other numbers
  padded  a sound method, but also something not to train on: a
          calculation never used, a false or nonsense sentence, a number
          copied wrong (`minor`: one wrong word or unit, the rest clean)
  lucky   the answer does not follow from a sound method

Tries the checks call duplicates (the same outline as a kept try) are left
out. Each problem is in the tune half or the held-out half (the first byte
of the sha256 of its question, odd: held out): rules are tried on the tune
half, and the held-out half gives the number to believe.

Each line has the problem as self-train saw it (question, steps, answer:
the problem files' fields), the try's text, the label and a note, and for
the textbook problems the page's address and licence (CC BY 4.0). GSM8K is
MIT-licensed; the tries are our own model's output. Usage, from the
repository root:

  python -I operations/build_checker_labels.py --labels LABELS.tsv \\
      --textbook data/libretexts-problems/problems.txt --textbook-meta data/libretexts-problems/problems.jsonl \\
      --textbook-audit runs/rp1today.vote32-textbook.audit.jsonl \\
      --gsm8k data/licensed-10b/gsm8k-dev.txt --gsm8k-audit runs/rp1today.vote32-gsm8k-dev.audit.jsonl \\
      --out tests/checker_labels.jsonl

LABELS.tsv: one line per try, "tb:PROBLEM:TRY" or "gs:PROBLEM:TRY", the
label, and the note (a note starting "minor:" marks a minor slip).
"""
import argparse
import hashlib
import json
import pathlib
import sys


def unescape(field):
    """reasoning.h's unescape_field."""
    out, i = [], 0
    while i < len(field):
        c = field[i]
        if c != "\\" or i + 1 == len(field):
            out.append(c)
            i += 1
            continue
        n = field[i + 1]
        out.append("\t" if n == "t" else "\n" if n == "n" else n)
        i += 2
    return "".join(out)


def load_problems(path):
    """reasoning.h's load_problems: task, difficulty, question, steps, answer."""
    problems = []
    for line in pathlib.Path(path).read_bytes().decode("utf-8").split("\n"):
        line = line.removesuffix("\r")
        if not line:
            continue
        fields = line.split("\t")
        assert len(fields) == 5, line
        problems.append({"question": unescape(fields[2]), "steps": unescape(fields[3]), "answer": unescape(fields[4])})
    return problems


def load_texts(path):
    texts = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "try" in r:
            texts[(r["problem"], r["try"])] = r["text"]
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--textbook", required=True)
    ap.add_argument("--textbook-meta", required=True)
    ap.add_argument("--textbook-audit", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--gsm8k-audit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sets = {
        "tb": ("textbook", load_problems(a.textbook), load_texts(a.textbook_audit)),
        "gs": ("gsm8k-dev", load_problems(a.gsm8k), load_texts(a.gsm8k_audit)),
    }
    meta = [json.loads(l) for l in open(a.textbook_meta, encoding="utf-8")]
    assert len(meta) == len(sets["tb"][1]), "problems.jsonl and problems.txt differ in length"
    for m, p in zip(meta, sets["tb"][1]):
        assert m["question"] == p["question"], m["question"]

    out = []
    for line in open(a.labels, encoding="utf-8"):
        key, label, note = line.rstrip("\n").split("\t")
        tag, problem, attempt = key.split(":")
        problem, attempt = int(problem), int(attempt)
        assert label in ("clean", "padded", "lucky"), key
        name, problems, texts = sets[tag]
        p = problems[problem]
        record = {
            "set": name,
            "split": "held-out" if hashlib.sha256(p["question"].encode()).digest()[0] % 2 else "tune",
            "problem": problem,
            "try": attempt,
            "question": p["question"],
            "steps": p["steps"],
            "answer": p["answer"],
            "text": texts[(problem, attempt)],
            "label": label,
            "minor": note.startswith("minor:"),
            "note": note.removeprefix("minor:").strip(),
        }
        if tag == "tb":
            m = meta[problem]
            record.update(source=m["source"], url=m["url"], license=m["license"])
        else:
            record.update(source="GSM8K (MIT licence), a train problem held out as dev by scripts/get_data.py")
        out.append(record)
    out.sort(key=lambda r: (r["set"] != "textbook", r["problem"], r["try"]))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in out:
        k = (r["set"], r["split"], r["label"] if not r["minor"] else "minor")
        counts[k] = counts.get(k, 0) + 1
    print(f"{len(out)} labelled tries -> {a.out}")
    for k in sorted(counts):
        print(" ", *k, counts[k])


if __name__ == "__main__":
    sys.exit(main())
