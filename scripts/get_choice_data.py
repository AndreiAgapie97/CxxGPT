#!/usr/bin/env python3
"""get_choice_data.py: download multiple-choice tests for `cxxgpt eval-choice`.

    python3 scripts/get_choice_data.py piqa pubmedqa boolq   # some tests
    python3 scripts/get_choice_data.py all                   # every test
    python3 scripts/get_choice_data.py --check all           # only check the sources

For evaluation only: none of these files may go into a training mix. The
tests, their licences (checked at the source on 2026-10-17) and what
chance scores:
  - piqa: PIQA's validation set, 1,838 questions on how to do everyday
    things, two solutions each (chance 50%); Academic Free License 3.0.
  - pubmedqa: PubMedQA's 500 expert-labelled test questions, a PubMed
    abstract without its conclusion and a yes/no/maybe question (chance
    33.3%, always "yes" 55.2%); MIT.
  - boolq: BoolQ's dev set, 3,270 yes/no questions on a Wikipedia passage
    (chance 50%, always "yes" 62.2%); CC BY-SA 3.0.

Each upstream file is pinned (a commit, or a file that has not changed
since 2019) and checked against the sha256 written here: a file that
changed upstream stops the script. They are kept unchanged in
data/choice/raw/NAME/ and converted to data/choice/NAME.jsonl, one question
a line, for eval-choice:

    {"id": "...", "context": "Question: ...\\nAnswer:", "options": [" ...", ...], "gold": 1}

Each option starts with the space that separates it from the context
(eval-choice tokenizes the two apart). data/choice/NAME.questions.txt has
each question's own text on one line (the context and its options), in the
same order, for `cxxgpt leak-check --questions` (whose --hits file
eval-choice --skip then leaves out).

The prompts are lm-evaluation-harness's for each test, so scores compare
with published tables of small models (which mostly report the
per-character score, "acc_norm"). Needs only the standard library.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import zipfile

_spec = importlib.util.spec_from_file_location("get_data", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                        "get_data.py"))
get_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(get_data)
log = get_data.log


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(test, name, url, digest, raw):
    """Downloads url to raw/name (kept between runs) and checks its sha256.

    CHOICE_MIRROR (for the tests) replaces every source with
    $CHOICE_MIRROR/TEST/NAME."""
    mirror = os.environ.get("CHOICE_MIRROR")
    if mirror:
        url = f"{mirror}/{test}/{name}"
    path = get_data.download(url, os.path.join(raw, name))
    got = sha256(path)
    if got != digest:
        sys.exit(f"{path}: sha256 {got}, expected {digest} (the upstream file changed: look at it, then at the pin)")
    return path


def read_jsonl(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def item(id_, context, options, gold):
    options = [" " + " ".join(o.split()) for o in options]
    if len(options) < 2 or not 0 <= gold < len(options) or any(o == " " for o in options):
        sys.exit(f"{id_}: a malformed question (gold {gold}, options {options})")
    return {"id": str(id_), "context": context, "options": options, "gold": gold}


# ---- Converters: upstream bytes to eval-choice items, with lm-eval's prompts ----

def piqa(data, labels):
    # "Question: {goal}\nAnswer:" " {sol1}" / " {sol2}"; the labels file has
    # one 0 or 1 a line, in the same order.
    rows, gold = read_jsonl(data), [int(x) for x in labels.decode("utf-8").split()]
    if len(rows) != len(gold):
        sys.exit(f"piqa: {len(rows)} questions, {len(gold)} labels")
    return [item(i, f"Question: {r['goal']}\nAnswer:", [r["sol1"], r["sol2"]], g)
            for i, (r, g) in enumerate(zip(rows, gold))]


def pubmedqa(data, test_ids):
    # "Abstract: {contexts}\nQuestion: {question}\nAnswer:" " yes" / " no" /
    # " maybe", on the 500 test questions (their PMIDs in test_ground_truth),
    # without the abstract's conclusion (LONG_ANSWER), which gives the answer.
    rows, truth = json.loads(data.decode("utf-8")), json.loads(test_ids.decode("utf-8"))
    out = []
    for pmid, answer in truth.items():
        r = rows[pmid]
        if r["final_decision"] != answer:
            sys.exit(f"pubmedqa {pmid}: {r['final_decision']} in the data, {answer} in the test labels")
        context = "Abstract: " + " ".join(r["CONTEXTS"]) + f"\nQuestion: {r['QUESTION']}\nAnswer:"
        out.append(item(pmid, context, ["yes", "no", "maybe"], ["yes", "no", "maybe"].index(answer)))
    return out


def boolq(data):
    # "{passage}\nQuestion: {question}?\nAnswer:" " no" / " yes" (SuperGLUE's
    # copy: BoolQ/val.jsonl, the answer in "label").
    rows = read_jsonl(unzip(data, "BoolQ/val.jsonl"))
    return [item(r.get("idx", i), f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", ["no", "yes"],
                 int(r["label"])) for i, r in enumerate(rows)]


def unzip(data, member):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(member)


PIQA = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/16c6be5562fdf2e56a6d5ee4f39a047494eb0617/piqa/data"
PUBMEDQA = "https://raw.githubusercontent.com/pubmedqa/pubmedqa"

# Each test: its upstream files (name, url, sha256), the converter that makes
# its items from their bytes, and how many questions it has.
TESTS = {
    "piqa": {
        "files": [("valid.jsonl", f"{PIQA}/valid.jsonl",
                   "93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d"),
                  ("valid-labels.lst", f"{PIQA}/valid-labels.lst",
                   "b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb")],
        "convert": piqa, "size": 1838},
    "pubmedqa": {
        "files": [("ori_pqal.json", f"{PUBMEDQA}/1f00b98d5cc626844bf8c4ca513b6e62c40071ec/data/ori_pqal.json",
                   "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d"),
                  ("test_ground_truth.json",
                   f"{PUBMEDQA}/3e7616d288b52edaf38f5fd5c59821c7806bde86/data/test_ground_truth.json",
                   "939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51")],
        "convert": pubmedqa, "size": 500},
    "boolq": {
        "files": [("BoolQ.zip", "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip",
                   "853fbe7922f70c59629f06a39e8d9ca440c3d740e760fd3b87a5ddf3dcba2436")],
        "convert": boolq, "size": 3270},
}


def question_line(it):
    return " ".join((it["context"] + " " + " ".join(o.strip() for o in it["options"])).split())


def write_if_changed(path, text):
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    get_data.replace_if_changed(path + ".part", path)


def get(test, out, check):
    spec = TESTS[test]
    if check:
        for _name, url, _digest in spec["files"]:
            get_data.check_url(url)
        return
    raw = os.path.join(out, "raw", test)
    blobs = []
    for name, url, digest in spec["files"]:
        with open(fetch(test, name, url, digest, raw), "rb") as f:
            blobs.append(f.read())
    items = spec["convert"](*blobs)
    if spec["size"] is not None and len(items) != spec["size"]:
        sys.exit(f"{test}: {len(items)} questions, expected {spec['size']}")
    write_if_changed(os.path.join(out, f"{test}.jsonl"),
                     "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    write_if_changed(os.path.join(out, f"{test}.questions.txt"), "".join(question_line(it) + "\n" for it in items))
    chance = sum(1 / len(it["options"]) for it in items) / len(items)
    log(f"  {test}: {len(items)} questions, chance {100 * chance:.1f}% -> {os.path.join(out, test)}.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(TESTS) + ["all"])
    ap.add_argument("--out", default="data/choice", help="directory (default: data/choice)")
    ap.add_argument("--check", action="store_true", help="only check that the sources answer (a byte each)")
    args = ap.parse_args()
    for name in sorted(TESTS) if "all" in args.names else args.names:
        log(f"== {name}")
        get(name, args.out, args.check)


if __name__ == "__main__":
    main()
