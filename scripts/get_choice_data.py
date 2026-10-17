#!/usr/bin/env python3
"""get_choice_data.py: download multiple-choice tests for `cxxgpt eval-choice`.

    python3 scripts/get_choice_data.py arc-easy arc-challenge ...   # some tests
    python3 scripts/get_choice_data.py all                          # every test
    python3 scripts/get_choice_data.py --check all                  # only check the sources

For evaluation only: none of these files may go into a training mix.

Each test's upstream files are pinned (a commit or a versioned file) and
checked against the sha256 written here; a file that changed upstream stops
the script. They are kept unchanged in data/choice/raw/NAME/, and converted
to data/choice/NAME.jsonl, one question a line, for eval-choice:

    {"id": "...", "context": "Question: ...\\nAnswer:", "options": [" ...", ...], "gold": 1}

Each option starts with the space that separates it from the context
(eval-choice tokenizes the two apart). data/choice/NAME.questions.txt has
each question's own text on one line (the context and its options), in the
same order, for `cxxgpt leak-check --questions` (its --hits file then
leaves leaked questions out: eval-choice --skip).

The formats follow lm-evaluation-harness's prompts for each test, so scores
compare with published tables of small models (which mostly report the
per-character score, "acc_norm"). Needs only the standard library.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
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
        sys.exit(f"{path}: sha256 {got}, expected {digest} (the upstream file changed: check it, then the pin)")
    return path


def read_jsonl(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def item(id_, context, options, gold):
    options = [" " + " ".join(o.split()) for o in options]
    if not 0 <= gold < len(options) or len(options) < 2 or any(o == " " for o in options):
        sys.exit(f"{id_}: a malformed question (gold {gold}, {len(options)} options)")
    return {"id": str(id_), "context": context, "options": options, "gold": gold}


def letter_index(key, labels):
    """ARC and OpenBookQA keys: "A".."E", sometimes "1".."5"."""
    if key not in labels:
        sys.exit(f"answer key {key!r} is not one of the labels {labels}")
    return labels.index(key)


# ---- Converters: upstream bytes to eval-choice items ----

def arc(data):
    # lm-eval's arc_easy / arc_challenge: "Question: {stem}\nAnswer:" " {choice}".
    out = []
    for r in read_jsonl(data):
        q = r["question"]
        labels = [c["label"] for c in q["choices"]]
        out.append(item(r["id"], f"Question: {q['stem']}\nAnswer:", [c["text"] for c in q["choices"]],
                        letter_index(r["answerKey"], labels)))
    return out


def openbookqa(data):
    # lm-eval's openbookqa: the stem alone, then " {choice}".
    out = []
    for r in read_jsonl(data):
        q = r["question"]
        labels = [c["label"] for c in q["choices"]]
        out.append(item(r["id"], q["stem"], [c["text"] for c in q["choices"]], letter_index(r["answerKey"], labels)))
    return out


def sciq(data):
    # lm-eval's sciq: "{support}\nQuestion: {question}\nAnswer:" " {choice}".
    # Upstream the right answer is a field of its own; it goes to a place
    # chosen by the question's sha256 (likelihood scoring does not care
    # where, but "always option N" then means what it says).
    out = []
    for i, r in enumerate(json.loads(data.decode("utf-8"))):
        gold = hashlib.sha256(r["question"].encode("utf-8")).digest()[0] % 4
        options = [r["distractor1"], r["distractor2"], r["distractor3"]]
        options.insert(gold, r["correct_answer"])
        context = f"{r['support'].strip()}\nQuestion: {r['question']}\nAnswer:".lstrip()
        out.append(item(i, context, options, gold))
    return out


def piqa(data, labels):
    # lm-eval's piqa: "Question: {goal}\nAnswer:" " {sol}".
    rows, gold = read_jsonl(data), [int(x) for x in labels.decode().split()]
    if len(rows) != len(gold):
        sys.exit(f"piqa: {len(rows)} questions, {len(gold)} labels")
    return [item(i, f"Question: {r['goal']}\nAnswer:", [r["sol1"], r["sol2"]], g)
            for i, (r, g) in enumerate(zip(rows, gold))]


def hellaswag_text(text):
    # lm-eval's preprocessing: WikiHow's "[title]" markers out, spaces tidied.
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def hellaswag(data):
    # lm-eval's hellaswag: "{activity}: {ctx_a} {Ctx_b}" " {ending}".
    out = []
    for r in read_jsonl(data):
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        context = hellaswag_text(r["activity_label"] + ": " + ctx)
        out.append(item(r["ind"], context, [hellaswag_text(e) for e in r["endings"]], int(r["label"])))
    return out


def boolq(data):
    # lm-eval's boolq: "{passage}\nQuestion: {question}?\nAnswer:" " no" / " yes".
    return [item(i, f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", ["no", "yes"], int(r["answer"]))
            for i, r in enumerate(read_jsonl(data))]


def pubmedqa(data, test_ids):
    # lm-eval's pubmedqa: "Abstract: {contexts}\nQuestion: {question}\nAnswer:"
    # " yes" / " no" / " maybe", on the 500 expert-labelled test questions.
    rows, ids = json.loads(data.decode("utf-8")), json.loads(test_ids.decode("utf-8"))
    out = []
    for pmid in ids:
        r = rows[pmid]
        context = "Abstract: " + " ".join(r["CONTEXTS"]) + f"\nQuestion: {r['QUESTION']}\nAnswer:"
        out.append(item(pmid, context, ["yes", "no", "maybe"], ["yes", "no", "maybe"].index(r["final_decision"])))
    return out


def unzip(data, member):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(member)


# Each test: its upstream files (name, url, sha256), how to make its items
# from their bytes, and the facts the README lists.
TESTS = {}


def question_line(it):
    text = it["context"] + " " + " ".join(o.strip() for o in it["options"])
    return " ".join(text.split())


def write_if_changed(path, text):
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    get_data.replace_if_changed(path + ".part", path)


def get(test, out, check):
    spec = TESTS[test]
    if check:
        for name, url, _digest in spec["files"]:
            get_data.check_url(url)
        return
    raw = os.path.join(out, "raw", test)
    blobs = []
    for name, url, digest in spec["files"]:
        with open(fetch(test, name, url, digest, raw), "rb") as f:
            blobs.append(f.read())
    items = spec["convert"](*blobs)
    if spec.get("size") and len(items) != spec["size"]:
        sys.exit(f"{test}: {len(items)} questions, expected {spec['size']}")
    write_if_changed(os.path.join(out, f"{test}.jsonl"),
                     "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    write_if_changed(os.path.join(out, f"{test}.questions.txt"), "".join(question_line(it) + "\n" for it in items))
    chance = sum(1 / len(it["options"]) for it in items) / len(items)
    log(f"  {test}: {len(items)} questions, chance {100 * chance:.1f}% -> {out}/{test}.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(TESTS) + ["all"])
    ap.add_argument("--out", default="data/choice", help="directory (default: data/choice)")
    ap.add_argument("--check", action="store_true", help="only check that the sources answer (downloads a byte each)")
    args = ap.parse_args()
    names = sorted(TESTS) if "all" in args.names else args.names
    for name in names:
        log(f"== {name}")
        get(name, args.out, args.check)


if __name__ == "__main__":
    main()
