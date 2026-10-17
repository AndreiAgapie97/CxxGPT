#!/usr/bin/env python3
"""Offline tests of scripts/get_choice_data.py: a local HTTP fake, never the real files.

Every test's converter runs on program-written fixture records in the
upstream layout; the download, the checksum check and the output files
are real. Checking the real sources is `get_choice_data.py --check all`.
"""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "get_choice_data", Path(__file__).resolve().parents[1] / "scripts/get_choice_data.py")
choice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(choice)


def jsonl(rows):
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def zipped(member, data):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(member, data)
    return buf.getvalue()


# Fixture records in each upstream format (made up here, not real questions).
ARC = jsonl([
    {"id": "f1", "question": {"stem": "Which is a fixture?", "choices": [
        {"text": "this one", "label": "A"}, {"text": "that  one", "label": "B"}]}, "answerKey": "B"},
    {"id": "f2", "question": {"stem": "Pick three.", "choices": [
        {"text": "one", "label": "1"}, {"text": "two", "label": "2"}, {"text": "three", "label": "3"}]},
     "answerKey": "3"},
])
SCIQ = json.dumps([{"question": "What is made up?", "distractor1": "a", "distractor2": "b", "distractor3": "c",
                    "correct_answer": "this", "support": "Support text."},
                   {"question": "No support?", "distractor1": "x", "distractor2": "y", "distractor3": "z",
                    "correct_answer": "w", "support": ""}]).encode()
PIQA = jsonl([{"goal": "Open a fixture", "sol1": "pull", "sol2": "push"}])
PIQA_LABELS = b"1\n"
HELLASWAG = jsonl([{"ind": 7, "activity_label": "Fixtures", "ctx_a": "A person writes a test.",
                    "ctx_b": "they", "endings": ["run it [title] twice.", "stop.", "sing.", "go."], "label": "0"}])
BOOLQ = jsonl([{"question": "is this a fixture", "passage": "A made-up passage.", "answer": True}])
PUBMEDQA = json.dumps({"111": {"QUESTION": "Does it work?", "CONTEXTS": ["First part.", "Second part."],
                               "final_decision": "maybe"},
                       "222": {"QUESTION": "Not in the test?", "CONTEXTS": ["x"], "final_decision": "yes"}}).encode()
PUBMEDQA_TEST = json.dumps({"111": "maybe"}).encode()
OBQA = jsonl([{"id": "o1", "question": {"stem": "A fixture is", "choices": [
    {"text": "made up", "label": "A"}, {"text": "real", "label": "B"}]}, "answerKey": "A"}])


class ChoiceDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        files = self.files = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = files.get(self.path.split("?")[0])
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.env = patch.dict(os.environ, {"CHOICE_MIRROR": f"http://127.0.0.1:{self.server.server_port}"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.temp.cleanup()

    def serve(self, test, blobs):
        """Serves a test's files as blobs, with TESTS pinned to their digests."""
        spec = choice.TESTS[test]
        files = [(name, url, hashlib.sha256(b).hexdigest()) for (name, url, _), b in zip(spec["files"], blobs)]
        for (name, _, _), b in zip(spec["files"], blobs):
            self.files[f"/{test}/{name}"] = b
        return patch.dict(choice.TESTS, {test: dict(spec, files=files, size=None)})

    def run_test(self, test, blobs):
        with self.serve(test, blobs), contextlib.redirect_stdout(io.StringIO()):
            choice.get(test, str(self.root), False)
        rows = [json.loads(l) for l in (self.root / f"{test}.jsonl").read_text(encoding="utf-8").splitlines()]
        lines = (self.root / f"{test}.questions.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), len(lines))
        for r in rows:
            self.assertTrue(all(o.startswith(" ") and o.strip() for o in r["options"]), r)
            self.assertTrue(0 <= r["gold"] < len(r["options"]))
        return rows, lines

    def blobs_for(self, test):
        names = [name for name, _, _ in choice.TESTS[test]["files"]]
        table = {"arc-easy": [ARC], "arc-challenge": [ARC], "sciq": [SCIQ], "piqa": [PIQA, PIQA_LABELS],
                 "hellaswag": [HELLASWAG], "boolq": [BOOLQ], "pubmedqa": [PUBMEDQA, PUBMEDQA_TEST],
                 "openbookqa": [OBQA]}
        blobs = table[test]
        # A test whose upstream file is a zip gets the fixture zipped.
        member = choice.TESTS[test].get("member")
        if member:
            blobs = [zipped(member, blobs[0])] + blobs[1:]
        self.assertEqual(len(blobs), len(names), test)
        return blobs

    def test_every_converter(self):
        for test in choice.TESTS:
            with self.subTest(test=test):
                rows, lines = self.run_test(test, self.blobs_for(test))
                self.assertTrue(rows and lines[0])

    def test_formats(self):
        if "arc-easy" in choice.TESTS:
            rows, lines = self.run_test("arc-easy", self.blobs_for("arc-easy"))
            self.assertEqual(rows[0], {"id": "f1", "context": "Question: Which is a fixture?\nAnswer:",
                                       "options": [" this one", " that one"], "gold": 1})
            self.assertEqual(rows[1]["gold"], 2)  # numeric labels
            self.assertEqual(lines[0], "Question: Which is a fixture? Answer: this one that one")
        if "sciq" in choice.TESTS:
            rows, _ = self.run_test("sciq", self.blobs_for("sciq"))
            gold = hashlib.sha256(b"What is made up?").digest()[0] % 4
            self.assertEqual(rows[0]["options"][rows[0]["gold"]], " this")
            self.assertEqual(rows[0]["gold"], gold)
            self.assertEqual(rows[0]["context"], "Support text.\nQuestion: What is made up?\nAnswer:")
            self.assertEqual(rows[1]["context"], "Question: No support?\nAnswer:")
        if "hellaswag" in choice.TESTS:
            rows, _ = self.run_test("hellaswag", self.blobs_for("hellaswag"))
            self.assertEqual(rows[0]["context"], "Fixtures: A person writes a test. They")
            self.assertEqual(rows[0]["options"][0], " run it. twice.")
        if "pubmedqa" in choice.TESTS:
            rows, _ = self.run_test("pubmedqa", self.blobs_for("pubmedqa"))
            self.assertEqual(rows, [{"id": "111", "context": "Abstract: First part. Second part.\n"
                                     "Question: Does it work?\nAnswer:", "options": [" yes", " no", " maybe"],
                                     "gold": 2}])
        if "boolq" in choice.TESTS:
            rows, _ = self.run_test("boolq", self.blobs_for("boolq"))
            self.assertEqual(rows[0]["options"], [" no", " yes"])
            self.assertEqual(rows[0]["gold"], 1)
        if "piqa" in choice.TESTS:
            rows, _ = self.run_test("piqa", self.blobs_for("piqa"))
            self.assertEqual(rows[0]["gold"], 1)

    def test_changed_upstream_file_stops(self):
        test = next(iter(choice.TESTS))
        blobs = self.blobs_for(test)
        with self.serve(test, blobs):
            self.files[f"/{test}/{choice.TESTS[test]['files'][0][0]}"] = blobs[0] + b"\n"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
                choice.get(test, str(self.root), False)
        self.assertIn("sha256", str(stop.exception))
        self.assertFalse((self.root / f"{test}.jsonl").exists())

    def test_rerun_keeps_timestamps(self):
        test = next(iter(choice.TESTS))
        blobs = self.blobs_for(test)
        self.run_test(test, blobs)
        out = self.root / f"{test}.jsonl"
        first = out.read_bytes()
        os.utime(out, (1, 1))
        self.run_test(test, blobs)
        self.assertEqual(out.stat().st_mtime_ns, 1_000_000_000)
        self.assertEqual(out.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
