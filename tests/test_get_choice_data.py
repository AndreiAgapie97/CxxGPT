#!/usr/bin/env python3
"""Offline tests of scripts/get_choice_data.py: a local HTTP fake, never the real files.

Every test's converter runs on fixture records written here in the
upstream layout; the download, the checksum check and the output files are
real. Checking the real sources is `get_choice_data.py --check all`.
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


# Made-up records in each upstream layout (not real questions).
BLOBS = {
    "piqa": [jsonl([{"goal": "Open a fixture", "sol1": "pull  it", "sol2": "push it"},
                    {"goal": "Close it", "sol1": "shut", "sol2": "open"}]), b"1\n0\n"],
    "pubmedqa": [json.dumps({"111": {"QUESTION": "Does it work?", "CONTEXTS": ["First part.", "Second part."],
                                     "LONG_ANSWER": "It does, maybe.", "final_decision": "maybe"},
                             "222": {"QUESTION": "Not in the test?", "CONTEXTS": ["x"],
                                     "final_decision": "yes"}}).encode(),
                 json.dumps({"111": "maybe"}).encode()],
    "boolq": [zipped("BoolQ/val.jsonl", jsonl([
        {"question": "is this a fixture", "passage": "Fixture -- A made-up passage.", "idx": 0, "label": True},
        {"question": "is it real", "passage": "No.", "idx": 1, "label": False}]))],
}


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
        """Serves a test's fixture files, with its pins set to their digests."""
        spec = choice.TESTS[test]
        files = [(name, url, hashlib.sha256(b).hexdigest()) for (name, url, _), b in zip(spec["files"], blobs)]
        for (name, _, _), b in zip(spec["files"], blobs):
            self.files[f"/{test}/{name}"] = b
        return patch.dict(choice.TESTS, {test: dict(spec, files=files, size=None)})

    def run_test(self, test, blobs=None):
        with self.serve(test, blobs or BLOBS[test]), contextlib.redirect_stdout(io.StringIO()):
            choice.get(test, str(self.root), False)
        rows = [json.loads(l) for l in (self.root / f"{test}.jsonl").read_text(encoding="utf-8").splitlines()]
        lines = (self.root / f"{test}.questions.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), len(lines))
        return rows, lines

    def test_every_test_has_fixtures(self):
        self.assertEqual(set(BLOBS), set(choice.TESTS))
        for test, spec in choice.TESTS.items():
            self.assertEqual(len(BLOBS[test]), len(spec["files"]), test)
            for _name, url, digest in spec["files"]:
                self.assertTrue(url.startswith("https://"), url)
                self.assertRegex(digest, "^[0-9a-f]{64}$")

    def test_piqa(self):
        rows, lines = self.run_test("piqa")
        self.assertEqual(rows[0], {"id": "0", "context": "Question: Open a fixture\nAnswer:",
                                   "options": [" pull it", " push it"], "gold": 1})
        self.assertEqual(rows[1]["gold"], 0)
        self.assertEqual(lines[0], "Question: Open a fixture Answer: pull it push it")

    def test_pubmedqa(self):
        # Only the test questions, and never their conclusion (LONG_ANSWER).
        rows, lines = self.run_test("pubmedqa")
        self.assertEqual(rows, [{"id": "111", "context": "Abstract: First part. Second part.\n"
                                 "Question: Does it work?\nAnswer:", "options": [" yes", " no", " maybe"],
                                 "gold": 2}])
        self.assertNotIn("It does", lines[0])
        # A test label that disagrees with the data stops the script.
        blobs = [BLOBS["pubmedqa"][0], json.dumps({"111": "yes"}).encode()]
        with self.assertRaises(SystemExit):
            self.run_test("pubmedqa", blobs)

    def test_boolq(self):
        rows, _ = self.run_test("boolq")
        self.assertEqual(rows[0], {"id": "0", "context": "Fixture -- A made-up passage.\n"
                                   "Question: is this a fixture?\nAnswer:", "options": [" no", " yes"], "gold": 1})
        self.assertEqual(rows[1]["gold"], 0)

    def test_changed_upstream_file_stops(self):
        blobs = BLOBS["piqa"]
        with self.serve("piqa", blobs):
            self.files["/piqa/valid.jsonl"] = blobs[0] + b"\n"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
                choice.get("piqa", str(self.root), False)
        self.assertIn("sha256", str(stop.exception))
        self.assertFalse((self.root / "piqa.jsonl").exists())

    def test_wrong_size_stops(self):
        with self.serve("piqa", BLOBS["piqa"]), patch.dict(choice.TESTS["piqa"], size=1838):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
                choice.get("piqa", str(self.root), False)
        self.assertIn("expected 1838", str(stop.exception))

    def test_rerun_keeps_timestamps(self):
        self.run_test("boolq")
        out = self.root / "boolq.jsonl"
        first = out.read_bytes()
        os.utime(out, (1, 1))
        self.run_test("boolq")
        self.assertEqual(out.stat().st_mtime_ns, 1_000_000_000)
        self.assertEqual(out.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
