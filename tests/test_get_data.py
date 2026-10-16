#!/usr/bin/env python3
"""Offline GSM8K download tests: a local HTTP fake, never the real files.

The download, dev-split and rerun paths are real. Checking the real files
and their fields remains CI's data-sources job, which has network access.
"""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("get_data", Path(__file__).resolve().parents[1] / "scripts/get_data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.files = {}
        fixtures = self.files

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                if path not in fixtures:
                    self.send_error(404)
                    return
                body = fixtures[path]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.env = patch.dict(os.environ, {"GSM8K_ENDPOINT": self.url + "/gsm8k"})
        self.env.start()
        self.args = types.SimpleNamespace(out=str(self.root), check=False)
        # Program-created fixture records, never real training data.
        self.rows = [{"question": f"Fixture {i}: compute {i} plus 1?", "answer": f"Add one.\n#### {i + 1}"}
                     for i in range(620)]
        self.files["/gsm8k/train.jsonl"] = self.jsonl(self.rows)
        self.files["/gsm8k/test.jsonl"] = self.jsonl([{"question": "Test fixture", "answer": "#### 1"}])

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.temp.cleanup()

    @staticmethod
    def jsonl(rows):
        return "".join(json.dumps(row) + "\n" for row in rows).encode()

    def read(self, relative):
        return [json.loads(line) for line in (self.root / relative).read_text().splitlines()]

    def split(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data.get_gsm8k(str(self.root / "gsm8k"), self.args)

    def test_split_deterministic_disjoint_and_upgrade(self):
        # Old preparations kept the full upstream file at train.jsonl.
        (self.root / "gsm8k").mkdir()
        (self.root / "gsm8k/train.jsonl").write_bytes(self.files["/gsm8k/train.jsonl"])
        self.split()
        train, dev = self.read("gsm8k/train.jsonl"), self.read("gsm8k/dev.jsonl")
        self.assertEqual((len(train), len(dev)), (120, 500))
        tq, dq = {r["question"] for r in train}, {r["question"] for r in dev}
        self.assertFalse(tq & dq)
        self.assertEqual(tq | dq, {r["question"] for r in self.rows})
        first = (self.root / "gsm8k/dev.jsonl").read_bytes()
        timestamp = (self.root / "gsm8k/dev.jsonl").stat().st_mtime_ns
        self.split()
        self.assertEqual(first, (self.root / "gsm8k/dev.jsonl").read_bytes())
        self.assertEqual(timestamp, (self.root / "gsm8k/dev.jsonl").stat().st_mtime_ns)
        # Lock the choice (the 500 questions with the smallest sha256) and the
        # file layout (upstream order) across future changes.
        self.assertEqual(hashlib.sha256(first).hexdigest(), "2a9b35bfc43d286e8a47108cfc16ec12f5d3300c56544a97501e9a014c7caf49")
        self.assertEqual((self.root / "gsm8k/test.jsonl").read_bytes(), self.files["/gsm8k/test.jsonl"])
        # The choice depends on the question texts only, not on the row order.
        self.files["/gsm8k/train.jsonl"] = self.jsonl(self.rows[::-1])
        os.remove(self.root / "gsm8k/raw/train.jsonl")
        self.split()
        self.assertEqual({r["question"] for r in self.read("gsm8k/dev.jsonl")}, dq)
        # The rule itself, checked against a direct reading of it.
        digests = sorted(hashlib.sha256(q.encode()).hexdigest() for q in tq | dq)
        self.assertEqual({hashlib.sha256(q.encode()).hexdigest() for q in dq}, set(digests[:500]))

    def test_identical_rerun_keeps_timestamps(self):
        # Run the real script, as prepare_data.sh does, twice. The second run
        # writes the same bytes, so it must leave the files (and their
        # timestamps) alone; otherwise prepare_data.sh re-encodes GSM8K on
        # every rerun.
        script = Path(__file__).resolve().parents[1] / "scripts/get_data.py"
        env = dict(os.environ, GSM8K_ENDPOINT=self.url + "/gsm8k")

        def run():
            subprocess.run([sys.executable, str(script), "gsm8k", "--out", str(self.root)], env=env,
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        outputs = ["gsm8k/train.jsonl", "gsm8k/dev.jsonl"]
        run()
        first = {name: (self.root / name).read_bytes() for name in outputs}
        # An old timestamp: a rewrite would replace it, even on a coarse clock.
        for name in outputs:
            os.utime(self.root / name, (1, 1))
        run()
        for name in outputs:
            self.assertEqual((self.root / name).stat().st_mtime_ns, 1_000_000_000, name)
            self.assertEqual((self.root / name).read_bytes(), first[name], name)
        self.assertFalse(list(self.root.rglob("*.part")))
        # A changed source still replaces the output.
        self.files["/gsm8k/train.jsonl"] = self.jsonl(self.rows[:-1])
        os.remove(self.root / "gsm8k/raw/train.jsonl")
        run()
        self.assertNotEqual((self.root / "gsm8k/train.jsonl").stat().st_mtime_ns, 1_000_000_000)
        self.assertNotEqual((self.root / "gsm8k/train.jsonl").read_bytes(), first["gsm8k/train.jsonl"])

    def test_preparation_produces_dev_and_refreshes_filtered_tokens(self):
        repo = Path(__file__).resolve().parents[1]
        data_root = self.root / "prepared"
        for name in ["gsm8k/train.jsonl", "gsm8k/dev.jsonl", "gsm8k/test.jsonl"]:
            path = data_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        mock_python = self.root / "skip-downloads"
        mock_python.write_text("#!/bin/sh\nexit 0\n")
        mock_python.chmod(0o755)
        mock_cxxgpt = self.root / "mock-cxxgpt"
        mock_cxxgpt.write_text(f"""#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["MOCK_ARGS"], "a") as log:
    log.write(json.dumps(args) + "\\n")
out = args[args.index("--out") + 1]
pathlib.Path(out + ".txt").write_text("questions")
pathlib.Path(out + ".tok").write_text("tokens")
""")
        mock_cxxgpt.chmod(0o755)
        log = self.root / "commands.jsonl"
        env = dict(os.environ, DATA=str(data_root), CX=str(mock_cxxgpt), PY=str(mock_python), MOCK_ARGS=str(log))

        def prepare(check=True):
            return subprocess.run(["bash", "scripts/prepare_data.sh"], cwd=repo, env=env,
                                  check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # No tokenizer yet: the script never makes one (prepare_clean.sh does),
        # so it stops before encoding anything.
        result = prepare(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prepare_clean.sh", result.stdout)
        self.assertFalse(log.exists())
        (data_root / "tok.bpe").write_text("fixture\n")
        prepare()
        self.assertTrue((data_root / "gsm8k/dev.txt").exists())
        self.assertTrue((data_root / "gsm8k/dev.tok").exists())
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([call[0] for call in calls], ["prep-qa"] * 3)
        self.assertEqual((data_root / "tok.bpe").read_text(), "fixture\n")
        prepare()
        self.assertEqual(calls, [json.loads(line) for line in log.read_text().splitlines()])
        # A token file older than its source is made again.
        os.utime(data_root / "gsm8k/train.tok", (1, 1))
        prepare()
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
