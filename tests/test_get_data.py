#!/usr/bin/env python3
"""Offline dataset tests: a local HTTP fake, never public dataset downloads.

The fake Parquet adapter reads JSON fixture rows; the download/listing and
filtering paths are real. Checking actual Parquet columns remains CI's
existing data-sources job, which has pyarrow and network access.
"""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import os
import subprocess
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
        self.hf = patch.object(data, "HF", self.url)
        self.hf.start()
        self.args = types.SimpleNamespace(out=str(self.root), check=False, split="train_1M", sources="")
        # Program-created fixture records, never real training data.
        self.rows = [{"question": f"Fixture {i}: compute {i} plus 1?", "answer": f"Add one.\n#### {i + 1}"}
                     for i in range(620)]
        self.files["/gsm8k/train.jsonl"] = self.jsonl(self.rows)
        self.files["/gsm8k/test.jsonl"] = self.jsonl([{"question": "Test fixture", "answer": "#### 1"}])

    def tearDown(self):
        self.hf.stop()
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
        # Lock the fixed seed and ordered source split across future changes.
        self.assertEqual(hashlib.sha256(first).hexdigest(), "4cf88629674a4bef92cec5f4525d92f0791171cba92edac4c1d27284290ae988")
        self.assertEqual((self.root / "gsm8k/test.jsonl").read_bytes(), self.files["/gsm8k/test.jsonl"])

    def test_filters_exactly_planted_rows(self):
        self.split()
        dev = self.read("gsm8k/dev.jsonl")[0]["question"]
        train = self.read("gsm8k/train.jsonl")[0]["question"]
        spaced = " \t" + dev.replace(" ", "\n  ") + "  "
        meta = [{"query": f"Reworded fixture {i}", "response": "A step. The answer is: 2", "original_question": q}
                for i, q in enumerate([dev, spaced, train, "Unrelated fixture"])]
        self.files["/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json"] = json.dumps(meta).encode()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            data.get_metamath(str(self.root / "metamath"), self.args)
        self.assertIn("MetaMathQA dev filter: removed 2 rows", output.getvalue())
        self.assertEqual([r["question"] for r in self.read("metamath/train.jsonl")],
                         ["Reworded fixture 2", "Reworded fixture 3"])
        rows = [{"problem": q, "generated_solution": "A step.", "expected_answer": "2", "problem_source": source}
                for q, source in [(dev, "gsm8k"), (spaced, "gsm8k"), (train, "gsm8k"),
                                  (dev, "augmented_gsm8k"), (dev + " Changed", "gsm8k"), (dev, "math")]]
        self.files["/api/datasets/nvidia/OpenMathInstruct-2/tree/main/data"] = json.dumps([
            {"type": "file", "path": "data/train_1M-00000.parquet"}]).encode()
        self.files["/datasets/nvidia/OpenMathInstruct-2/resolve/main/data/train_1M-00000.parquet"] = json.dumps(rows).encode()

        def fake_parquet_rows(path, columns):
            self.assertIn("problem_source", columns)
            for row in json.loads(Path(path).read_text()):
                yield {column: row[column] for column in columns}

        output = io.StringIO()
        with patch.object(data, "parquet_rows", fake_parquet_rows), contextlib.redirect_stdout(output):
            data.get_openmath2(str(self.root / "openmath2"), self.args)
        self.assertIn("OpenMath2 dev filter: removed 2 rows", output.getvalue())
        self.assertEqual([r["question"] for r in self.read("openmath2/train_1M.jsonl")],
                         [train, dev, dev + " Changed", dev])
        # Repeated conversion also applies the filter to cached raw downloads.
        with contextlib.redirect_stdout(io.StringIO()):
            data.get_metamath(str(self.root / "metamath"), self.args)
        self.assertEqual(len(self.read("metamath/train.jsonl")), 2)


    def test_identical_rerun_keeps_timestamps(self):
        # Run the real script, as prepare_data.sh does, twice. The second run
        # writes the same bytes, so it must leave the files (and their
        # timestamps) alone; otherwise prepare_data.sh re-encodes ~1.4M
        # MetaMath/OpenMath2 problems on every rerun.
        meta = [{"query": f"Reworded fixture {i}", "response": "A step. The answer is: 2",
                 "original_question": f"Unrelated fixture {i}"} for i in range(50)]
        meta_url = "/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json"
        self.files[meta_url] = json.dumps(meta).encode()
        script = Path(__file__).resolve().parents[1] / "scripts/get_data.py"
        env = dict(os.environ, HF_ENDPOINT=self.url, GSM8K_ENDPOINT=self.url + "/gsm8k")

        def run():
            subprocess.run(["python3", str(script), "metamath", "--out", str(self.root)], env=env,
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        outputs = ["metamath/train.jsonl", "gsm8k/train.jsonl", "gsm8k/dev.jsonl"]
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
        self.files[meta_url] = json.dumps(meta[:10]).encode()
        os.remove(self.root / "metamath/MetaMathQA-395K.json")
        run()
        self.assertNotEqual((self.root / "metamath/train.jsonl").stat().st_mtime_ns, 1_000_000_000)
        self.assertNotEqual((self.root / "metamath/train.jsonl").read_bytes(), first["metamath/train.jsonl"])

    def test_preparation_produces_dev_and_refreshes_filtered_tokens(self):
        repo = Path(__file__).resolve().parents[1]
        data_root = self.root / "prepared"
        for name in ["fineweb-edu/000.txt", "tinystories/train.txt", "gsm8k/train.jsonl",
                     "gsm8k/dev.jsonl", "gsm8k/test.jsonl", "metamath/train.jsonl",
                     "openmath2/train_1M.jsonl", "tok.bpe"]:
            path = data_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        mock_python = self.root / "skip-downloads"
        mock_python.write_text("#!/bin/sh\nexit 0\n")
        mock_python.chmod(0o755)
        mock_cxxgpt = self.root / "mock-cxxgpt"
        mock_cxxgpt.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["MOCK_ARGS"], "a") as log:
    log.write(json.dumps(args) + "\\n")
out = args[args.index("--out") + 1]
if args[0] == "prep-qa":
    pathlib.Path(out + ".txt").write_text("questions")
    pathlib.Path(out + ".tok").write_text("tokens")
else:
    pathlib.Path(out).write_text("tokens")
""")
        mock_cxxgpt.chmod(0o755)
        log = self.root / "commands.jsonl"
        env = dict(os.environ, DATA=str(data_root), CX=str(mock_cxxgpt), PY=str(mock_python), MOCK_ARGS=str(log))
        def prepare():
            subprocess.run(["bash", "scripts/prepare_data.sh"], cwd=repo, env=env,
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        prepare()
        self.assertTrue((data_root / "gsm8k/dev.txt").exists())
        self.assertTrue((data_root / "gsm8k/dev.tok").exists())
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(sum(call[0] == "prep-qa" for call in calls), 5)
        prepare()
        self.assertEqual(calls, [json.loads(line) for line in log.read_text().splitlines()])
        # Old token files from an unfiltered preparation must be regenerated.
        for name in ["gsm8k/train.tok", "metamath/train.tok", "openmath2/train.tok"]:
            os.utime(data_root / name, (1, 1))
        prepare()
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(sum(call[0] == "prep-qa" for call in calls), 8)

    def test_standalone_filter_creates_dev_and_missing_provenance_fails(self):
        self.files["/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json"] = json.dumps([
            {"query": "Fixture", "response": "The answer is: 2"}]).encode()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "original_question"):
            data.get_metamath(str(self.root / "metamath"), self.args)
        self.assertEqual(len(self.read("gsm8k/dev.jsonl")), 500)
        self.assertFalse((self.root / "metamath/train.jsonl").exists())
        # A present-but-null original is also unusable provenance.
        (self.root / "metamath/MetaMathQA-395K.json").write_text(json.dumps([
            {"query": "Fixture", "response": "The answer is: 2", "original_question": None}]))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "must be text"):
            data.get_metamath(str(self.root / "metamath"), self.args)


if __name__ == "__main__":
    unittest.main()
