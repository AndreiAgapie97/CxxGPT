#!/usr/bin/env python3
"""scripts/dashboard.py on made-up run files: what a night writes, read
while it grows, and the page's server. No GPU and no real run needed."""
import http.client
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("dashboard", Path(__file__).resolve().parents[1] / "scripts/dashboard.py")
dash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dash)

HEADER = """rp1: 1000 steps of 65536 tokens; stops after 450 minutes at the latest
resumed runs/rp1.ckpt at step 46946
model: vocab 16384, seq 512, dim 768, layers 12, heads 12, MoE 8 experts (top-2, hidden 1024, 0 shared)
params: 267.480 M total, 97.610 M active per token
gpu: NVIDIA GeForce RTX 4070 Laptop GPU (8.5 GB, compute 8.9), bf16 matmuls, fused tensor-core attention
data sources: data/targeted-math/word-train.tok*4,data/licensed-10b/blocks/000.tok
data: 1000 train tokens, 100 val tokens; 8 x 512 x 16 = 65536 tokens per step
gpu memory: 7004 MB after the first step
"""

EXAM = """reasoning eval on the GPU, 28 answers at a time
reasoning eval at step {step} (16.4 s):
task     d=1   all
gsm8k    {p}%   {p}%
overall {p}% ({c} / 250) (greedy)
gold completions exceeding --max-new 200: 6 / 250
--- wrong (gsm8k d=1): Janet’s ducks lay 16 eggs per day.
    expected: Janet sells 16 - 3 - 4 = 9 duck eggs a day.
              #### 18
    got:      The ducks lay 16 eggs.
              #### {c}
"""


def step_line(step, loss=2.0, val=None, toks=26000, ms=2500.0):
    v = f" | val {val:.4f}" if val is not None else ""
    return (f"step {step:6d} | loss {loss:.4f}{v} | lr 1.20e-04 | gnorm  0.300 | "
            f"{ms:6.1f} ms | {toks:8d} tok/s | load 1.50\n")


class ParseTest(unittest.TestCase):
    def test_log(self):
        text = HEADER + step_line(47000) + step_line(47500, val=1.1) + EXAM.format(step=47500, p="7.6", c=19)
        text += "step 47501: loss nan, gradient norm nan; update skipped\n"
        text += "under 1%: layer 4 expert 2 (0.8%)\n" + step_line(47550)
        log = dash.parse_log(text.splitlines())
        self.assertEqual(log["session"]["total_steps"], 1000)
        self.assertEqual(log["session"]["first_step"], 47000)
        self.assertEqual(log["session"]["skipped"], 1)
        self.assertEqual(log["last_step"], 47550)
        self.assertEqual(log["exams"], [{"step": 47500, "seconds": 16.4, "percent": 7.6, "correct": 19,
                                         "total": 250, "effort": "greedy"}])
        w = log["last_wrong"]
        self.assertEqual(w["question"], "Janet’s ducks lay 16 eggs per day.")
        self.assertEqual(w["expected"], "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\n#### 18")
        self.assertEqual(w["got"], "The ducks lay 16 eggs.\n#### 19")
        self.assertEqual(log["meta"]["gpu memory"], "7004 MB after the first step")
        self.assertEqual([e["kind"] for e in log["events"]], ["info", "skip", "warn"])
        self.assertEqual(log["speed"][-1], [47550, 26000, 2500.0])
        self.assertIsNone(log["ended"])

    def test_log_ends_and_restarts(self):
        lines = (HEADER + step_line(47000) + "time limit reached\n").splitlines()
        self.assertEqual(dash.parse_log(lines)["ended"], "time limit reached")
        # A restart (the trainer prints its model line again) is running again.
        lines += (HEADER + step_line(47050)).splitlines()
        self.assertIsNone(dash.parse_log(lines)["ended"])
        # Windows line ends and an older exam line without the effort.
        old = "reasoning eval at step 500 (18.4 s):\r\noverall 8.0% (20 / 250)\r\n"
        self.assertEqual(dash.parse_log(old.splitlines())["exams"][0]["effort"], "greedy")
        crash = "src/train.h:247: check failed: n == 0\n  truncated tokens in data/x.tok\n"
        self.assertTrue(dash.parse_log((HEADER + crash).splitlines())["ended"].startswith("crashed"))

    def test_csv_resume_and_unmeasured_val(self):
        text = ("step,loss,val_loss,lr,grad_norm,ms_per_step,load_ratio\n"
                "50,2.0,-1.00000,1e-4,0.3,2500,1.5\n100,1.9,1.2,1e-4,0.3,2500,1.5\n"
                "150,1.8,-1.00000,1e-4,0.3,2500,1.5\n"
                "# resumed at step 100\n120,1.85,-1.00000,1e-4,0.3,2500,1.5\nnan,garbage\n")
        m = dash.parse_csv(text.splitlines())
        self.assertEqual([r["step"] for r in m["rows"]], [50, 100, 120])
        self.assertEqual([r["val_loss"] for r in m["rows"]], [None, 1.2, None])
        self.assertEqual(m["resumes"], [100])

    def test_gpu_and_load(self):
        gpu = dash.parse_gpu([
            "2026/10/15 11:01:50.120, 1605 MHz, 56, 59.01 W, 0 %, 8 MiB, 0x0000000000000001, start",
            "2026/10/15 11:02:00.357, 2475 MHz, 75, 104.25 W, 98 %, 7142 MiB, 0x0000000000000040, step    250 \r",
            "Unable to determine the device handle for GPU0000:01:00.0: Unknown Error",
        ])
        self.assertEqual(len(gpu), 2)
        self.assertEqual((gpu[0]["step"], gpu[1]["step"], gpu[1]["temp"], gpu[1]["reasons"]), (None, 250, 75, 0x40))
        load = dash.parse_load(["step,layer,expert_0,expert_1,max_load_ratio",
                                "50,0,0.5,0.5,1.0", "50,1,0.995,0.005,1.99", "100,0,0.6,0.4,1.2", "100,1,0.7,0.3,1.4"])
        self.assertEqual(load["lowest"], [[50, 0.005], [100, 0.3]])
        self.assertEqual(load["last"], {"step": 100, "layers": [[0.6, 0.4], [0.7, 0.3]]})


class LiveRunTest(unittest.TestCase):
    """A run read while it grows, the way the page polls it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runs = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text, mode="a"):
        with open(self.runs / name, mode, encoding="utf-8", newline="") as f:
            f.write(text)

    def test_grows_alerts_and_ends(self):
        now = time.time()
        stamp = lambda t: time.strftime("%Y/%m/%d %H:%M:%S.000", time.localtime(t))
        self.write("rp1.log", HEADER.replace("1000 steps", "48946 steps"))
        self.write("rp1.csv", "step,loss,val_loss,lr,grad_norm,ms_per_step,load_ratio\n")
        run = dash.Run(self.runs, "rp1")
        self.assertEqual(run.read(now)["summary"]["state"], "training")
        self.assertEqual(dash.list_runs(self.runs), ["rp1"])

        # 20 steady intervals, then a loss and speed jump; half a line is
        # still being written.
        gpu = f"{stamp(now - 3000)}, 2475 MHz, 75, 104 W, 99 %, 7142 MiB, 0x0, start\n"
        for i in range(1, 21):
            step = 46950 + 50 * i
            self.write("rp1.log", step_line(step))
            self.write("rp1.csv", f"{step},2.0,-1.00000,1.2e-4,0.30,2500,1.5\n")
            gpu += f"{stamp(now - 3000 + 125 * i)}, 2475 MHz, 75, 104 W, 99 %, 7142 MiB, 0x0, step {step:6d}\n"
        self.write("rp1.gpu.csv", gpu + f"{stamp(now)}, 2100 MHz, 88, 80 W, 99 %, 7142 MiB, 0x40, step  47950\n")
        self.write("rp1.log", step_line(48000, loss=3.0, toks=15000))
        self.write("rp1.csv", "48000,3.0,-1.00000,1.2e-4,1.5,2500,1.5\n48050,2.0")
        self.write("rp1.ckpt.control", "lr-scale 0.5\n")
        r = run.read(now)
        s = r["summary"]
        self.assertEqual((s["state"], s["last_step"], s["total_steps"]), ("training", 48000, 48946))
        self.assertEqual(r["rows"][-1]["step"], 48000)  # not the half-written line
        texts = " | ".join(a["text"] for a in s["alerts"])
        for want in ("loss 3.000 jumped", "gradient norm 1.500", "15000 tok/s", "GPU at 88",
                     "hardware thermal slowdown", "control file: lr-scale 0.5"):
            self.assertIn(want, texts)
        # 946 steps left at 2.5 s, well inside the time limit.
        self.assertAlmostEqual(s["eta_seconds"], 946 * 2.5, delta=1)
        json.dumps(r, allow_nan=False)

        # The rest of the half line arrives; then the run ends.
        self.write("rp1.csv", ",-1.00000,1.2e-4,0.3,2500,1.5\n")
        self.write("rp1.log", step_line(48050) + "time limit reached\nsaved runs/rp1.ckpt at step 48050\n")
        r = run.read(now)
        self.assertEqual(r["rows"][-1]["step"], 48050)
        self.assertEqual(r["summary"]["state"], "ended")

    def test_time_left_and_stall(self):
        now = time.time()
        stamp = lambda t: time.strftime("%Y/%m/%d %H:%M:%S.000", time.localtime(t))
        header = HEADER.replace("1000 steps", "48946 steps")  # STEP0 46946 + 2000
        self.write("rp1.log", header + step_line(47000))
        self.write("rp1.csv", "step,loss,val_loss,lr,grad_norm,ms_per_step,load_ratio\n47000,2.0,-1,1e-4,0.3,2000,1.5\n")
        # The first sample showing the session's step came 10 minutes ago.
        self.write("rp1.gpu.csv", f"{stamp(now - 700)}, 1 MHz, 50, 10 W, 0 %, 8 MiB, 0x1, step  46900\n"
                                  f"{stamp(now - 600)}, 1 MHz, 50, 10 W, 0 %, 8 MiB, 0x0, step  47000\n")
        s = dash.Run(self.runs, "rp1").read(now)["summary"]
        self.assertAlmostEqual(s["eta_seconds"], 1946 * 2.0, delta=1)  # steps left, before the limit
        os.utime(self.runs / "rp1.log", (now - 600, now - 600))
        s = dash.Run(self.runs, "rp1").read(now)["summary"]
        self.assertEqual(s["state"], "stalled")
        self.assertIn("no new log line for 10 min", s["alerts"][0]["text"])

    def test_server(self):
        self.write("rp1.log", HEADER + step_line(47000))
        (self.runs / "secret.txt").write_text("not served")
        server = dash.http.server.ThreadingHTTPServer(("127.0.0.1", 0), dash.make_handler(self.runs, {}))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            def get(path):
                c = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
                c.request("GET", path)
                res = c.getresponse()
                return res.status, res.read()
            status, body = get("/")
            self.assertEqual(status, 200)
            self.assertIn(b"CxxGPT training", body)
            self.assertEqual(json.loads(get("/api/runs")[1]), ["rp1"])
            status, body = get("/api/run?name=rp1")
            self.assertEqual((status, json.loads(body)["summary"]["last_step"]), (200, 47000))
            for name in ("secret", "..%2Fsecret", "rp1.log"):
                self.assertEqual(get("/api/run?name=" + name)[0], 404)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
