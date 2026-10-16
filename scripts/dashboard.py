#!/usr/bin/env python3
"""dashboard.py: watch a training run in the browser while it trains.

    python3 scripts/dashboard.py              # the run written to last
    python3 scripts/dashboard.py rp1          # runs/rp1.*
    python3 scripts/dashboard.py --port 8800 --no-open

Opens http://127.0.0.1:8765/ (this computer only). The page reloads its data
every 10 s from the files night.sh already writes, so it can be started,
stopped and restarted at any time, also in the middle of a night:
  - runs/RUN.log: the settings, the steps, the GSM8K exams and the last
    wrong answer, skipped steps and warnings;
  - runs/RUN.csv: loss, validation loss, learning rate, gradient norm and
    time per step, every 50 steps;
  - runs/RUN.load.csv: each layer's share per expert;
  - runs/RUN.val.csv: each held-out file's loss (train --val-sources),
    GSM8K's final answers apart;
  - runs/RUN.gpu.csv: the GPU's clock, temperature, power and memory every
    10 s (the only file with clock times);
  - runs/RUN.ckpt.control: the control file, if one was written.
It only reads them. Each poll reads only what was added since the last one.
Needs only the standard library.
"""
import argparse
import http.server
import json
import math
import re
import statistics
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "dashboard.html"

STALL_SECONDS = 300      # no new log line for this long: the run looks stuck
HOT_CELSIUS = 87         # the RTX 4070 laptop slows itself down near here
THROTTLE_BITS = {        # nvidia-smi clocks_throttle_reasons.active
    0x4: "power cap",
    0x8: "hardware slowdown",
    0x20: "software thermal slowdown",
    0x40: "hardware thermal slowdown",
    0x80: "power brake",
}


class Tail:
    """The lines of a file that grows at its end, read from where the last
    read stopped. A file that got shorter (rewritten) is read again."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.rest = b""      # a last line without its newline yet
        self.lines = []

    def update(self):
        try:
            size = self.path.stat().st_size
        except OSError:
            self.offset, self.rest, self.lines = 0, b"", []
            return False
        if size < self.offset:
            self.offset, self.rest, self.lines = 0, b"", []
        if size == self.offset:
            return False
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = self.rest + f.read()
            self.offset = f.tell()
        parts = data.split(b"\n")
        self.rest = parts.pop()
        self.lines += [p.decode("utf-8", errors="replace").rstrip("\r") for p in parts]
        return True


def number(text):
    """A float, or None for text that is not a finite number."""
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse_csv(lines):
    """The metrics CSV: one row per log interval. After a resume from an
    older checkpoint the steps go back; the later rows replace the earlier
    ones from that step on."""
    columns, rows, resumes = [], [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"# resumed at step (\d+)", line)
            if m:
                resumes.append(int(m.group(1)))
            continue
        if line.startswith("step,"):
            columns = line.split(",")
            continue
        values = [number(v) for v in line.split(",")]
        if not columns or len(values) != len(columns) or values[0] is None:
            continue
        row = dict(zip(columns, values))
        row["step"] = int(row["step"])
        if "val_loss" in row and (row["val_loss"] is None or row["val_loss"] < 0):
            row["val_loss"] = None  # -1: not measured at this step
        while rows and rows[-1]["step"] >= row["step"]:
            rows.pop()
        rows.append(row)
    return {"columns": columns, "rows": rows, "resumes": resumes}


STEP_LINE = re.compile(r"^step\s+(\d+) \| loss ([\d.eE+-]+|nan|inf)(.*)$")


def parse_log(lines):
    """The training log (night.sh tees the trainer's output into it). A
    night that is restarted appends to it: "session" is the last start."""
    out = {"meta": {}, "session": None, "exams": [], "events": [], "speed": [],
           "last_wrong": None, "ended": None, "last_step": None}
    exam = None
    wrong = None
    for i, line in enumerate(lines):
        m = re.match(r"^(\S+): (\d+) steps of (\d+) tokens; stops after (\d+) minutes", line)
        if m:
            out["session"] = {"line": i, "total_steps": int(m.group(2)),
                              "tokens_per_step": int(m.group(3)), "minutes": int(m.group(4)),
                              "first_step": None, "skipped": 0}
            out["ended"] = None
            continue
        m = STEP_LINE.match(line)
        if m:
            step = int(m.group(1))
            out["last_step"] = step
            if out["session"] is not None and out["session"]["first_step"] is None:
                out["session"]["first_step"] = step
            t = re.search(r"(\d+) tok/s", m.group(3))
            ms = re.search(r"([\d.]+) ms", m.group(3))
            if t:
                while out["speed"] and out["speed"][-1][0] >= step:
                    out["speed"].pop()
                out["speed"].append([step, int(t.group(1)), number(ms.group(1)) if ms else None])
            if "(final validation)" in line:
                out["ended"] = "finished"
            wrong = None
            continue
        if wrong is not None:
            if line.startswith("    expected: "):
                wrong["expected"] = line[14:]
                wrong["part"] = "expected"
                continue
            if line.startswith("    got:      "):
                wrong["got"] = line[14:]
                wrong["part"] = "got"
                continue
            if line.startswith("              ") and wrong.get("part"):
                wrong[wrong["part"]] += "\n" + line[14:]
                continue
            wrong.pop("part", None)
            wrong = None
        m = re.match(r"^reasoning eval at step (\d+) \(([\d.]+) s\):", line)
        if m:
            exam = {"step": int(m.group(1)), "seconds": float(m.group(2))}
            continue
        m = re.match(r"^overall ([\d.]+)% \((\d+) / (\d+)\)(?: \((.*)\))?", line)
        if m and exam is not None:
            exam.update(percent=float(m.group(1)), correct=int(m.group(2)),
                        total=int(m.group(3)), effort=m.group(4) or "greedy")
            while out["exams"] and out["exams"][-1]["step"] >= exam["step"]:
                out["exams"].pop()
            out["exams"].append(exam)
            continue
        m = re.match(r"^--- wrong \((.*?)\): (.*)$", line)
        if m:
            wrong = {"step": exam["step"] if exam else None, "task": m.group(1),
                     "question": m.group(2), "expected": "", "got": ""}
            out["last_wrong"] = wrong
            continue
        for key in ("model", "params", "gpu", "data sources", "data", "gpu memory"):
            if line.startswith(key + ": "):
                out["meta"][key] = line[len(key) + 2:]
                if key == "model":
                    out["ended"] = None  # every start of the trainer prints it
                break
        else:
            event = log_event(line)
            if event:
                kind, text = event
                out["events"].append({"line": i, "kind": kind, "text": text,
                                      "step": out["last_step"]})
                if kind == "skip" and out["session"] is not None:
                    out["session"]["skipped"] += 1
                if kind in ("end", "stop") or text.endswith("(interrupted)"):
                    out["ended"] = text
                if kind == "crash":
                    out["ended"] = "crashed: " + text
    return out


def log_event(line):
    """The log lines worth showing as events: (kind, text), or None."""
    line = line.strip()
    rules = [
        (r"^resumed .* at step \d+", "info"),
        (r"^starting from the weights in ", "info"),
        (r"^saved .* at step \d+", "save"),
        (r"^step \d+: loss .*; update skipped", "skip"),
        (r"^stopping: ", "stop"),
        (r"^time limit reached", "end"),
        (r"^under 1%: ", "warn"),
        (r"^warning: ", "warn"),
        (r"^reasoning eval skipped", "warn"),
        (r"check failed", "crash"),
        (r"truncated tokens", "crash"),
    ]
    for pattern, kind in rules:
        if re.search(pattern, line):
            return kind, line
    return None


def parse_val_sources(lines):
    """runs/RUN.val.csv (train --val-sources): step,name,part,targets,loss.
    Per name and part, the losses by step; a step written again (a resume
    from an older checkpoint) replaces the earlier number."""
    series = {}
    for line in lines:
        f = line.strip().split(",")
        if len(f) != 5 or f[0] == "step":
            continue
        loss = number(f[4])
        try:
            step = int(f[0])
        except ValueError:
            continue
        if loss is not None:
            series.setdefault(f[1], {}).setdefault(f[2], {})[step] = loss
    return {name: {part: sorted(points.items()) for part, points in parts.items()}
            for name, parts in series.items()}


def parse_gpu(lines):
    """runs/RUN.gpu.csv, no header: time, clock, temperature, power, load,
    memory, slowdown reasons, and the last logged step ("start" before)."""
    samples = []
    for line in lines:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 8:
            continue
        try:
            t = datetime.strptime(f[0].split(".")[0], "%Y/%m/%d %H:%M:%S").timestamp()
            reasons = int(f[6], 16)
        except ValueError:
            continue  # a failed nvidia-smi query
        step = re.match(r"step\s+(\d+)", f[7])
        samples.append({"t": t, "clock": number(f[1].split()[0]), "temp": number(f[2]),
                        "power": number(f[3].split()[0]), "util": number(f[4].split()[0]),
                        "mem": number(f[5].split()[0]), "reasons": reasons,
                        "step": int(step.group(1)) if step else None})
    return samples


def parse_load(lines):
    """runs/RUN.load.csv: per step and layer, each expert's share."""
    by_step = {}
    for line in lines:
        f = line.strip().split(",")
        if len(f) < 4 or f[0] == "step" or line.startswith("#"):
            continue
        try:
            step, layer = int(f[0]), int(f[1])
            shares = [float(x) for x in f[2:-1]]
        except ValueError:
            continue
        by_step.setdefault(step, {})[layer] = shares
    steps = sorted(by_step)
    lowest = [[s, min(min(v) for v in by_step[s].values())] for s in steps]
    last = None
    if steps:
        layers = by_step[steps[-1]]
        last = {"step": steps[-1], "layers": [layers[k] for k in sorted(layers)]}
    return {"lowest": lowest, "last": last}


def thin(points, limit):
    """Every k-th point (and the last), so a long night stays a quick page."""
    if len(points) <= limit:
        return points
    k = -(-len(points) // limit)
    return points[::k] + ([points[-1]] if (len(points) - 1) % k else [])


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(log, metrics, gpu, load, control, log_age, now):
    """What the page shows: the state, the time left, the warnings."""
    rows = metrics["rows"]
    session = log["session"] or {}
    last = rows[-1] if rows else None
    last_step = log["last_step"] if log["last_step"] is not None else (last["step"] if last else None)

    if not log["ended"] and session and last_step is not None and last_step >= session["total_steps"]:
        log = dict(log, ended=f"reached its last step, {last_step}")
    if log["ended"]:
        state = "crashed" if log["ended"].startswith("crashed") else "ended"
    elif log_age is None:
        state = "no log"
    elif log_age > STALL_SECONDS:
        state = "stalled"
    else:
        state = "training"

    # Time left: the steps still to go at the recent speed, or the time
    # limit, whichever comes first. Only gpu.csv has clock times: the
    # session started about when its samples began showing its own steps
    # (each sample notes the last logged step), one log interval earlier.
    recent_ms = median([r.get("ms_per_step") for r in rows[-10:]])
    eta = None
    if state == "training" and session and last_step is not None and recent_ms:
        eta = max(0, session["total_steps"] - last_step) * recent_ms / 1000
        first, started = session.get("first_step"), None
        if first is not None:
            for g in reversed(gpu):
                if g["step"] is None or g["step"] < first:
                    break
                started = g["t"]
        if started is not None:
            started -= 50 * recent_ms / 1000
            eta = max(0, min(eta, started + session["minutes"] * 60 - now))

    alerts = []

    def alert(level, text):
        alerts.append({"level": level, "text": text})

    if state == "stalled":
        alert("bad", f"no new log line for {log_age / 60:.0f} min, and the run has not ended")
    if state == "crashed":
        alert("bad", log["ended"])
    if log["ended"] and log["ended"].startswith("stopping"):
        alert("bad", log["ended"])
    if session.get("skipped"):
        alert("warn", f"{session['skipped']} skipped steps (non-finite gradients) since the start")
    if len(rows) > 10:
        before = rows[-21:-1]
        base = median([r["loss"] for r in before])
        if base and last["loss"] is not None and last["loss"] > 1.2 * base:
            alert("warn", f"loss {last['loss']:.3f} jumped above its recent median {base:.3f}")
        g = median([r.get("grad_norm") for r in before])
        if g and last.get("grad_norm") and last["grad_norm"] > 3 * g:
            alert("warn", f"gradient norm {last['grad_norm']:.3f}, more than 3x its recent median {g:.3f}")
    speed = log["speed"]
    if len(speed) > 10:
        base = median([s[1] for s in speed[-41:-1]])
        if base and speed[-1][1] < 0.8 * base:
            alert("warn", f"{speed[-1][1]} tok/s, below 80% of the recent {base:.0f} "
                          "(PAROIKOS, another GPU job, or the GPU's heat)")
    if load["last"]:
        starving = [f"layer {li} expert {e} ({100 * s:.1f}%)"
                    for li, shares in enumerate(load["last"]["layers"])
                    for e, s in enumerate(shares) if s < 0.01]
        if starving:
            alert("warn", "experts under 1%: " + ", ".join(starving))
    if gpu and now - gpu[-1]["t"] < 120:
        g = gpu[-1]
        if g["temp"] is not None and g["temp"] >= HOT_CELSIUS:
            alert("warn", f"GPU at {g['temp']:.0f} °C")
        recent = [s for s in gpu if s["t"] > g["t"] - 300]
        slow = sorted({text for s in recent for bit, text in THROTTLE_BITS.items()
                       if s["reasons"] & bit and bit != 0x4})
        if slow:
            alert("warn", "GPU slowed down in the last 5 min: " + ", ".join(slow))
    if control:
        alert("info", f"control file: {control}")

    exams = log["exams"]
    best = max(exams, key=lambda e: (e["percent"], e["step"])) if exams else None
    vals = [r for r in rows if r.get("val_loss") is not None]
    best_val = min(vals, key=lambda r: r["val_loss"]) if vals else None
    return {
        "state": state, "log_age": log_age, "last_step": last_step,
        "total_steps": session.get("total_steps"), "tokens_per_step": session.get("tokens_per_step"),
        "minutes": session.get("minutes"), "eta_seconds": eta, "ms_per_step": recent_ms,
        "tok_per_s": speed[-1][1] if speed else None,
        "loss": last["loss"] if last else None,
        "val_loss": vals[-1]["val_loss"] if vals else None,
        "best_val": [best_val["step"], best_val["val_loss"]] if best_val else None,
        "exam": exams[-1] if exams else None, "best_exam": best,
        "gpu": gpu[-1] if gpu and now - gpu[-1]["t"] < 120 else None,
        "alerts": alerts,
    }


class Run:
    """One run's files, read incrementally."""

    def __init__(self, runs, name):
        self.runs, self.name = runs, name
        base = runs / name
        self.tails = {k: Tail(Path(str(base) + suffix)) for k, suffix in
                      [("log", ".log"), ("csv", ".csv"), ("gpu", ".gpu.csv"), ("load", ".load.csv"),
                       ("val", ".val.csv")]}
        self.control = Path(str(base) + ".ckpt.control")
        self.parsed = {}
        self.lock = threading.Lock()

    def read(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            for key, tail in self.tails.items():
                if tail.update() or key not in self.parsed:
                    self.parsed[key] = {"log": parse_log, "csv": parse_csv, "gpu": parse_gpu,
                                        "load": parse_load, "val": parse_val_sources}[key](tail.lines)
            log, metrics = self.parsed["log"], self.parsed["csv"]
            gpu, load = self.parsed["gpu"], self.parsed["load"]
            try:
                control = self.control.read_text(errors="replace").strip()
            except OSError:
                control = ""
            try:
                log_age = now - self.tails["log"].path.stat().st_mtime
            except OSError:
                log_age = None
            summary = summarize(log, metrics, gpu, load, control, log_age, now)
            session_gpu = gpu[-2000:]
            return {
                "name": self.name, "now": now, "summary": summary, "meta": log["meta"],
                "columns": metrics["columns"],
                "rows": thin(metrics["rows"], 1500), "resumes": metrics["resumes"],
                "exams": log["exams"], "speed": thin(log["speed"], 1500),
                "events": log["events"][-40:], "last_wrong": log["last_wrong"],
                "gpu": thin(session_gpu, 1000),
                "load_lowest": thin(load["lowest"], 1000), "load_last": load["last"],
                "val_sources": self.parsed["val"],
                "tail": self.tails["log"].lines[-40:],
            }


def list_runs(runs):
    """Runs with a training log: runs/NAME.log, also one folder down."""
    found = []
    for log in list(runs.glob("*.log")) + list(runs.glob("*/*.log")):
        name = log.relative_to(runs).with_suffix("").as_posix()
        if name.endswith(".labels"):
            continue
        csv = log.with_suffix(".csv")
        try:
            with open(log, "rb") as f:
                head = f.read(4096)
        except OSError:
            continue
        if csv.exists() or b" steps of " in head:
            found.append((log.stat().st_mtime, name))
    return [name for _, name in sorted(found, reverse=True)]


def make_handler(runs, cache):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # quiet: the page asks every 10 s

        def send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path == "/":
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/runs":
                self.send(200, json.dumps(list_runs(runs)).encode(), "application/json")
            elif path == "/api/run":
                from urllib.parse import unquote
                name = unquote(params.get("name", ""))
                if name not in list_runs(runs):  # only runs found above: no other files
                    self.send(404, b'{"error": "no such run"}', "application/json")
                    return
                run = cache.setdefault(name, Run(runs, name))
                body = json.dumps(run.read(), allow_nan=False, default=str)
                self.send(200, body.encode(), "application/json")
            else:
                self.send(404, b"not found", "text/plain")
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", nargs="?", help="the run to show first (default: the last one written)")
    ap.add_argument("--runs", default="runs", help="the folder with the runs (default: runs)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args()
    runs = Path(args.runs)
    if not runs.is_dir():
        sys.exit(f"no folder {runs}: run this from the repository's top folder")
    names = list_runs(runs)
    if args.run and args.run not in names:
        sys.exit(f"no run {args.run} in {runs} (found: {', '.join(names[:10]) or 'none'})")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runs, {}))
    url = f"http://127.0.0.1:{args.port}/" + (f"?run={args.run}" if args.run else "")
    print(f"dashboard: {url} (Ctrl-C stops it; the run keeps training)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
