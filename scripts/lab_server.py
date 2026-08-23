#!/usr/bin/env python3
"""Local lab server: serves dashboard/ AND accepts a new chunk from the browser.

`python -m http.server -d dashboard` is enough to READ the lab, but a new export
then has to be processed by hand on the command line. This server adds the one
thing a static server cannot do: take a chunk the user drops on the page, run the
pipeline on it, and make it appear in the dataset dropdown.

    python scripts/lab_server.py            # http://127.0.0.1:8017
    python scripts/lab_server.py --port 8018 --open

Endpoints (everything else is the static dashboard/ directory):

    GET  /api/health          -> {ok, datasets: [...]}   (the page uses this to
                                 decide whether to show the upload control)
    POST /api/upload          -> {job}   body = raw file bytes,
                                 headers: X-Filename: <name.jsonl|.jsonl.gz|.tar.gz>
    GET  /api/job?id=<job>    -> {state: queued|running|done|error, log, dataset}

What an upload runs (Mode B — a full refit on the uploaded chunk):

    scripts/enrich_dataset.py -> run_pipeline.py -> infer_tiers.py
    -> matching_check.py -> cache_trap.py -> build_findings.py

which is exactly what the README documents; the per-dataset files and the
manifest fall out of run_pipeline/build_findings (scripts/dashboard_datasets.py).

Notes, because this touches the licensed dataset:
  - binds to 127.0.0.1 only; this is a local tool, not a service
  - uploads land in `uploads/` (gitignored), never in the repo tree proper
  - the per-dataset previews file it generates is gitignored like previews.js
  - one job at a time: the pipeline steps write shared files under results/
"""
import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tarfile
import threading
import uuid
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import dashboard_datasets  # noqa: E402

UPLOADS = ROOT / "uploads"
MAX_BYTES = 600 * 1024 * 1024
JOBS = {}
JOB_LOCK = threading.Lock()   # the pipeline writes shared results/ files


def _safe_name(name):
    return Path(str(name or "upload.jsonl")).name.replace("\\", "_").replace("/", "_")


def _extract(raw_path, dest_dir):
    """-> the .jsonl to process. Accepts .jsonl, .jsonl.gz and .tar.gz archives."""
    name = raw_path.name
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(raw_path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jsonl")]
            if not members:
                raise ValueError("archive contains no .jsonl chunk")
            member = members[0]
            out = dest_dir / Path(member.name).name
            with tf.extractfile(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return out
    if name.endswith(".gz"):
        out = dest_dir / name[:-3]
        with gzip.open(raw_path, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return out
    if not name.endswith(".jsonl"):
        raise ValueError("expected a .jsonl chunk or a .jsonl.gz / .tar.gz archive")
    return raw_path


def _validate(jsonl_path):
    """Check the first line looks like a challenge-format request. -> n_lines."""
    with open(jsonl_path, encoding="utf-8") as f:
        first = f.readline()
        if not first.strip():
            raise ValueError("file is empty")
        try:
            req = json.loads(first)
        except json.JSONDecodeError as e:
            raise ValueError(f"first line is not JSON: {e}")
        missing = [k for k in ("model", "input", "tools") if k not in req]
        if missing:
            raise ValueError(f"first line is missing {missing} — expected the challenge "
                             f"format (one request per line: model, input, tools)")
        n = 1 + sum(1 for line in f if line.strip())
    return n


def _run(job, argv, label):
    job["log"].append(f"$ {label}")
    proc = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-4:]
    job["log"].extend(tail)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()[-6:]
        job["log"].extend(err)
        raise RuntimeError(f"{label} failed (exit {proc.returncode})")


def _process(job_id, raw_path):
    job = JOBS[job_id]
    try:
        with JOB_LOCK:
            job["state"] = "running"
            work = raw_path.parent
            chunk = _extract(raw_path, work)
            n = _validate(chunk)
            job["log"].append(f"{chunk.name}: {n} requests, schema ok")
            linked = work / "linked"
            staged = work / "staged"
            staged.mkdir(exist_ok=True)
            if chunk.parent != staged:                  # enrich reads a directory
                chunk = chunk.replace(staged / chunk.name)
            _run(job, ["scripts/enrich_dataset.py", str(staged), str(linked)],
                 "scripts/enrich_dataset.py")
            enriched = linked / chunk.name
            _run(job, ["run_pipeline.py", str(enriched)], "run_pipeline.py")
            # the dataset is already viewable at this point — the finding cards
            # are extra, and a thin or unusual chunk can legitimately defeat one
            # of them (too few trajectories to rank models, say). Report which
            # step failed and keep the dataset rather than throwing it away.
            failed = []
            steps = [["infer_tiers.py"], ["matching_check.py"],
                     ["cache_trap.py", str(enriched)],   # measured layer needs
                     ["build_findings.py"]]              # THIS chunk's calls
            for argv in steps:
                step = argv[0]
                try:
                    _run(job, argv, step)
                except RuntimeError as e:
                    failed.append(step)
                    job["log"].append(str(e))
            job["dataset"] = dashboard_datasets.slug_of(chunk.name)
            job["state"] = "done"
            job["failed_steps"] = failed
            job["log"].append(f"dataset ready: {job['dataset']}"
                              + (f" (finding cards incomplete: {', '.join(failed)})"
                                 if failed else ""))
    except Exception as e:                              # surfaced in the browser
        job["state"] = "error"
        job["error"] = str(e)
        job["log"].append(f"ERROR: {e}")


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # the lab rewrites data/*.js on every run — never let a stale copy stick
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json({"ok": True, "upload": True,
                               "datasets": dashboard_datasets.rebuild_manifest()})
        if path == "/api/job":
            job_id = parse_qs(urlparse(self.path).query).get("id", [""])[0]
            job = JOBS.get(job_id)
            if not job:
                return self._json({"error": "unknown job"}, 404)
            return self._json({k: job[k] for k in
                               ("state", "log", "dataset", "error", "failed_steps")})
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/upload":
            return self._json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json({"error": "empty upload"}, 400)
        if length > MAX_BYTES:
            return self._json({"error": f"file is larger than "
                                        f"{MAX_BYTES // (1024 * 1024)} MB"}, 413)
        name = _safe_name(self.headers.get("X-Filename"))
        job_id = uuid.uuid4().hex[:12]
        work = UPLOADS / job_id
        work.mkdir(parents=True, exist_ok=True)
        raw = work / name
        remaining = length
        with open(raw, "wb") as f:                      # stream: chunks are ~100 MB
            while remaining > 0:
                block = self.rfile.read(min(1 << 20, remaining))
                if not block:
                    break
                f.write(block)
                remaining -= len(block)
        JOBS[job_id] = {"state": "queued", "log": [f"received {name} "
                                                   f"({length / 1e6:.1f} MB)"],
                        "dataset": None, "error": None, "failed_steps": []}
        threading.Thread(target=_process, args=(job_id, raw), daemon=True).start()
        return self._json({"job": job_id})

    def log_message(self, fmt, *args):                  # quieter console
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--open", action="store_true", help="open the lab in a browser")
    a = ap.parse_args()
    dashboard_datasets.rebuild_manifest()
    handler = partial(Handler, directory=str(ROOT / "dashboard"))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"lab:  {url}\ndeck: {url}present.html\nupload: enabled (POST /api/upload)")
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
