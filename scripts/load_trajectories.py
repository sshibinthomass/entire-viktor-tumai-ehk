#!/usr/bin/env python3
"""Load a redacted-trajectories export and reconstruct trajectories.

The export is chunked JSONL — `trajectories_v1_<index>.jsonl.tar.gz` archives that
extract to `export/trajectories_v1_<index>.jsonl` (any `export/*.jsonl` is read). Each line is one LLM
request: `model`, `input` (Responses-format item list), `tools`. There are no
trajectory ids — requests of the same task are recovered by grouping on the
task's opening messages (system + first user text), then ordering by history
length (each request's input contains every item of the previous one).

Usage: python scripts/load_trajectories.py export/
Importable: iter_requests, group_trajectories, est_tokens, first_user_text.
"""
import json, sys, hashlib
from pathlib import Path
from collections import Counter, defaultdict

SYNTHETIC_MARKER = "do the synthetic thing for <PERSON_A> on <PROJECT_NAME>"

def iter_requests(export_dir):
    """Yield (chunk_name, line_no, request) for every line of every chunk."""
    chunks = sorted(Path(export_dir).glob("*.jsonl"))
    if not chunks:
        sys.exit(f"no *.jsonl chunks found in {export_dir}")
    for p in chunks:
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    yield p.name, i, json.loads(line)

def _message_text(item):
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"input_text", "text"}
    )


def first_user_text(req):
    """Text of the first user message — stable across all requests of a task."""
    for item in req["input"]:
        if item.get("role") == "user":
            return _message_text(item)
    return ""


def is_generated_synthetic(req):
    """Recognize only the exact marker emitted by make_synthetic_sample.py."""

    return SYNTHETIC_MARKER in first_user_text(req)

def group_key(req):
    """Hash the complete opening system + first user message.

    Long Viktor memory envelopes often share their first 2,000 characters.  A
    truncated first-user hash therefore merges unrelated tasks and can create
    false mixed-model trajectories.  Full opening messages follow the challenge
    reconstruction rule and remain safe because only the digest is returned.
    """

    system = ""
    for item in req["input"]:
        if item.get("role") == "system":
            system = _message_text(item)
            break
    opening = json.dumps([system, first_user_text(req)], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(opening.encode("utf-8")).hexdigest()[:24]

def group_trajectories(requests):
    """Group requests by task and order each group by input length (= call order)."""
    groups = defaultdict(list)
    for req in requests: groups[group_key(req)].append(req)
    for k in groups: groups[k].sort(key=lambda r: len(r["input"]))
    return dict(groups)

def est_tokens(obj):
    """Crude token estimate: serialized chars / 4. There is NO usage field in the
    export — every token number in this repo is an estimate. State that in your writeup."""
    return len(json.dumps(obj)) // 4 if not isinstance(obj, str) else len(obj) // 4

def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    reqs = [r for _, _, r in iter_requests(export)]
    generated = sum(is_generated_synthetic(r) for r in reqs)
    models = Counter(r["model"] for r in reqs)
    print(f"requests={len(reqs)}  per-model request counts: {dict(models)}")
    if generated:
        print(f"generated synthetic requests present={generated}  (exclude from real-data evaluation)")
    groups = group_trajectories(reqs)
    sizes = sorted(len(v) for v in groups.values())
    print(f"reconstructed trajectories={len(groups)}  calls/trajectory min/median/max: "
          f"{sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}")
    mixed = [k for k, v in groups.items() if len({r['model'] for r in v}) > 1]
    print(f"trajectories with >1 model: {len(mixed)}"
          + ("  (premise says one model per trajectory — inspect these)" if mixed else "  (matches the one-model-per-trajectory premise)"))
    total = sum(est_tokens(r["input"]) for r in reqs)
    print(f"est. input tokens (chars/4, no usage in export): {total:,}")
    # spot-check one request for the expected fields
    r = reqs[0]
    missing = [k for k in ("model", "input", "tools") if k not in r]
    extra = [k for k in r if k not in ("model", "input", "tools")]
    print(f"schema check on first request: missing={missing or 'none'} extra={extra or 'none'}")

if __name__ == "__main__": main()
