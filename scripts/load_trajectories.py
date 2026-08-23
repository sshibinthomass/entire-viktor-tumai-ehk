#!/usr/bin/env python3
"""Load a redacted-trajectories export and reconstruct trajectories.

The export is chunked JSONL — `trajectories_v1_<index>.jsonl.tar.gz` archives that
extract to `export/trajectories_v1_<index>.jsonl` (any `export/*.jsonl` is read). Each line is one LLM
request: `model`, `input` (Responses-format item list), `tools`. There are no
trajectory ids — requests of the same task are recovered by grouping on the
task's opening messages (system + first user text), then ordering by history
length (each request's input contains every item of the previous one).

Reconstruction is a CHECKED invariant, not an assumption:
  - the grouping key hashes the full system text plus the full first user text
    (either alone is not unique across tasks)
  - requests whose key is empty (no system, no user text — e.g. image-only
    openings, which all hash alike) become per-request singleton trajectories
    instead of merging into one false giant
  - after sorting each group by input length, a NESTING VALIDATOR asserts that
    call i's input items are an item-level prefix of call i+1's; groups that
    fail are split into maximal prefix-consistent chains, and the split count
    is reported

Usage: python scripts/load_trajectories.py export/
Importable: iter_requests, group_trajectories, reconstruct, est_tokens,
first_user_text, system_text, group_key.
"""
import json, sys, hashlib
from pathlib import Path
from collections import Counter, defaultdict


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


def _txt(part):
    if isinstance(part, str):
        return part
    return part.get("text", "") if isinstance(part, dict) else ""


def _content(item):
    """Join an item's content whether it is a plain string or a parts list."""
    c = item.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(_txt(p) for p in c)
    return ""


def system_text(req):
    """Full text of the system message (parts joined) — stable across a task's calls."""
    for item in req["input"]:
        if item.get("role") == "system":
            return _content(item)
    return ""


def first_user_text(req):
    """Text of the first user message — stable across all requests of a task."""
    for item in req["input"]:
        if item.get("role") == "user":
            return _content(item)
    return ""


def group_key(req):
    """Task fingerprint: system text + FULL first user text (no truncation).

    Returns None when both are empty (e.g. an image-only opening): such
    requests cannot be safely grouped and become singleton trajectories."""
    sys_t, usr_t = system_text(req), first_user_text(req)
    if not sys_t and not usr_t:
        return None
    return hashlib.sha1((sys_t + "\x00" + usr_t).encode()).hexdigest()[:16]


def _is_item_prefix(shorter, longer):
    """True if `shorter`'s input items equal the first items of `longer`'s."""
    a, b = shorter["input"], longer["input"]
    if len(a) > len(b):
        return False
    return a == b[:len(a)]


def reconstruct(reqs):
    """Reconstruct trajectories from a list of raw requests.

    Returns (trajectories, stats): trajectories is a list of index-lists into
    `reqs` (each ordered by call order); stats counts singletons and groups
    split by the nesting validator."""
    groups = defaultdict(list)
    singles = []
    for i, req in enumerate(reqs):
        k = group_key(req)
        (singles if k is None else groups[k]).append(i)

    trajectories, n_split = [], 0
    for k in groups:
        idx = sorted(groups[k], key=lambda i: len(reqs[i]["input"]))
        # nesting validator: chain-split any group whose members don't nest
        chains = []  # each chain: list of indices, last one has the longest input
        for i in idx:
            best = None
            for ch in chains:
                if _is_item_prefix(reqs[ch[-1]], reqs[i]):
                    if best is None or len(reqs[ch[-1]]["input"]) > len(reqs[best[-1]]["input"]):
                        best = ch
            if best is None:
                chains.append([i])
            else:
                best.append(i)
        if len(chains) > 1:
            n_split += 1
        trajectories.extend(chains)
    trajectories.extend([i] for i in singles)
    # deterministic order: by first request's original position
    trajectories.sort(key=lambda ch: ch[0])
    stats = {"n_requests": len(reqs), "n_trajectories": len(trajectories),
             "n_singleton_empty_key": len(singles), "n_groups_split_by_validator": n_split}
    return trajectories, stats


def group_trajectories(requests):
    """Group requests by task -> {trajectory_key: [requests in call order]}.

    Keys are the group hash, suffixed `.1`, `.2`, ... for chains the nesting
    validator split off, and `solo:<n>` for empty-key singletons."""
    reqs = list(requests)
    chains, _ = reconstruct(reqs)
    out, seen = {}, Counter()
    for ch in chains:
        k = group_key(reqs[ch[0]])
        if k is None:
            key = f"solo:{seen['solo']}"
            seen["solo"] += 1
        else:
            seen[k] += 1
            key = k if seen[k] == 1 else f"{k}.{seen[k] - 1}"
        out[key] = [reqs[i] for i in ch]
    return out


def est_tokens(obj):
    """Crude token estimate: serialized chars / 4. There is NO usage field in the
    export — every token number in this repo is an estimate. State that in your writeup."""
    return len(json.dumps(obj)) // 4 if not isinstance(obj, str) else len(obj) // 4


def serialize_items(req):
    """One serialization per input item (reused for both token counting and
    prefix comparison — avoids re-serializing the whole growing input per call)."""
    return [json.dumps(it) for it in req["input"]]


def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    reqs = [r for _, _, r in iter_requests(export)]
    models = Counter(r["model"] for r in reqs)
    print(f"requests={len(reqs)}  per-model request counts: {dict(models)}")
    chains, stats = reconstruct(reqs)
    sizes = sorted(len(c) for c in chains)
    print(f"reconstructed trajectories={len(chains)}  calls/trajectory min/median/max: "
          f"{sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]}")
    print(f"nesting validator: {stats['n_groups_split_by_validator']} groups split, "
          f"{stats['n_singleton_empty_key']} empty-key singletons")
    mixed = sum(1 for c in chains if len({reqs[i]["model"] for i in c}) > 1)
    print(f"trajectories with >1 model: {mixed}"
          + ("  (premise says one model per trajectory — inspect these)" if mixed
             else "  (matches the one-model-per-trajectory premise)"))
    total = sum(est_tokens(r["input"]) for r in reqs)
    print(f"est. input tokens (chars/4, no usage in export): {total:,}")
    # spot-check one request for the expected fields
    r = reqs[0]
    core = ("model", "input", "tools")
    missing = [k for k in core if k not in r]
    extra = [k for k in r if k not in core + ("request_id", "trajectory_id", "call_index")]
    print(f"schema check on first request: missing={missing or 'none'} extra={extra or 'none'}")


if __name__ == "__main__":
    main()
