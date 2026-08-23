#!/usr/bin/env python3
"""Spot-check the tool-error detector: print a sample of flagged and
non-flagged tool outputs for eyeballing, plus how the new detector and the old
substring rule disagree.

The error signal carries weight 0.10 in the difficulty score, drives the
>=5-errors -> D3 override and feeds the tier inference — its precision is
worth 5 minutes of reading. Record the reviewed precision in SOLUTION.md.

Usage: python scripts/check_error_markers.py [export_linked/trajectories_v1_01.jsonl] [n_sample]
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluator.metrics import is_error_output  # noqa: E402

OLD_MARKERS = ("error", "traceback", "exit code 1", "failed", "exception")


def old_rule(out):
    return any(m in out[:400].lower() for m in OLD_MARKERS)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "export_linked/trajectories_v1_01.jsonl"
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    outputs = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            req = json.loads(line)
            for it in req["input"]:
                if it.get("type") in ("function_call_output", "custom_tool_call_output"):
                    outputs.append(json.dumps(it.get("output")))
    print(f"{len(outputs)} tool outputs scanned")
    flag_new = [o for o in outputs if is_error_output(o)]
    flag_old = [o for o in outputs if old_rule(o)]
    new_only = [o for o in outputs if is_error_output(o) and not old_rule(o)]
    old_only = [o for o in outputs if old_rule(o) and not is_error_output(o)]
    print(f"flagged: new rule {len(flag_new)}  old substring rule {len(flag_old)}  "
          f"(new-only {len(new_only)}, old-only {len(old_only)})")

    rng = random.Random(7)
    print(f"\n--- {n_sample} flagged samples (new rule) — read these: are they errors? ---")
    for o in rng.sample(flag_new, min(n_sample, len(flag_new))):
        print("  *", o[:200].replace("\n", " "))
    print(f"\n--- {min(10, len(old_only))} OLD-only flags the new rule dropped "
          f"(should be false positives like '0 failed') ---")
    for o in rng.sample(old_only, min(10, len(old_only))):
        print("  *", o[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
