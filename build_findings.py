#!/usr/bin/env python3
"""Bundle the analysis findings into dashboard/findings.js.

Collects results/model_tiers.json (inferred tier order), results/cache_trap.json
(the cache-reset demonstration) and results/matching_check.json (matched
cross-model validation) into one window.VIKTOR_FINDINGS object the dashboard
and the presentation render. Missing files are skipped gracefully.

Usage: python build_findings.py   (after infer_tiers / cache_trap / matching_check)
"""
import json
from pathlib import Path

OUT = Path("dashboard/findings.js")
PARTS = {
    "modelTiers": "results/model_tiers.json",
    "cacheTrap": "results/cache_trap.json",
    "matching": "results/matching_check.json",
    "finalValidation": "results/final_validation.json",
}


def main():
    bundle = {}
    for key, path in PARTS.items():
        p = Path(path)
        if p.exists():
            bundle[key] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"skipping {key}: {path} not found")
    OUT.write_text("window.VIKTOR_FINDINGS = " + json.dumps(bundle, separators=(",", ":"))
                   + ";\n", encoding="utf-8")
    print(f"wrote {OUT} with: {', '.join(bundle)}")


if __name__ == "__main__":
    main()
