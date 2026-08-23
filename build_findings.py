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

from scripts import dashboard_datasets

OUT = Path("dashboard/findings.js")
DATA = Path("dashboard/data.js")
PARTS = {
    "modelTiers": "results/model_tiers.json",
    "cacheTrap": "results/cache_trap.json",
    "matching": "results/matching_check.json",
    "finalValidation": "results/final_validation.json",
    "comparison": "results/comparison.json",
    "sweeps": "results/sweeps.json",
    "tuning": "results/tuning_report.json",
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
    # the findings above describe whichever chunk run_pipeline.py last ran on —
    # keep a copy next to that chunk's data file so the lab's dataset dropdown
    # switches cards and numbers together (see scripts/dashboard_datasets.py)
    source = None
    if DATA.exists():
        head = DATA.read_text(encoding="utf-8")[:4000]
        start = head.find('"source":"')
        if start >= 0:
            source = head[start + 10:head.find('"', start + 10)]
    if source:
        p = dashboard_datasets.write(source, "findings", bundle, "VIKTOR_FINDINGS")
        dashboard_datasets.rebuild_manifest()
        print(f"wrote {p} (dataset '{dashboard_datasets.slug_of(source)}')")
    else:
        print(f"no meta.source in {DATA} — skipped the per-dataset findings copy")


if __name__ == "__main__":
    main()
