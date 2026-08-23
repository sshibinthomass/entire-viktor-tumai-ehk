#!/usr/bin/env python3
"""Enrich raw trajectory JSONL exports with explicit Primary Keys and Foreign Keys.

Reconstruction (grouping key + nesting validator) comes from
load_trajectories.reconstruct — this script only assigns ids on top of it:

Child Record (export_linked/*.jsonl), one line per request:
- `request_id`:     1..R sequential PRIMARY KEY
- `trajectory_id`:  1..N sequential FOREIGN KEY (same ids results/routes.jsonl uses)
- `call_index`:     1-based call number within the trajectory
- `trajectory_key`: the reconstruction hash (ties ids back to the grouping)
- `model`, `input`, `tools`: passed through untouched

The raw export is NEVER modified. Output files are written to a temp file and
renamed into place so a crash cannot truncate anything.

Usage:
    python scripts/enrich_dataset.py export/ export_linked/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from load_trajectories import iter_requests, reconstruct, group_trajectories  # noqa: E402


def enrich_export(input_dir: str, output_dir: str):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(in_path.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"No *.jsonl files found in {input_dir}")

    print(f"Reading requests from {input_dir}...")
    requests_with_meta = list(iter_requests(input_dir))
    reqs = [req for _, _, req in requests_with_meta]

    chains, stats = reconstruct(reqs)
    keys = list(group_trajectories(reqs).keys())  # same order as chains
    print(f"reconstructed {stats['n_trajectories']} trajectories from "
          f"{stats['n_requests']} requests "
          f"(validator split {stats['n_groups_split_by_validator']} groups; "
          f"{stats['n_singleton_empty_key']} empty-key singletons)")

    # (position in reqs) -> (request_id, trajectory_id, call_index, trajectory_key)
    req_map = {}
    request_id = 1
    for traj_id, (chain, key) in enumerate(zip(chains, keys), start=1):
        for call_idx, pos in enumerate(chain, start=1):
            req_map[pos] = (request_id, traj_id, call_idx, key)
            request_id += 1

    pos_of = {}  # (chunk_name, line_no) -> position in reqs
    for pos, (chunk_name, line_no, _) in enumerate(requests_with_meta):
        pos_of[(chunk_name, line_no)] = pos

    total_processed = 0
    for chunk_file in jsonl_files:
        out_file = out_path / chunk_file.name
        tmp_file = out_path / (chunk_file.name + ".tmp")
        print(f"Writing enriched chunk: {out_file.name}...")

        with open(chunk_file, "r", encoding="utf-8") as fin, \
             open(tmp_file, "w", encoding="utf-8") as fout:
            for line_no, line in enumerate(fin):
                if not line.strip():
                    continue
                req = json.loads(line)
                rid, tid, cidx, key = req_map[pos_of[(chunk_file.name, line_no)]]
                enriched_req = {
                    "request_id": rid,        # PRIMARY KEY (Child)
                    "trajectory_id": tid,     # FOREIGN KEY (Parent: results/routes.jsonl)
                    "call_index": cidx,       # 1-based call number in trajectory
                    "trajectory_key": key,    # reconstruction hash
                    "model": req.get("model"),
                    "input": req.get("input"),
                    "tools": req.get("tools"),
                }
                fout.write(json.dumps(enriched_req) + "\n")
                total_processed += 1
        tmp_file.replace(out_file)  # atomic-ish: no truncated output on crash

    print(f"\nDone! Enriched {total_processed} request lines across {len(jsonl_files)} file(s).")
    print(f"Enriched dataset saved to: {out_path.resolve()}")


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "export"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "export_linked"
    enrich_export(input_dir, output_dir)
