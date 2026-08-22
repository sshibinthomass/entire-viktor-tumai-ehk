#!/usr/bin/env python3
"""Enrich raw trajectory JSONL exports with explicit Primary Keys and Foreign Keys.

Parent Record (results/routes.jsonl):
- Primary Key: `trajectory` (e.g. "92691dd8d434e0d5")

Child Record (export_linked/*.jsonl):
- Primary Key (PK): `request_id` (e.g. "92691dd8d434e0d5_0")
- Foreign Key (FK): `trajectory_id` (references `trajectory` in results/routes.jsonl)
- `call_index`: integer index (0, 1, 2...) of call within trajectory

Usage:
    python scripts/enrich_dataset.py export/ export_linked/
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from load_trajectories import iter_requests, group_key

def enrich_export(input_dir: str, output_dir: str):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(in_path.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"No *.jsonl files found in {input_dir}")

    print(f"Reading requests from {input_dir}...")
    requests_with_meta = list(iter_requests(input_dir))

    # Group requests while preserving chunk filename and line number
    groups = defaultdict(list)
    for chunk_name, line_no, req in requests_with_meta:
        key = group_key(req)
        groups[key].append((chunk_name, line_no, req))

    # Build lookup map: (chunk_name, line_no) -> (trajectory_id, call_index)
    req_map = {}
    for traj_id, group_reqs in groups.items():
        # Order group by input history length (= call order)
        group_reqs.sort(key=lambda item: len(item[2]["input"]))
        for idx, (chunk_name, line_no, req) in enumerate(group_reqs):
            req_map[(chunk_name, line_no)] = (traj_id, idx)

    # Process each chunk file and output enriched version
    total_processed = 0
    for chunk_file in jsonl_files:
        out_file = out_path / chunk_file.name
        print(f"Writing enriched chunk: {out_file.name}...")
        
        with open(chunk_file, "r", encoding="utf-8") as fin, \
             open(out_file, "w", encoding="utf-8") as fout:
            for line_no, line in enumerate(fin):
                if not line.strip():
                    continue
                req = json.loads(line)
                traj_id, call_idx = req_map[(chunk_file.name, line_no)]
                
                # Construct enriched record with PK and FK
                enriched_req = {
                    "request_id": f"{traj_id}_{call_idx}",   # PRIMARY KEY (Child)
                    "trajectory_id": traj_id,               # FOREIGN KEY (Parent: results/routes.jsonl -> trajectory)
                    "call_index": call_idx,                  # Sequence position in trajectory
                    "model": req.get("model"),
                    "input": req.get("input"),
                    "tools": req.get("tools")
                }
                
                fout.write(json.dumps(enriched_req) + "\n")
                total_processed += 1

    print(f"\nDone! Enriched {total_processed} request lines across {len(jsonl_files)} file(s).")
    print(f"Enriched dataset saved to: {out_path.resolve()}")

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "export"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "export_linked"
    enrich_export(input_dir, output_dir)
