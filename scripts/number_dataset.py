#!/usr/bin/env python3
"""Renumber trajectories and requests with clean sequential IDs (1 to N).

Updates:
1. `results/routes.jsonl`:
   - `trajectory_id`: 1, 2, 3 ... 978
   - `hex_hash`: original 16-char hex hash (e.g. "92691dd8d434e0d5")

2. `export/*.jsonl` (and `export_linked/*.jsonl`):
   - `request_id`: 1, 2, 3 ... 1153 (Sequential Primary Key)
   - `trajectory_id`: 1, 2, 3 ... 978 (Foreign Key)
   - `call_index`: 1, 2, 3 ... (1-indexed call number within trajectory)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from load_trajectories import iter_requests, group_key

def renumber_all(export_dir: str = "export", routes_file: str = "results/routes.jsonl"):
    exp_path = Path(export_dir)
    jsonl_files = sorted(exp_path.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"No *.jsonl files in {export_dir}")

    print("Reading requests and grouping trajectories...")
    requests_meta = list(iter_requests(export_dir))

    # Group requests by original prompt hash key
    groups = defaultdict(list)
    for chunk_name, line_no, req in requests_meta:
        hash_key = group_key(req)
        groups[hash_key].append((chunk_name, line_no, req))

    # Assign sequential trajectory_id (1 to N) deterministically
    sorted_hash_keys = list(groups.keys())
    hash_to_seq_id = {hash_key: idx + 1 for idx, hash_key in enumerate(sorted_hash_keys)}

    # Build request mapping: (chunk_name, line_no) -> (seq_traj_id, call_idx_1based)
    req_map = {}
    global_req_id = 1
    for hash_key in sorted_hash_keys:
        group_reqs = groups[hash_key]
        # Sort by input history length (= call order)
        group_reqs.sort(key=lambda item: len(item[2]["input"]))
        seq_traj_id = hash_to_seq_id[hash_key]
        for call_idx_0based, (chunk_name, line_no, req) in enumerate(group_reqs):
            req_map[(chunk_name, line_no)] = (global_req_id, seq_traj_id, call_idx_0based + 1)
            global_req_id += 1

    # 1. Update export files in-place (and export_linked/ if present)
    total_requests_processed = 0
    for chunk_file in jsonl_files:
        lines = []
        with open(chunk_file, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin):
                if not line.strip():
                    continue
                req = json.loads(line)
                req_id, traj_id, call_idx = req_map[(chunk_file.name, line_no)]
                
                # Format request with clean 1..N sequential IDs
                renumbered_req = {
                    "request_id": req_id,       # PK (1..1153)
                    "trajectory_id": traj_id,   # FK (1..978)
                    "call_index": call_idx,      # Call # in trajectory (1, 2, 3...)
                    "model": req.get("model"),
                    "input": req.get("input"),
                    "tools": req.get("tools")
                }
                lines.append(renumbered_req)
                total_requests_processed += 1

        with open(chunk_file, "w", encoding="utf-8") as fout:
            for r in lines:
                fout.write(json.dumps(r) + "\n")
        print(f"Updated {chunk_file} with sequential IDs.")

    # Also update export_linked if it exists
    linked_dir = Path("export_linked")
    if linked_dir.exists():
        for chunk_file in jsonl_files:
            target_linked = linked_dir / chunk_file.name
            if target_linked.exists():
                with open(chunk_file, "r", encoding="utf-8") as fin, \
                     open(target_linked, "w", encoding="utf-8") as fout:
                    fout.write(fin.read())

    # 2. Update results/routes.jsonl
    routes_path = Path(routes_file)
    if routes_path.exists():
        route_records = []
        with open(routes_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                hash_key = rec.get("trajectory") or rec.get("hex_hash")
                seq_traj_id = hash_to_seq_id.get(hash_key, len(route_records) + 1)
                
                renumbered_rec = {
                    "trajectory_id": seq_traj_id,          # PK (1..978)
                    "hex_hash": hash_key,                  # Original SHA1 hash
                    "n_calls": rec.get("n_calls"),
                    "logged_model": rec.get("logged_model"),
                    "route": rec.get("route"),
                    "cost_logged_usd": rec.get("cost_logged_usd"),
                    "cost_routed_usd": rec.get("cost_routed_usd"),
                    "switches": rec.get("switches")
                }
                route_records.append(renumbered_rec)

        # Sort route records by trajectory_id (1..978)
        route_records.sort(key=lambda r: r["trajectory_id"])

        with open(routes_path, "w", encoding="utf-8") as fout:
            for r in route_records:
                fout.write(json.dumps(r) + "\n")
        print(f"Updated {routes_file} with sequential trajectory_id (1..{len(route_records)}).")

    print(f"\nDone! Renumbered {total_requests_processed} requests across {len(hash_to_seq_id)} trajectories.")

if __name__ == "__main__":
    renumber_all()
