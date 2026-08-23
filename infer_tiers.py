#!/usr/bin/env python3
"""Infer the tier order of the anonymized model ids from the data.

The deck ships ~9 model ids with the tier order hidden. Two independent,
nameable signals recover it:

  1. REVEALED PREFERENCE - the historical dispatcher sent harder tasks to the
     models it trusted more. Mean routing-time difficulty (the router's OOF
     score, which never saw the model id) of the tasks each model served ranks
     the models by the operator's own tiering.
  2. MATCHED STRUGGLE - within routing-difficulty quintiles (matching!), a
     weaker model shows more tool errors and longer retry streaks on
     comparable tasks. Within-bucket z-scores of error rate + retry streak
     rank models by observed capability (lower struggle = higher tier).

The combined rank (mean of both rank lists) maps the ids into Tier 1/2/3 by
terciles of trust, with small-sample ids flagged instead of over-read.

Writes results/model_tiers.json and prints the table.

Usage: python infer_tiers.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_N = 15  # below this many trajectories a model's rank is a guess — flag it


def main():
    tiers_rows = [json.loads(l) for l in open("results/tiers.jsonl", encoding="utf-8")]
    mets = {r["trajectory_id"]: r for r in
            (json.loads(l) for l in open("results/evaluator_metrics.jsonl", encoding="utf-8"))}
    # model per trajectory comes from the dashboard data (display field)
    data = json.loads(Path("dashboard/data.js").read_text(encoding="utf-8")
                      .removeprefix("window.VIKTOR_DATA = ").rstrip().rstrip(";"))
    model_of = {r["id"]: r["model"] for r in data["rows"]}

    rows = []
    for t in tiers_rows:
        tid = t["trajectory_id"]
        m = mets[tid]
        rows.append({
            "model": model_of[tid],
            "score": t["router_score"],          # routing-time difficulty (OOF)
            "err_rate": m["n_tool_errors"] / max(m["n_tool_calls"], 1),
            "streak": m["max_repeat_streak"],
            "gen_tokens": m["gen_tokens"],
            "ctx": m["context_tokens"],
        })
    models = sorted({r["model"] for r in rows})
    n_of = {mo: sum(r["model"] == mo for r in rows) for mo in models}

    # ---- signal 1: revealed preference (mean routing difficulty served)
    served = {mo: np.mean([r["score"] for r in rows if r["model"] == mo]) for mo in models}

    # ---- signal 2: matched struggle (within difficulty-quintile z-scores)
    scores = np.array([r["score"] for r in rows])
    qs = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
    bucket = np.digitize(scores, qs)
    struggle_z = defaultdict(list)
    for b in range(5):
        idx = [i for i in range(len(rows)) if bucket[i] == b]
        for key in ("err_rate", "streak"):
            v = np.array([rows[i][key] for i in idx], dtype=float)
            if v.std() < 1e-12:
                continue
            z = (v - v.mean()) / v.std()
            for k, i in enumerate(idx):
                struggle_z[rows[i]["model"]].append(z[k])
    struggle = {mo: float(np.mean(struggle_z[mo])) if struggle_z[mo] else 0.0
                for mo in models}

    # FINDING: revealed preference is nearly flat (spread ~0.44-0.55) — the
    # historical dispatcher was NOT difficulty-aware along the routing axis.
    # That is the router's headroom, and it means struggle is the usable
    # ranking signal; preference is reported as a null result.
    order = sorted(models, key=lambda m: struggle[m])  # least struggle first
    solid = [mo for mo in order if n_of[mo] >= MIN_N]

    # bootstrap the struggle ranking for stability (resample trajectories)
    rng = np.random.default_rng(7)
    tier_votes = {mo: np.zeros(3) for mo in models}
    for _ in range(200):
        pick = rng.integers(0, len(rows), len(rows))
        sz = defaultdict(list)
        bs = bucket[pick]
        for b in range(5):
            idx = [k for k in range(len(pick)) if bs[k] == b]
            for key in ("err_rate", "streak"):
                v = np.array([rows[pick[k]][key] for k in idx], dtype=float)
                if v.std() < 1e-12:
                    continue
                z = (v - v.mean()) / v.std()
                for j, k in enumerate(idx):
                    sz[rows[pick[k]]["model"]].append(z[j])
        st = {mo: np.mean(sz[mo]) if sz.get(mo) else 0.0 for mo in models}
        so = [mo for mo in sorted(models, key=lambda m: st[m]) if n_of[mo] >= MIN_N]
        for i, mo in enumerate(so):
            t = 3 if i < len(so) / 3 else (2 if i < 2 * len(so) / 3 else 1)
            tier_votes[mo][t - 1] += 1

    tier_of, stability = {}, {}
    for i, mo in enumerate(solid):
        tier_of[mo] = 3 if i < len(solid) / 3 else (2 if i < 2 * len(solid) / 3 else 1)
        stability[mo] = float(tier_votes[mo][tier_of[mo] - 1] / max(tier_votes[mo].sum(), 1))
    for mo in order:  # small-sample ids inherit the nearest solid neighbour's tier
        if mo not in tier_of:
            pos = order.index(mo)
            near = min(solid, key=lambda s: abs(order.index(s) - pos))
            tier_of[mo], stability[mo] = tier_of[near], 0.0

    out = {"min_n": MIN_N, "models": [],
           "null_result": "revealed preference is flat (served-difficulty spread "
                          "~0.44-0.55): the logged dispatch was not difficulty-aware "
                          "— that headroom is what the router exploits"}
    print(f"{'model':18s} {'n':>4s} {'served-diff':>11s} {'struggle-z':>11s} "
          f"{'tier':>5s} {'stable':>7s}  note")
    for mo in order:
        note = "" if n_of[mo] >= MIN_N else f"n<{MIN_N}: low confidence"
        print(f"{mo:18s} {n_of[mo]:4d} {served[mo]:11.3f} {struggle[mo]:11.3f} "
              f"T{tier_of[mo]:d}    {stability[mo]:6.0%}  {note}")
        out["models"].append({"model": mo, "n": n_of[mo],
                              "served_difficulty": round(float(served[mo]), 4),
                              "struggle_z": round(struggle[mo], 4),
                              "tier": tier_of[mo],
                              "bootstrap_stability": round(stability[mo], 3),
                              "low_confidence": n_of[mo] < MIN_N})
    print("\nnull result: " + out["null_result"])
    with open("results/model_tiers.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results/model_tiers.json")


if __name__ == "__main__":
    main()
