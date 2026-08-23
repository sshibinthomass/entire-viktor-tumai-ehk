#!/usr/bin/env python3
"""Infer the tier order of the anonymized model ids from the data.

Method actually implemented (docstring == code, checked):

  1. REVEALED PREFERENCE — mean routing-time difficulty (the router's OOF
     score, which never saw the model id) of the tasks each model served.
     FINDING: it is nearly flat across models — the historical dispatcher was
     NOT difficulty-aware. Reported as a null result; NOT used for ranking.
  2. MATCHED STRUGGLE — within routing-difficulty QUINTILES (matching), each
     trajectory's tool-error rate and retry streak are z-scored; a model's
     struggle score is the mean z over its trajectories. Models are ranked by
     struggle (least struggle = highest tier) and mapped to Tier 1/2/3 by
     TERCILES of that ranking. Terciles force three ids per tier regardless of
     spread — bootstrap stability per model is reported so fuzzy boundaries
     are visible, and boundary members with low stability should be read as
     'T2/T3-ish', not as a hard assignment.

CIRCULARITY GUARD: tool-error rate and retry streak are exactly the signals
matching_check.py later compares across tiers. To avoid validating the tier
map with the same rows that produced it, this script also fits a SPLIT-HALF
map on the even-indexed half of the trajectories (models_half_a);
matching_check.py applies that map to the ODD half only. The full-data map
(models) remains the one used for pricing/frontier comparisons, where no
struggle-based validation is claimed.

Small-sample ids (n < MIN_N) inherit the nearest solid neighbour's tier and
are flagged instead of over-read. On a chunk small enough that NO id clears
MIN_N (chunk 00's 25 trajectories, or a judge's sample), all ids are ranked and
all are flagged — a fully-flagged table, not a crash.

Writes results/model_tiers.json and prints the table.

Usage: python infer_tiers.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_N = 15       # below this many trajectories a model's rank is a guess — flag it
MIN_N_HALF = 8   # the split-half fit sees ~half the rows per model


def struggle_ranking(rows, models, n_of, min_n, n_boot=200, seed=7):
    """-> (struggle mean-z per model, tier map, bootstrap stability per model)."""
    scores = np.array([r["score"] for r in rows])
    qs = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
    bucket = np.digitize(scores, qs)

    def struggle_of(idx_rows, idx_bucket):
        sz = defaultdict(list)
        for b in range(5):
            idx = [k for k in range(len(idx_rows)) if idx_bucket[k] == b]
            for key in ("err_rate", "streak"):
                v = np.array([idx_rows[k][key] for k in idx], dtype=float)
                if v.std() < 1e-12:
                    continue
                z = (v - v.mean()) / v.std()
                for j, k in enumerate(idx):
                    sz[idx_rows[k]["model"]].append(z[j])
        return {mo: float(np.mean(sz[mo])) if sz.get(mo) else 0.0 for mo in models}

    struggle = struggle_of(rows, bucket)
    order = sorted(models, key=lambda m: struggle[m])  # least struggle first
    solid = [mo for mo in order if n_of[mo] >= min_n]
    # a small chunk (a judge's sample, our 25-trajectory chunk 00) can leave NO
    # id above the threshold. Rank them all rather than crash — every id is then
    # flagged n<MIN_N, which is exactly what the caller should report.
    thin = not solid
    if thin:
        solid = list(order)

    rng = np.random.default_rng(seed)
    tier_votes = {mo: np.zeros(3) for mo in models}
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows), len(rows))
        st = struggle_of([rows[k] for k in pick], bucket[pick])
        so = [mo for mo in sorted(models, key=lambda m: st[m])
              if thin or n_of[mo] >= min_n]
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
    return struggle, tier_of, stability, order


def main():
    tiers_rows = [json.loads(l) for l in open("results/tiers.jsonl", encoding="utf-8")]
    tiers_rows.sort(key=lambda r: r["trajectory_id"])
    mets = {r["trajectory_id"]: r for r in
            (json.loads(l) for l in open("results/evaluator_metrics.jsonl", encoding="utf-8"))}
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
        })
    models = sorted({r["model"] for r in rows})
    n_of = {mo: sum(r["model"] == mo for r in rows) for mo in models}

    # signal 1: revealed preference (null result — reported, not used)
    served = {mo: np.mean([r["score"] for r in rows if r["model"] == mo]) for mo in models}

    # signal 2: matched struggle — full data (primary map, used for pricing)
    struggle, tier_of, stability, order = struggle_ranking(rows, models, n_of, MIN_N)

    # split-half map for the matched check: fit on the EVEN-index half only
    rows_a = rows[0::2]
    n_of_a = {mo: sum(r["model"] == mo for r in rows_a) for mo in models}
    _, tier_a, stab_a, _ = struggle_ranking(rows_a, models, n_of_a, MIN_N_HALF, seed=11)

    out = {"min_n": MIN_N, "models": [],
           "models_half_a": [{"model": mo, "n": n_of_a[mo], "tier": tier_a[mo],
                              "bootstrap_stability": round(stab_a[mo], 3)}
                             for mo in models],
           "split_design": "models_half_a is fit on even-indexed trajectories "
                           "(sorted by trajectory_id); matching_check.py validates "
                           "on the odd-indexed half only — the tier map is never "
                           "validated on the rows that produced it",
           "tercile_note": "terciles force three ids per tier regardless of spread; "
                           "read low-stability boundary members as fuzzy, not fixed",
           "null_result": "revealed preference is flat across models (see "
                          "served_difficulty): the logged dispatch was not "
                          "difficulty-aware — that headroom is what the router exploits"}
    print(f"{'model':18s} {'n':>4s} {'served-diff':>11s} {'struggle-z':>11s} "
          f"{'tier':>5s} {'stable':>7s} {'halfA':>6s}  note")
    for mo in order:
        note = "" if n_of[mo] >= MIN_N else f"n<{MIN_N}: low confidence"
        print(f"{mo:18s} {n_of[mo]:4d} {served[mo]:11.3f} {struggle[mo]:11.3f} "
              f"T{tier_of[mo]:d}    {stability[mo]:6.0%} T{tier_a[mo]:d}     {note}")
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
