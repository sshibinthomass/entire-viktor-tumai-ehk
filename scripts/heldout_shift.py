#!/usr/bin/env python3
"""Distribution-shift check between the FIT chunk and a HELD-OUT chunk.

`apply_frozen.py` reports how well the frozen router agrees with the frozen
evaluator on new data. It cannot say *why* the numbers moved. This script
answers the prior question: is the new chunk the same population?

Two comparisons, both on the quantities the solution actually consumes:

  1. ROUTING FEATURES — `router/features.py` reads only the system prompt, the
     first user message and the tool definitions, so these are computable on
     any call of a trajectory (the opening items are a prefix of every later
     input). Shift here moves the router's ECDF ranks and therefore its tiers.
  2. EVALUATOR METRICS — `evaluator/metrics.py` counts effort inside the
     deepest logged call. Shift here moves the frozen yardstick's grades.

For each quantity we report the fit/new medians, the two-sample KS statistic,
and `mean_fit_rank`: the average ECDF rank of the new values under the FIT
distribution (0.5 = aligned, >0.5 = new chunk is systematically larger). Ranks
are what the frozen router and evaluator actually apply, so this is the shift
that moves tiers. It uses the MID-ECDF (mean of the left and right ECDF) —
`rank_utils.ecdf_ranks` is right-continuous by design, which is correct inside
the router but would report ~1.0 for an all-zero binary feature here and read
as a shift where there is none.

Also reported: LOGGED CALLS PER TRAJECTORY, because the chunks are not all
sampled alike (chunk 00 ships whole trajectories, chunk 01+ ship one deep call
per task). Either shape routes and grades fine — routing features live in the
opening items, which are a prefix of every call, and `n_llm_calls` is recovered
from the item structure, not from how many lines were logged — but in a
one-call chunk `n_logged_calls` is 1 by construction and effort counters are
truncated at the sampled call. Print it so nobody reads a sampling difference
as a behavioural one.

Usage:
  python scripts/heldout_shift.py export_linked/trajectories_v1_02.jsonl \
      --fit export_linked/trajectories_v1_01.jsonl \
      [--out results/heldout_shift.json] [--top 12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on the path

from router.features import extract as extract_router
from router.ml import FEAT_NAMES
from evaluator.metrics import trajectory_metrics
from evaluator.difficulty import DEFAULT_WEIGHTS as EV_WEIGHTS
from run_pipeline import load_trajectories

METRIC_NAMES = list(EV_WEIGHTS)


def summarize(path):
    """-> (features dict of arrays, metrics dict of arrays, shape stats)."""
    trajs = load_trajectories(path)
    tids = sorted(trajs)
    feats = [extract_router(trajs[t]["first"]) for t in tids]
    mets = [trajectory_metrics(trajs[t]["deepest"], trajs[t]["n_calls"]) for t in tids]
    F = {f: np.array([float(r[f]) for r in feats]) for f in FEAT_NAMES}
    M = {k: np.array([float(m[k]) for m in mets]) for k in METRIC_NAMES}
    logged = np.array([trajs[t]["n_calls"] for t in tids], dtype=float)
    shape = {"n_trajectories": len(tids),
             "n_requests": int(logged.sum()),
             "logged_calls_median": float(np.median(logged)),
             "logged_calls_max": int(logged.max()),
             "one_call_only_pct": round(100 * float((logged == 1).mean()), 1)}
    return F, M, shape


def compare(fit, new):
    """-> [{name, fit_median, new_median, ks, mean_fit_rank}] sorted by |shift|."""
    rows = []
    for name, a in fit.items():
        b = new[name]
        srt = np.sort(a)
        if srt[0] == srt[-1]:                      # constant fit column -> neutral
            rank = np.full(b.shape, 0.5)
        else:                                      # mid-ECDF: ties centred on their mass
            lo = np.searchsorted(srt, b, side="left")
            hi = np.searchsorted(srt, b, side="right")
            rank = (lo + hi) / (2 * len(srt))
        rows.append({
            "name": name,
            "fit_median": round(float(np.median(a)), 3),
            "new_median": round(float(np.median(b)), 3),
            "ks": round(float(ks_2samp(a, b).statistic), 3),
            "mean_fit_rank": round(float(rank.mean()), 3),
        })
    rows.sort(key=lambda r: -abs(r["mean_fit_rank"] - 0.5))
    return rows


def _print(title, rows, top):
    print(f"\n{title}  (mean_fit_rank 0.5 = aligned; >0.5 = new chunk larger)")
    print(f"  {'quantity':<24} {'fit med':>9} {'new med':>9} {'KS':>6} {'rank':>6}")
    for r in rows[:top]:
        print(f"  {r['name']:<24} {r['fit_median']:>9.3g} {r['new_median']:>9.3g} "
              f"{r['ks']:>6.3f} {r['mean_fit_rank']:>6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new", help="enriched held-out chunk")
    ap.add_argument("--fit", required=True, help="enriched fit chunk (chunk 01)")
    ap.add_argument("--out", default="results/heldout_shift.json")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    # one chunk in memory at a time — these files are ~100 MB each
    fF, fM, fS = summarize(a.fit)
    print(f"fit  {Path(a.fit).name}: {fS['n_trajectories']} trajectories, "
          f"{fS['n_requests']} requests, median {fS['logged_calls_median']:.0f} "
          f"logged calls/trajectory ({fS['one_call_only_pct']}% single-call)")
    nF, nM, nS = summarize(a.new)
    print(f"new  {Path(a.new).name}: {nS['n_trajectories']} trajectories, "
          f"{nS['n_requests']} requests, median {nS['logged_calls_median']:.0f} "
          f"logged calls/trajectory ({nS['one_call_only_pct']}% single-call)")

    feat_rows, met_rows = compare(fF, nF), compare(fM, nM)
    _print("ROUTING FEATURES — most shifted", feat_rows, a.top)
    _print("EVALUATOR METRICS — most shifted", met_rows, a.top)

    n_big = sum(1 for r in feat_rows if abs(r["mean_fit_rank"] - 0.5) > 0.15)
    print(f"\n{n_big}/{len(feat_rows)} routing features shifted by >0.15 rank; "
          f"median |shift| {np.median([abs(r['mean_fit_rank'] - 0.5) for r in feat_rows]):.3f}")
    if abs(nS["one_call_only_pct"] - fS["one_call_only_pct"]) > 40:
        print("NOTE: the two chunks are sampled differently (one call per task vs "
              "whole trajectories). Routing features are unaffected — the opening "
              "items are a prefix of every call — and n_llm_calls is recovered from "
              "the item structure, but n_logged_calls and the effort counters are "
              "not comparable across the two shapes.")
    else:
        print(f"sampling shape matches ({nS['one_call_only_pct']}% vs "
              f"{fS['one_call_only_pct']}% single-call trajectories) — effort "
              f"counters are comparable across the two chunks.")

    rep = {"fit_source": Path(a.fit).name, "new_source": Path(a.new).name,
           "fit_shape": fS, "new_shape": nS,
           "features": feat_rows, "evaluator_metrics": met_rows,
           "n_features_shifted_gt_0_15": n_big}
    Path("results").mkdir(exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
