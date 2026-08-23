#!/usr/bin/env python3
"""End-to-end pipeline: extract -> route -> grade -> compare -> dashboard data.

  part 1 (router):    routing-time features from each trajectory's EARLIEST call
                      (first user message + system prompt + tools, nothing else)
                      -> Tier 1/2/3
  part 2 (evaluator): effort metrics from each trajectory's DEEPEST call
                      -> rule-based difficulty 1/2/3
  comparison:         agreement, confusion, under/over-routing, cost estimate
  dashboard:          dashboard/data.js with per-trajectory features, ranks and
                      out-of-fold ML probabilities so every hyperparameter can
                      be re-tuned live in the browser

Router methods (benchmarked in tune_router.py, all OOF under GroupKFold):
  blend  (default) - alpha*rank(ML difficulty) + (1-alpha)*heuristic, cut at
                     percentiles. Balanced winner: beats the heuristic on BOTH
                     the per-task and the token-weighted frontier.
  score            - unsupervised signed-group heuristic (no labels at all)
  kmeans           - unsupervised k-means (k=3), for comparison
  ml-tau           - cheapest tier with P(D<=t) >= tau (cheapest 90%-served)
  ml-lambda        - argmin_t cost$(t) + lambda*P(D>t) (best tasks-per-dollar;
                     known weakness: sacrifices token-heavy tasks)

The ML head trains on EVALUATOR labels (never the logged model id); shipped
probabilities are out-of-fold, so no task's tier saw its own outcome.

Usage: python run_pipeline.py [export.jsonl] [--method blend|score|kmeans|ml-tau|ml-lambda]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from router.features import extract as extract_router, FEATURE_GROUPS
from router.tiering import (route, rank_matrix, composite_scores, tiers_by_cuts,
                            tiers_by_kmeans, DEFAULT_GROUP_WEIGHTS, DEFAULT_CUTS)
from router.ml import (workspace_of, oof_cumulative_probs, blend_scores,
                       fold_ranks, first_user_text)
from evaluator.metrics import trajectory_metrics
from evaluator.difficulty import (grade, percentile_ranks, DEFAULT_WEIGHTS,
                                  DEFAULT_CUTS as EV_CUTS, DEFAULT_OVERRIDES)

ROOT = Path(__file__).parent
DEFAULT_EXPORT = r"D:\Github-Projects\entire-viktor-tumai-ehk\export_linked\trajectories_v1_01.jsonl"

# tier -> assumed [$ / 1M input, $ / 1M cached input, $ / 1M output]
# (representative rows from scripts/pricing.json; model ids are anonymized, so
# pricing is an assumption)
TIER_PRICES = {1: [0.2, 0.02, 1.2], 2: [2.0, 0.2, 10.0], 3: [5.0, 0.5, 25.0]}

# dispatch defaults picked from the tuned frontiers (alpha sweep: exact peaks
# at 1.0, weighted AUC at ~0.85 — 0.85 keeps both near their best)
DEFAULT_ALPHA, DEFAULT_TAU, DEFAULT_LAMBDA = 0.85, 0.80, 0.3


def load_trajectories(path):
    """-> {trajectory_id: {'first': req, 'deepest': req, 'n_calls': int, 'model': str}}"""
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                req = json.loads(line)
                groups[req["trajectory_id"]].append(req)
    out = {}
    for tid, reqs in groups.items():
        reqs.sort(key=lambda r: (r.get("call_index", 0), len(r["input"])))
        out[tid] = {"first": reqs[0], "deepest": reqs[-1], "n_calls": len(reqs),
                    "model": reqs[-1].get("model", "?")}
    return out


def costs_matrix(mets, cache_aware=True):
    """(n, 3) $ estimate per trajectory per tier. Cache-aware: every input token
    is paid uncached once; the growing prefix is replayed by later calls at the
    cached rate (linear-growth approx: replay ~= T*(n-1)/2). Routing whole
    tasks means no policy ever pays the cache-reset penalty."""
    out = np.zeros((len(mets), 3))
    for i, m in enumerate(mets):
        T, g, n = m["context_tokens"], m["gen_tokens"], max(m["n_llm_calls"], 1)
        for t in (1, 2, 3):
            pin, pc, pout = TIER_PRICES[t]
            if cache_aware:
                out[i, t - 1] = (T * pin + T * (n - 1) / 2 * pc + g * pout) / 1e6
            else:
                out[i, t - 1] = (T * pin + g * pout) / 1e6
    return out


def spearman(a, b):
    ra, rb = percentile_ranks(a), percentile_ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(tiers, diffs, r_scores, e_scores, C):
    n = len(tiers)
    conf = [[int(((tiers == r + 1) & (diffs == c + 1)).sum()) for c in range(3)]
            for r in range(3)]
    idx = np.arange(n)
    cost = float(C[idx, tiers - 1].sum())
    top = float(C[:, 2].sum())
    served = tiers >= diffs
    w = C[:, 2]
    return {
        "n": n,
        "exact_agreement": float((tiers == diffs).mean()),
        "adjacent_agreement": float((np.abs(tiers - diffs) <= 1).mean()),
        "spearman_scores": spearman(r_scores, e_scores),
        "under_routed": float((tiers < diffs).mean()),
        "over_routed": float((tiers > diffs).mean()),
        "served": float(served.mean()),
        "served_token_weighted": float((served * w).sum() / w.sum()),
        "confusion": conf,  # rows = router tier 1..3, cols = evaluator difficulty 1..3
        "est_cost_policy_usd": round(cost, 2),
        "est_cost_all_tier3_usd": round(top, 2),
        "est_savings_pct": round(100 * (1 - cost / top), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?", default=DEFAULT_EXPORT)
    ap.add_argument("--method", default="blend",
                    choices=["blend", "score", "kmeans", "ml-tau", "ml-lambda"])
    a = ap.parse_args()

    trajs = load_trajectories(a.export)
    tids = sorted(trajs)
    print(f"loaded {len(tids)} trajectories from {Path(a.export).name}")

    # part 1: routing-time view (earliest call only)
    feat_rows = [extract_router(trajs[t]["first"]) for t in tids]
    groups = np.array([workspace_of(trajs[t]["first"]) for t in tids])

    # part 2: evaluator (deepest call, full history) — frozen defaults
    metric_rows = [trajectory_metrics(trajs[t]["deepest"], trajs[t]["n_calls"])
                   for t in tids]
    diffs, e_scores = grade(metric_rows)

    # costs (cache-aware is the primary model)
    C = costs_matrix(metric_rows, cache_aware=True)
    C_naive = costs_matrix(metric_rows, cache_aware=False)

    # heuristic score + OOF ML probabilities
    ranks, feat_names = rank_matrix(feat_rows)
    h_scores = composite_scores(ranks, feat_names)
    print("training ML head (word+char TF-IDF + numeric, ordinal, GroupKFold OOF) ...")
    texts = [first_user_text(trajs[t]["first"]) for t in tids]
    cum = oof_cumulative_probs(feat_rows, diffs, groups, texts=texts)

    # dispatch
    if a.method == "score":
        tiers, r_scores = tiers_by_cuts(h_scores), h_scores
    elif a.method == "kmeans":
        tiers, r_scores = tiers_by_kmeans(ranks, feat_names, h_scores), h_scores
    elif a.method == "ml-tau":
        r_scores = fold_ranks(1 - (cum[:, 0] + cum[:, 1]) / 2,
                              1 - (cum[:, 0] + cum[:, 1]) / 2)
        tiers = np.where(cum[:, 0] >= DEFAULT_TAU, 1,
                         np.where(cum[:, 1] >= DEFAULT_TAU, 2, 3))
    elif a.method == "ml-lambda":
        r_scores = fold_ranks(1 - (cum[:, 0] + cum[:, 1]) / 2,
                              1 - (cum[:, 0] + cum[:, 1]) / 2)
        p_over = np.stack([1 - cum[:, 0], 1 - cum[:, 1], np.zeros(len(cum))], axis=1)
        tiers = np.argmin(C + DEFAULT_LAMBDA * p_over, axis=1) + 1
    else:  # blend (default)
        r_scores = blend_scores(cum, h_scores, DEFAULT_ALPHA)
        tiers = tiers_by_cuts(r_scores)
    tiers = np.asarray(tiers, dtype=int)

    report = compare(tiers, diffs, r_scores, e_scores, C)
    report["method"] = a.method
    report["naive_cost"] = compare(tiers, diffs, r_scores, e_scores, C_naive)[
        "est_cost_policy_usd"]

    # ---- results ----
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with open(results / "router_features.jsonl", "w", encoding="utf-8") as f:
        for r in feat_rows:
            f.write(json.dumps(r) + "\n")
    with open(results / "evaluator_metrics.jsonl", "w", encoding="utf-8") as f:
        for t, m in zip(tids, metric_rows):
            f.write(json.dumps({"trajectory_id": t, **m}) + "\n")
    with open(results / "tiers.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(tids):
            f.write(json.dumps({
                "trajectory_id": t, "method": a.method,
                "router_tier": int(tiers[i]), "router_score": round(float(r_scores[i]), 4),
                "p_d1": round(float(cum[i, 0]), 4), "p_d2": round(float(cum[i, 1]), 4),
                "evaluator_difficulty": int(diffs[i]),
                "evaluator_score": round(float(e_scores[i]), 4),
            }) + "\n")
    with open(results / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- dashboard data (features + ranks + OOF probs; knobs applied in JS) ----
    ev_names = list(DEFAULT_WEIGHTS)
    ev_ranks = np.column_stack([percentile_ranks([m[k] for m in metric_rows])
                                for k in ev_names])
    rows = []
    for i, t in enumerate(tids):
        rows.append({
            "id": t,
            "model": trajs[t]["model"],          # display only — never a router input
            "trigger": feat_rows[i]["_trigger"],
            "preview": feat_rows[i]["_preview"],
            "feat": {k: feat_rows[i][k] for k in feat_names},
            "rank": [round(float(x), 4) for x in ranks[i]],
            "cum": [round(float(cum[i, 0]), 4), round(float(cum[i, 1]), 4)],
            "ev": {k: metric_rows[i][k] for k in
                   list(DEFAULT_WEIGHTS) + ["n_assistant_msgs", "n_reasoning_items",
                                            "n_logged_calls"]},
            "evRank": [round(float(x), 4) for x in ev_ranks[i]],
        })
    data = {
        "meta": {
            "source": Path(a.export).name,
            "n": len(tids),
            "featureGroups": FEATURE_GROUPS,
            "featNames": feat_names,
            "evNames": ev_names,
            "defaults": {
                "groupWeights": DEFAULT_GROUP_WEIGHTS,
                "routerCuts": list(DEFAULT_CUTS),
                "evWeights": DEFAULT_WEIGHTS,
                "evCuts": list(EV_CUTS),
                "overrides": DEFAULT_OVERRIDES,
                "tierPrices": TIER_PRICES,
                "alpha": DEFAULT_ALPHA, "tau": DEFAULT_TAU, "lambda": DEFAULT_LAMBDA,
                "method": a.method, "costModel": "cache",
            },
        },
        "rows": rows,
    }
    dash = ROOT / "dashboard"
    dash.mkdir(exist_ok=True)
    with open(dash / "data.js", "w", encoding="utf-8") as f:
        f.write("window.VIKTOR_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")

    # ---- console report ----
    print(f"router method={a.method}  tiers: " +
          " ".join(f"T{k}={int((tiers == k).sum())}" for k in (1, 2, 3)))
    print("evaluator difficulty: " +
          " ".join(f"D{k}={int((diffs == k).sum())}" for k in (1, 2, 3)))
    print(f"exact agreement {report['exact_agreement']:.1%}  "
          f"adjacent {report['adjacent_agreement']:.1%}  "
          f"spearman {report['spearman_scores']:.3f}")
    print(f"served {report['served']:.1%}  token-weighted served "
          f"{report['served_token_weighted']:.1%}  "
          f"under-routed {report['under_routed']:.1%}")
    print(f"est. cost (cache-aware): policy ${report['est_cost_policy_usd']}  "
          f"all-Tier3 ${report['est_cost_all_tier3_usd']}  "
          f"savings {report['est_savings_pct']}%  "
          f"(naive model: ${report['naive_cost']}; chars/4 estimates)")
    print("wrote results/ + dashboard/data.js")


if __name__ == "__main__":
    main()
