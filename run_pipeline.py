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
  ml-lambda        - argmin_t cost_hat(t) + lambda*P(D>t) where cost_hat is a
                     ROUTING-TIME cost prediction (OOF regressors on the same
                     routing features — realized cost is hindsight and would
                     make the rule an oracle)

The ML head trains on EVALUATOR labels (never the logged model id); shipped
probabilities are out-of-fold, so no task's tier saw its own outcome.

License note: the dataset is challenge-use-only. Verbatim text (previews,
trigger headers) is written ONLY to dashboard/previews.js, which is gitignored;
every committed artifact carries numbers, ids and model names only.

Usage: python run_pipeline.py [export.jsonl] [--method blend|score|kmeans|ml-tau|ml-lambda]
                              [--residualize]
"""
import argparse
import json
import sys
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
DEFAULT_EXPORT = str(ROOT / "export_linked")  # a directory (all chunks) or one .jsonl file

# tier -> assumed [$ / 1M input, $ / 1M cached input, $ / 1M output]
# (model ids are anonymized, so pricing is an assumption). T1/T2 are the luna and
# terra rows from scripts/pricing.json; T3 is a mid opus-class assumption sitting
# a little under the opus-5 row ($5/$0.5/$25). The inferred tier map also puts
# fable-5 ($10/$1/$50) in Tier 3 — the tier3_price_sensitivity entry in
# comparison.json shows the headline under that pricier representative too.
TIER_PRICES = {1: [0.2, 0.02, 1.2], 2: [2.0, 0.2, 12.0], 3: [4.0, 0.4, 20.0]}
TIER3_FABLE = [10.0, 1.0, 50.0]

# dispatch defaults picked from the committed sweeps (sweep_defaults.py ->
# results/sweeps.json)
DEFAULT_ALPHA, DEFAULT_TAU, DEFAULT_LAMBDA = 0.85, 0.80, 0.3


def load_trajectories(path):
    """-> {trajectory_id: {'first': req, 'deepest': req, 'n_calls': int, 'model': str}}

    `path` may be one enriched .jsonl chunk or a directory of them (enrich
    assigns globally unique trajectory ids across chunks)."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"export not found: {p}\n"
                 f"Extract the dataset to export/ and run "
                 f"'python scripts/enrich_dataset.py export/ export_linked/' first.")
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"no *.jsonl chunks in {p} — run scripts/enrich_dataset.py first.")
    groups = defaultdict(list)
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    req = json.loads(line)
                    if "trajectory_id" not in req:
                        sys.exit(f"{fp.name} has no trajectory_id — this is a RAW export "
                                 f"line. Run 'python scripts/enrich_dataset.py export/ "
                                 f"export_linked/' and point the pipeline at export_linked/.")
                    groups[req["trajectory_id"]].append(req)
    out = {}
    for tid, reqs in groups.items():
        reqs.sort(key=lambda r: (r.get("call_index", 0), len(r["input"])))
        out[tid] = {"first": reqs[0], "deepest": reqs[-1], "n_calls": len(reqs),
                    "model": reqs[-1].get("model", "?")}
    return out


def costs_matrix(mets, cache_aware=True, tier_prices=None):
    """(n, 3) $ estimate per trajectory per tier. Cache-aware: every input token
    is paid uncached once; the growing prefix is replayed by later calls at the
    cached rate (linear-growth approx: replay ~= T*(n-1)/2). Routing whole
    tasks means no policy ever pays the cache-reset penalty. cache_aware=False
    is the CACHE-BLIND model (no replay term at all)."""
    tier_prices = tier_prices or TIER_PRICES
    out = np.zeros((len(mets), 3))
    for i, m in enumerate(mets):
        T, g, n = m["context_tokens"], m["gen_tokens"], max(m["n_llm_calls"], 1)
        for t in (1, 2, 3):
            pin, pc, pout = tier_prices[t]
            if cache_aware:
                out[i, t - 1] = (T * pin + T * (n - 1) / 2 * pc + g * pout) / 1e6
            else:
                out[i, t - 1] = (T * pin + g * pout) / 1e6
    return out


def oof_predicted_costs(feat_rows, mets, groups, n_splits=5, tier_prices=None):
    """ROUTING-TIME cost matrix: per-fold regressors predict log1p(context
    tokens), log1p(gen tokens) and log1p(llm calls) from the routing features,
    and the tier-price formula is applied to the predictions. This is what a
    deployable lambda-Bayes rule can actually know at dispatch — using the
    realized metrics instead would leak each task's hindsight size."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from router.ml import FEAT_NAMES

    n = len(feat_rows)
    V = {f: np.array([float(r[f]) for r in feat_rows]) for f in FEAT_NAMES}
    targets = {k: np.log1p([m[k] for m in mets])
               for k in ("context_tokens", "gen_tokens", "n_llm_calls")}
    preds = {k: np.zeros(n) for k in targets}
    n_splits = min(n_splits, len(set(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(n), groups=groups):
        R = np.column_stack([fold_ranks(V[f][tr], V[f]) for f in FEAT_NAMES])
        for k, y in targets.items():
            reg = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                                learning_rate=0.06, random_state=13)
            reg.fit(R[tr], y[tr])
            preds[k][te] = reg.predict(R[te])
    T = np.maximum(np.expm1(preds["context_tokens"]), 0)
    g = np.maximum(np.expm1(preds["gen_tokens"]), 0)
    ncalls = np.maximum(np.expm1(preds["n_llm_calls"]), 1)
    tier_prices = tier_prices or TIER_PRICES
    C_hat = np.zeros((n, 3))
    for t in (1, 2, 3):
        pin, pc, pout = tier_prices[t]
        C_hat[:, t - 1] = (T * pin + T * (ncalls - 1) / 2 * pc + g * pout) / 1e6
    return C_hat


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
    ap.add_argument("export", nargs="?", default=DEFAULT_EXPORT,
                    help="enriched export (run scripts/enrich_dataset.py first)")
    ap.add_argument("--method", default="blend",
                    choices=["blend", "score", "kmeans", "ml-tau", "ml-lambda"])
    ap.add_argument("--residualize", action="store_true",
                    help="residualize policy-sensitive evaluator metrics per "
                         "inferred model tier (needs results/model_tiers.json)")
    a = ap.parse_args()

    trajs = load_trajectories(a.export)
    tids = sorted(trajs)
    print(f"loaded {len(tids)} trajectories from {Path(a.export).name}")

    # part 1: routing-time view (earliest call only)
    feat_rows = [extract_router(trajs[t]["first"]) for t in tids]
    groups = np.array([workspace_of(trajs[t]["first"]) for t in tids])
    print(f"workspace groups for CV: {len(set(groups))}")

    # part 2: evaluator (deepest call, full history) — frozen defaults
    metric_rows = [trajectory_metrics(trajs[t]["deepest"], trajs[t]["n_calls"])
                   for t in tids]
    residual_tiers = None
    if a.residualize:
        mt_path = ROOT / "results" / "model_tiers.json"
        if not mt_path.exists():
            sys.exit("--residualize needs results/model_tiers.json (run infer_tiers.py "
                     "on a non-residualized pass first)")
        tier_of = {m["model"]: m["tier"]
                   for m in json.loads(mt_path.read_text(encoding="utf-8"))["models"]}
        residual_tiers = np.array([tier_of.get(trajs[t]["model"], 2) for t in tids])
        print("evaluator: residualizing n_tool_errors / max_repeat_streak per model tier")
    diffs, e_scores = grade(metric_rows, residual_tiers=residual_tiers)

    # costs (cache-aware is the primary model; 'blind' ignores replay entirely)
    C = costs_matrix(metric_rows, cache_aware=True)
    C_blind = costs_matrix(metric_rows, cache_aware=False)

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
        print("predicting routing-time costs for lambda dispatch (OOF) ...")
        C_hat = oof_predicted_costs(feat_rows, metric_rows, groups)
        tiers = np.argmin(C_hat + DEFAULT_LAMBDA * p_over, axis=1) + 1
    else:  # blend (default)
        r_scores = blend_scores(cum, h_scores, DEFAULT_ALPHA)
        tiers = tiers_by_cuts(r_scores)
    tiers = np.asarray(tiers, dtype=int)

    report = compare(tiers, diffs, r_scores, e_scores, C)
    report["method"] = a.method
    report["residualized_evaluator"] = bool(a.residualize)
    report["cost_cache_blind"] = compare(tiers, diffs, r_scores, e_scores, C_blind)[
        "est_cost_policy_usd"]
    # tier-price sensitivity: same policy with Tier 3 priced at the fable-5 row
    C_fable = costs_matrix(metric_rows, cache_aware=True,
                           tier_prices={**TIER_PRICES, 3: TIER3_FABLE})
    rf = compare(tiers, diffs, r_scores, e_scores, C_fable)
    report["tier3_price_sensitivity"] = {
        "opus_priced_savings_pct": report["est_savings_pct"],
        "fable_priced_savings_pct": rf["est_savings_pct"],
        "note": "Tier 3 representative row: opus-5 [5, .5, 25] vs fable-5 [10, 1, 50] $/1M",
    }

    # ---- results ----
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with open(results / "router_features.jsonl", "w", encoding="utf-8") as f:
        for r in feat_rows:
            f.write(json.dumps({k: v for k, v in r.items()
                                if not k.startswith("_")}) + "\n")
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
    previews = {}
    for i, t in enumerate(tids):
        previews[str(t)] = [feat_rows[i]["_trigger"], feat_rows[i]["_preview"]]
        rows.append({
            "id": t,
            "model": trajs[t]["model"],          # display only — never a router input
            # verbatim text lives ONLY in the gitignored dashboard/previews.js
            "trigger": "",
            "preview": "",
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
            "residualized": bool(a.residualize),
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
    # verbatim previews: LOCAL USE ONLY (gitignored — dataset is no-redistribution)
    with open(dash / "previews.js", "w", encoding="utf-8") as f:
        f.write("window.VIKTOR_PREVIEWS = ")
        json.dump(previews, f, separators=(",", ":"))
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
          f"(cache-blind model: ${report['cost_cache_blind']}; chars/4 estimates)")
    print(f"tier-3 price sensitivity: savings {report['est_savings_pct']}% (opus-priced T3) "
          f"vs {rf['est_savings_pct']}% (fable-priced T3)")
    print("wrote results/ + dashboard/data.js (+ dashboard/previews.js, local only)")


if __name__ == "__main__":
    main()
