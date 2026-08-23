#!/usr/bin/env python3
"""The sweeps behind the shipped dispatch defaults (alpha / tau / lambda).

These sweeps previously lived in an uncommitted scratch session — the shipped
defaults (alpha=0.85, tau=0.80, lambda=0.3) were not reproducible from the
repo. This script recomputes them with the SAME OOF protocol as the pipeline
(router.ml.oof_cumulative_probs: GroupKFold on workspace, TF-IDF+ordinal head,
per-fold rank fitting) and writes every operating point to results/sweeps.json
so the deck and SOLUTION.md quote from an artifact, not from memory.

Usage: python sweep_defaults.py [export.jsonl]
"""
import argparse
import json

import numpy as np

from router.ml import oof_cumulative_probs, blend_scores, fold_ranks
from router.tiering import DEFAULT_CUTS
from tune_router import build_dataset, frozen_evaluator, costs_matrix, eval_tiers
from run_pipeline import DEFAULT_EXPORT, oof_predicted_costs


def op_point(tiers, D, C):
    cost_share, served, wserved = eval_tiers(tiers, D, C)
    return {"exact": float((tiers == D).mean()),
            "served": served, "served_weighted": wserved,
            "cost_share": cost_share,
            "savings_pct": round(100 * (1 - cost_share), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?", default=DEFAULT_EXPORT)
    a = ap.parse_args()

    print("building dataset ...", flush=True)
    tids, feats, mets, groups, texts = build_dataset(a.export)
    D, _ = frozen_evaluator(mets)
    C = costs_matrix(mets, cache_aware=True)
    print("OOF probabilities (pipeline head: word+char TF-IDF + numeric, ordinal) ...",
          flush=True)
    cum = oof_cumulative_probs(feats, D, groups, texts=texts)
    print("OOF routing-time cost predictions ...", flush=True)
    C_hat = oof_predicted_costs(feats, mets, np.asarray(groups))

    # heuristic score: the tiering composite on full-data ranks (the heuristic
    # is label-free, so 'OOF' only affects the rank fit; full-data ranks are
    # what the pipeline ships)
    from router.tiering import rank_matrix, composite_scores
    ranks, names = rank_matrix(feats)
    h = composite_scores(ranks, names)

    out = {"protocol": "OOF GroupKFold(workspace), frozen evaluator, cache-aware costs",
           "alpha": [], "tau": [], "lambda_pred": []}

    for alpha in np.round(np.arange(0.0, 1.001, 0.05), 2):
        s = blend_scores(cum, h, float(alpha))
        q1, q2 = np.quantile(s, DEFAULT_CUTS[0]), np.quantile(s, DEFAULT_CUTS[1])
        tiers = np.where(s <= q1, 1, np.where(s <= q2, 2, 3)).astype(int)
        out["alpha"].append({"alpha": float(alpha), **op_point(tiers, D, C)})

    for tau in np.round(np.arange(0.30, 0.991, 0.02), 3):
        tiers = np.where(cum[:, 0] >= tau, 1,
                         np.where(cum[:, 1] >= tau, 2, 3)).astype(int)
        out["tau"].append({"tau": float(tau), **op_point(tiers, D, C)})

    p_over = np.stack([1 - cum[:, 0], 1 - cum[:, 1], np.zeros(len(cum))], axis=1)
    for lam in np.round(np.geomspace(1e-3, 10, 25), 5):
        tiers = (np.argmin(C_hat + lam * p_over, axis=1) + 1).astype(int)
        out["lambda_pred"].append({"lambda": float(lam), **op_point(tiers, D, C)})

    best_a = max(out["alpha"], key=lambda r: r["exact"])
    print(f"alpha sweep: exact peaks at alpha={best_a['alpha']} ({best_a['exact']:.1%})")
    t80 = min(out["tau"], key=lambda r: abs(r["tau"] - 0.80))
    print(f"tau=0.80 operating point: served {t80['served']:.1%} at "
          f"{t80['cost_share']:.1%} cost (weighted {t80['served_weighted']:.1%})")

    with open("results/sweeps.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results/sweeps.json")


if __name__ == "__main__":
    main()
