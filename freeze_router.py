#!/usr/bin/env python3
"""Fit the router on one chunk and FREEZE every learned transform to disk.

The interactive pipeline (run_pipeline.py) is transductive: percentile ranks,
blend ranks and tier cuts are recomputed on whatever batch it routes. That is
fine for analysis but it is not a router — a task's tier would depend on its
batch, and pointing the pipeline at a new chunk silently refits everything.

This script produces the deployable artifact instead. It fits on the given
(enriched) chunk and serializes:

  - per-feature ECDFs (sorted train values) for the numeric rank transform
  - the heuristic group weights and FIXED composite cut values
  - the word+char TF-IDF vectorizers and the two cumulative ordinal logits
  - the blend alpha, fixed blend-score cut values, tau and lambda defaults
  - routing-time cost regressors (for lambda dispatch) and tier prices
  - the FROZEN EVALUATOR (metric ECDFs, weights, fixed score cuts, overrides)
    so a held-out chunk is graded by chunk-01's yardstick, not refit on itself

apply_frozen.py loads the artifact and routes a new chunk — or ONE task —
without refitting anything.

The pickle contains TF-IDF vocabulary (dataset-derived text) — it is
gitignored on purpose; regenerate it locally with this script.

Usage: python freeze_router.py [export.jsonl] [--out results/frozen_router.pkl]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack as sp_hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from router.features import extract as extract_router, FEATURE_GROUPS
from router.ml import (FEAT_NAMES, WORD_TFIDF, CHAR_TFIDF, first_user_text)
from router.tiering import DEFAULT_GROUP_WEIGHTS, DEFAULT_CUTS
from evaluator.metrics import trajectory_metrics
from evaluator.difficulty import (grade, DEFAULT_WEIGHTS as EV_WEIGHTS,
                                  DEFAULT_CUTS as EV_CUTS, DEFAULT_OVERRIDES)
from run_pipeline import (load_trajectories, TIER_PRICES,
                          DEFAULT_ALPHA, DEFAULT_TAU, DEFAULT_LAMBDA, DEFAULT_EXPORT)


def ecdf_store(values):
    v = np.sort(np.asarray(values, dtype=float))
    return v


def ecdf_apply(store, x):
    if len(store) == 0 or store[0] == store[-1]:
        return np.full(np.atleast_1d(x).shape, 0.5)
    return np.searchsorted(store, np.asarray(x, dtype=float), side="right") / len(store)


def heuristic_score(rank_cols, weights):
    total = sum(weights.values())
    s = np.zeros(len(next(iter(rank_cols.values()))))
    for g, spec in FEATURE_GROUPS.items():
        cols = np.column_stack([rank_cols[f] for f in spec["features"]])
        s += spec["sign"] * weights[g] * cols.mean(axis=1)
    return s / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?", default=DEFAULT_EXPORT)
    ap.add_argument("--out", default="results/frozen_router.pkl")
    a = ap.parse_args()

    trajs = load_trajectories(a.export)
    tids = sorted(trajs)
    print(f"fitting frozen router on {len(tids)} trajectories from {Path(a.export).name}")
    feats = [extract_router(trajs[t]["first"]) for t in tids]
    mets = [trajectory_metrics(trajs[t]["deepest"], trajs[t]["n_calls"]) for t in tids]
    texts = [first_user_text(trajs[t]["first"]) for t in tids]

    # frozen evaluator (chunk-01 yardstick)
    D, e_scores = grade(mets)
    ev_ecdfs = {k: ecdf_store([m[k] for m in mets]) for k in EV_WEIGHTS}
    ev_cut_vals = [float(np.quantile(e_scores, EV_CUTS[0])),
                   float(np.quantile(e_scores, EV_CUTS[1]))]

    # numeric feature ECDFs + heuristic
    feat_ecdfs = {f: ecdf_store([r[f] for r in feats]) for f in FEAT_NAMES}
    rank_cols = {f: ecdf_apply(feat_ecdfs[f], [r[f] for r in feats]) for f in FEAT_NAMES}
    h = heuristic_score(rank_cols, DEFAULT_GROUP_WEIGHTS)
    h_cut_vals = [float(np.quantile(h, DEFAULT_CUTS[0])),
                  float(np.quantile(h, DEFAULT_CUTS[1]))]

    # ML head: word+char TF-IDF + numeric ranks -> two cumulative logits
    print("fitting TF-IDF + ordinal head on the full chunk ...")
    R = np.column_stack([rank_cols[f] for f in FEAT_NAMES])
    vecs, mats = [], []
    for kw in (WORD_TFIDF, CHAR_TFIDF):
        vec = TfidfVectorizer(**kw)
        mats.append(vec.fit_transform(texts))
        vecs.append(vec)
    Z = sp_hstack(mats + [csr_matrix(R)]).tocsr()
    lr2 = LogisticRegression(max_iter=3000).fit(Z, (D >= 2).astype(int))
    lr3 = LogisticRegression(max_iter=3000).fit(Z, (D >= 3).astype(int))
    p2 = lr2.predict_proba(Z)[:, 1]
    p3 = np.minimum(p2, lr3.predict_proba(Z)[:, 1])
    ml_difficulty = 1 - ((1 - p2) + (1 - p3)) / 2

    # blend: fixed rank ECDFs for both components + fixed blend cuts
    ml_ecdf, h_ecdf = ecdf_store(ml_difficulty), ecdf_store(h)
    blend = DEFAULT_ALPHA * ecdf_apply(ml_ecdf, ml_difficulty) \
        + (1 - DEFAULT_ALPHA) * ecdf_apply(h_ecdf, h)
    blend_cut_vals = [float(np.quantile(blend, DEFAULT_CUTS[0])),
                      float(np.quantile(blend, DEFAULT_CUTS[1]))]

    # routing-time cost regressors for lambda dispatch
    print("fitting routing-time cost regressors ...")
    regs = {}
    for k in ("context_tokens", "gen_tokens", "n_llm_calls"):
        reg = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                            learning_rate=0.06, random_state=13)
        reg.fit(R, np.log1p([m[k] for m in mets]))
        regs[k] = reg

    artifact = {
        "fit_source": Path(a.export).name,
        "n_fit": len(tids),
        "feat_names": FEAT_NAMES,
        "feat_ecdfs": feat_ecdfs,
        "group_weights": DEFAULT_GROUP_WEIGHTS,
        "h_cut_vals": h_cut_vals,
        "tfidf_vecs": vecs,
        "lr2": lr2, "lr3": lr3,
        "ml_ecdf": ml_ecdf, "h_ecdf": h_ecdf,
        "alpha": DEFAULT_ALPHA, "blend_cut_vals": blend_cut_vals,
        "tau": DEFAULT_TAU, "lambda": DEFAULT_LAMBDA,
        "cost_regs": regs, "tier_prices": TIER_PRICES,
        "evaluator": {"weights": EV_WEIGHTS, "ecdfs": ev_ecdfs,
                      "cut_vals": ev_cut_vals, "overrides": DEFAULT_OVERRIDES},
    }
    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) — gitignored: it embeds "
          f"TF-IDF vocabulary derived from the licensed dataset")


if __name__ == "__main__":
    main()
