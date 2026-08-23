#!/usr/bin/env python3
"""Honest validation of the tuned router.

exp_text.py picks its winner by looking at OOF scores — a maximum over a grid,
biased upward. This script re-runs the selection INSIDE each training fold
(inner GroupKFold picks the config over the FULL grid, not a shortlist), then
scores the picked config on the untouched outer test fold.

Dispatch is TRAIN-FOLD CALIBRATED: each outer fold's tier cuts come from that
fold's inner-OOF scores and TRAIN labels only, so no test-label information
places the cuts. Two numbers are reported:

  exact                  — deployable (train-calibrated cuts): THE headline
  exact_known_marginals  — cuts from the pooled label marginals, i.e. what the
                           same scores achieve if deployment marginals are
                           known; reference only, it is transductive

Selection bias is MEASURED, not asserted: the best pooled-grid config's OOF
exact (a max over the grid) minus the nested number.

Usage: python exp_final.py
"""
import json

import numpy as np
from scipy.sparse import hstack as sp_hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from experiments import (CACHE, build, rank_fold, cut_dispatch, cut_dispatch_oof,
                         show, align, metrics)

if not CACHE.exists():
    print(f"{CACHE} missing — building it first")
    build()
z = np.load(CACHE, allow_pickle=True)
X, D, e, groups = z["X"], z["D"], z["e"], z["groups"]
texts, sys_texts, tool_texts = list(z["texts"]), list(z["sys_texts"]), list(z["tool_texts"])
n = len(D)

WORD = dict(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True,
            strip_accents="unicode")
CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60000,
            sublinear_tf=True)

# the FULL grid the inner CV chooses from (channels x C x structure) — the old
# 8-config shortlist was pre-filtered by looking at OOF results on the same
# folds, which reintroduced selection bias
CHANNEL_SETS = [
    ("usr-word", [("U", WORD)]),
    ("usr-char", [("U", CHAR)]),
    ("usr-word+char", [("U", WORD), ("U", CHAR)]),
    ("usr+sys+tools-word", [("U", WORD), ("S", WORD), ("T", WORD)]),
    ("usr-word+char+sys/tools", [("U", WORD), ("U", CHAR), ("S", WORD), ("T", WORD)]),
]
CONFIGS = [{"name": f"{ch_name}/C={C}/{'ord' if ordinal else 'multi'}",
            "channels": ch, "C": C, "ordinal": ordinal}
           for ch_name, ch in CHANNEL_SETS
           for C in (0.5, 1.0, 2.0)
           for ordinal in (False, True)]

TXT = {"U": texts, "S": sys_texts, "T": tool_texts}


def fit_predict(cfg, tr, te):
    mats_tr, mats_te = [], []
    for key, kw in cfg["channels"]:
        vec = TfidfVectorizer(**kw)
        mats_tr.append(vec.fit_transform([TXT[key][i] for i in tr]))
        mats_te.append(vec.transform([TXT[key][i] for i in te]))
    R = rank_fold(X, tr)
    mats_tr.append(csr_matrix(R[tr]))
    mats_te.append(csr_matrix(R[te]))
    Ztr, Zte = sp_hstack(mats_tr).tocsr(), sp_hstack(mats_te).tocsr()
    if cfg.get("ordinal"):
        ytr = D[tr]
        p2 = LogisticRegression(max_iter=3000, C=cfg["C"]).fit(
            Ztr, (ytr >= 2).astype(int)).predict_proba(Zte)[:, 1]
        p3 = LogisticRegression(max_iter=3000, C=cfg["C"]).fit(
            Ztr, (ytr >= 3).astype(int)).predict_proba(Zte)[:, 1]
        p3 = np.minimum(p2, p3)
        return np.column_stack([1 - p2, p2 - p3, p3])
    m = LogisticRegression(max_iter=3000, C=cfg["C"]).fit(Ztr, D[tr])
    return align(m, Zte)


def score_of(P):
    return 1 - (P[:, 0] + (P[:, 0] + P[:, 1])) / 2


def main():
    folds = list(GroupKFold(5).split(np.zeros(n), groups=groups))
    P = np.zeros((n, 3))
    tiers = np.zeros(n, dtype=int)
    picks = []
    for fi, (tr, te) in enumerate(folds):
        inner = list(GroupKFold(4).split(np.zeros(len(tr)), groups=groups[tr]))
        best_cfg, best_acc, best_str = None, -1, None
        for cfg in CONFIGS:
            Pin = np.zeros((len(tr), 3))
            for itr, ite in inner:
                Pin[ite] = fit_predict(cfg, tr[itr], tr[ite])
            s = score_of(Pin)
            acc = (cut_dispatch_oof(s, D[tr], inner) == D[tr]).mean()
            if acc > best_acc:
                best_cfg, best_acc, best_str = cfg, acc, s
        picks.append(best_cfg["name"])
        print(f"fold {fi}: inner pick = {best_cfg['name']} "
              f"(inner deployable acc {best_acc:.1%})", flush=True)
        P[te] = fit_predict(best_cfg, tr, te)
        # dispatch cuts from TRAIN information only: inner-OOF scores + train labels
        s_te = score_of(P[te])
        p1, p2 = (D[tr] == 1).mean(), (D[tr] <= 2).mean()
        q1, q2 = np.quantile(best_str, p1), np.quantile(best_str, p2)
        tiers[te] = np.where(s_te <= q1, 1, np.where(s_te <= q2, 2, 3))

    s = score_of(P)
    m = show("\nFINAL (nested selection, deployable cuts)", tiers, D, e, s)
    m["exact_known_marginals"] = float((cut_dispatch(s, D) == D).mean())
    print(f"  reference (transductive, cuts from pooled label marginals): "
          f"{m['exact_known_marginals']:.1%} exact")
    print("confusion (rows router T1-3, cols observed D1-3):")
    for row in m["conf"]:
        print("   ", row)

    # ---- measured selection bias: best pooled-grid config vs the nested number
    print("\nmeasuring grid-selection bias (every config, pooled OOF) ...", flush=True)
    grid = {}
    for cfg in CONFIGS:
        Pg = np.zeros((n, 3))
        for tr, te in folds:
            Pg[te] = fit_predict(cfg, tr, te)
        sg = score_of(Pg)
        grid[cfg["name"]] = float((cut_dispatch_oof(sg, D, folds) == D).mean())
    grid_best = max(grid, key=grid.get)
    bias = grid[grid_best] - m["exact"]
    print(f"grid best (max over {len(CONFIGS)} configs, optimistic): "
          f"{grid_best} = {grid[grid_best]:.1%}  ->  selection bias "
          f"{bias * 100:+.1f} pts vs nested {m['exact']:.1%}")

    with open("results/final_validation.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": m, "fold_picks": picks,
                   "n_configs_in_grid": len(CONFIGS),
                   "grid_best": {"name": grid_best, "exact": grid[grid_best]},
                   "selection_bias_pts": round(bias * 100, 2),
                   "grid_exact_by_config": grid}, f, indent=2)
    np.savez("results/final_probs.npz", P=P, tiers=tiers)
    print("wrote results/final_validation.json + results/final_probs.npz")


if __name__ == "__main__":
    main()
