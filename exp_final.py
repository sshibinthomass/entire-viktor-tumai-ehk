#!/usr/bin/env python3
"""Honest validation of the tuned router.

exp_text.py picks its winner by looking at OOF scores — a maximum over a grid,
biased upward. This script re-runs the selection INSIDE each training fold
(inner GroupKFold picks the config), then scores the picked config on the
untouched outer test fold. The resulting number contains no grid-selection
optimism and is the one to quote.

Also prints the final confusion matrix and the per-class recalls.

Usage: python exp_final.py
"""
import json

import numpy as np
from scipy.sparse import hstack as sp_hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from experiments import CACHE, rank_fold, cut_dispatch, show, align, metrics

z = np.load(CACHE, allow_pickle=True)
X, D, e, groups = z["X"], z["D"], z["e"], z["groups"]
texts, sys_texts, tool_texts = list(z["texts"]), list(z["sys_texts"]), list(z["tool_texts"])
n = len(D)

WORD = dict(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True,
            strip_accents="unicode")
CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60000,
            sublinear_tf=True)

# the small config set the inner CV chooses from (channels x C x structure)
CONFIGS = []
for ch_name, ch in [("usr-word", [("U", WORD)]),
                    ("usr-word+char", [("U", WORD), ("U", CHAR)])]:
    for C in (0.5, 1.0):
        for ordinal in (False, True):
            CONFIGS.append({"name": f"{ch_name}/C={C}/{'ord' if ordinal else 'multi'}",
                            "channels": ch, "C": C, "ordinal": ordinal})

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


def main():
    folds = list(GroupKFold(5).split(np.zeros(n), groups=groups))
    P = np.zeros((n, 3))
    picks = []
    for fi, (tr, te) in enumerate(folds):
        inner = list(GroupKFold(4).split(np.zeros(len(tr)), groups=groups[tr]))
        best_cfg, best_acc = None, -1
        for cfg in CONFIGS:
            Pin = np.zeros((len(tr), 3))
            for itr, ite in inner:
                Pin[ite] = fit_predict(cfg, tr[itr], tr[ite])
            s = 1 - (Pin[:, 0] + (Pin[:, 0] + Pin[:, 1])) / 2
            acc = (cut_dispatch(s, D[tr]) == D[tr]).mean()
            if acc > best_acc:
                best_cfg, best_acc = cfg, acc
        picks.append(best_cfg["name"])
        print(f"fold {fi}: inner pick = {best_cfg['name']} (inner acc {best_acc:.1%})")
        P[te] = fit_predict(best_cfg, tr, te)

    s = 1 - (P[:, 0] + (P[:, 0] + P[:, 1])) / 2
    tiers = cut_dispatch(s, D)
    m = show("\nFINAL (nested selection, honest)", tiers, D, e, s)
    print("confusion (rows router T1-3, cols observed D1-3):")
    for row in m["conf"]:
        print("   ", row)
    with open("results/final_validation.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": m, "fold_picks": picks}, f, indent=2)
    np.savez("results/final_probs.npz", P=P)
    print("wrote results/final_validation.json + results/final_probs.npz")


if __name__ == "__main__":
    main()
