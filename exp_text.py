#!/usr/bin/env python3
"""Text-model sweep for the router: vectorizers, channels, ordinal structure,
averaging. Same OOF protocol as experiments.py (GroupKFold on workspace,
marginal-matched cut dispatch). The winner here is then re-validated with
inner-CV selection in exp_final.py to keep the quoted number honest.

Usage: python exp_text.py
"""
import json

import numpy as np
from scipy.sparse import hstack as sp_hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from experiments import (CACHE, ALL_FEATS, rank_fold, cut_dispatch, show, align)
from router.ml import fold_ranks

z = np.load(CACHE, allow_pickle=True)
X, D, e, groups = z["X"], z["D"], z["e"], z["groups"]
texts, sys_texts, tool_texts = list(z["texts"]), list(z["sys_texts"]), list(z["tool_texts"])
n = len(D)
folds = list(GroupKFold(5).split(np.zeros(n), groups=groups))

WORD = dict(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True,
            strip_accents="unicode")
CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60000,
            sublinear_tf=True)


def sparse_fit(channels, C=1.0, ordinal=False, with_numeric=True):
    """channels: list of (texts, vectorizer-kwargs). Sparse TF-IDF + numeric
    ranks -> logistic (multinomial or cumulative-ordinal). Returns OOF probs."""
    P = np.zeros((n, 3))
    for tr, te in folds:
        mats_tr, mats_te = [], []
        for txts, kw in channels:
            vec = TfidfVectorizer(**kw)
            mats_tr.append(vec.fit_transform([txts[i] for i in tr]))
            mats_te.append(vec.transform([txts[i] for i in te]))
        if with_numeric:
            R = rank_fold(X, tr)
            mats_tr.append(csr_matrix(R[tr]))
            mats_te.append(csr_matrix(R[te]))
        Ztr, Zte = sp_hstack(mats_tr).tocsr(), sp_hstack(mats_te).tocsr()
        if ordinal:
            p2 = LogisticRegression(max_iter=3000, C=C).fit(
                Ztr, (D[tr] >= 2).astype(int)).predict_proba(Zte)[:, 1]
            p3 = LogisticRegression(max_iter=3000, C=C).fit(
                Ztr, (D[tr] >= 3).astype(int)).predict_proba(Zte)[:, 1]
            p3 = np.minimum(p2, p3)
            P[te] = np.column_stack([1 - p2, p2 - p3, p3])
        else:
            m = LogisticRegression(max_iter=3000, C=C).fit(Ztr, D[tr])
            P[te] = align(m, Zte)
    return P


def score_of(P):
    return 1 - (P[:, 0] + (P[:, 0] + P[:, 1])) / 2


def run(name, P):
    s = score_of(P)
    m = show(name, cut_dispatch(s, D), D, e, s)
    return P, m


results = {}
U, S, T = texts, sys_texts, tool_texts
print(f"n={n}  reference: tfidf-svd80+lr was 61.9% exact\n")

results["word-svd(ref-like) usr"] = run("word sparse, usr only, C=1",
                                        sparse_fit([(U, WORD)]))
results["char usr"] = run("char_wb(3-5) sparse, usr only", sparse_fit([(U, CHAR)]))
results["word+char usr"] = run("word+char, usr", sparse_fit([(U, WORD), (U, CHAR)]))
results["word usr+sys+tools"] = run("word, usr+sys+tools",
                                    sparse_fit([(U, WORD), (S, WORD), (T, WORD)]))
results["word+char usr + word sys/tools"] = run(
    "word+char usr + word sys+tools",
    sparse_fit([(U, WORD), (U, CHAR), (S, WORD), (T, WORD)]))
best_ch = max(results, key=lambda k: results[k][1]["exact"])
print(f"\nbest channel set: {best_ch} -> C / ordinal sweep on it:")

CH = {"word-svd(ref-like) usr": [(U, WORD)], "char usr": [(U, CHAR)],
      "word+char usr": [(U, WORD), (U, CHAR)],
      "word usr+sys+tools": [(U, WORD), (S, WORD), (T, WORD)],
      "word+char usr + word sys/tools": [(U, WORD), (U, CHAR), (S, WORD), (T, WORD)]}[best_ch]
for C in (0.5, 2.0):
    results[f"best C={C}"] = run(f"  C={C}", sparse_fit(CH, C=C))
results["best ordinal"] = run("  ordinal cumulative, C=1", sparse_fit(CH, ordinal=True))
results["best no-numeric"] = run("  text only (no numeric)", sparse_fit(CH, with_numeric=False))

# averaging with the numeric ordlog probs from experiments.py
pz = np.load("results/exp_probs.npz")
P_ord = pz["ordlog_v2-feats"] if "ordlog_v2-feats" in pz.files else pz[pz.files[0]]
bestP = max(results.values(), key=lambda v: v[1]["exact"])[0]
for w in (0.25, 0.5):
    run(f"avg: {1-w:.2f}*text + {w:.2f}*ordlog", (1 - w) * bestP + w * P_ord)

np.savez("results/exp_text_probs.npz", best=bestP)
with open("results/exp_text_report.json", "w", encoding="utf-8") as f:
    json.dump({k: v[1] for k, v in results.items()}, f, indent=2)
print("\nwrote results/exp_text_probs.npz + results/exp_text_report.json")
