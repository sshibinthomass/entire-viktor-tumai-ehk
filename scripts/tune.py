#!/usr/bin/env python3
"""Tune the demand model's hyperparameters, and measure the selection bias.

WHY THIS SCRIPT IS SUSPICIOUS OF ITSELF
    The objective is Spearman(cluster rank, observed generated tokens). That
    "observed effort" is a PROXY for difficulty, not difficulty. Tuning hard
    against a proxy buys you a model that fits the proxy's quirks, and the
    reported score then flatters itself twice: once from tuning, once from
    reporting on the same folds.

    So this script does two passes:
      SEARCH   grid over K / SVD dims / TF-IDF, scored by cross-fitted rho.
               The winner's rho here is OPTIMISTIC - it is a maximum over the
               grid, and a maximum of noisy estimates is biased upward.
      NESTED   outer GroupKFold; inside each outer fold the grid is re-searched
               on the training folds only and the winner scored on the held-out
               fold. That number is honest and is the one to quote.
    The gap between them is the selection bias, printed explicitly. If the gap
    is large, the tuning found noise.

    Grouping is GroupKFold on the workspace fingerprint throughout. Most
    workspaces are single-model, so a random split lets tenant identity leak and
    inflates everything (+0.74 vs +0.52 on the same data).

Usage:
    python scripts/tune.py                # search + nested check
    python scripts/tune.py --stage kdim   # only the K x SVD grid
"""
import argparse
import itertools

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from fit_router import effort_ranks, load_rows, rank_clusters

# Kept small on purpose: every extra axis widens the maximum-over-grid bias.
GRID_KDIM = {"k": [8, 12, 16, 20, 24, 32], "dims": [40, 80, 120]}
GRID_TFIDF = {"ngram": [(1, 1), (1, 2)], "min_df": [3, 5, 10],
              "sublinear": [True, False]}
BASE = {"k": 16, "dims": 80, "ngram": (1, 2), "min_df": 5, "sublinear": True}


def build(train_rows, cfg):
    vec = TfidfVectorizer(ngram_range=cfg["ngram"], min_df=cfg["min_df"],
                          max_features=30000, sublinear_tf=cfg["sublinear"],
                          strip_accents="unicode")
    T = vec.fit_transform([r["_trigger_text"] for r in train_rows])
    dims = min(cfg["dims"], T.shape[1] - 1)
    svd = TruncatedSVD(max(2, dims), random_state=0).fit(T)
    return vec, svd


def embed_with(rows, vec, svd):
    return normalize(svd.transform(vec.transform([r["_trigger_text"] for r in rows])))


def crossfit_rho(rows, groups, effort, gen, cfg, folds=5, seed=0):
    """Cross-fitted rank -> Spearman against observed effort. Everything is
    refitted per fold, including the vectoriser, so nothing leaks."""
    oof = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(np.zeros(len(rows)), groups=groups):
        tr_rows = [rows[i] for i in tr]
        te_rows = [rows[i] for i in te]
        vec, svd = build(tr_rows, cfg)
        km = KMeans(cfg["k"], n_init=10, random_state=seed).fit(embed_with(tr_rows, vec, svd))
        rk = rank_clusters(km.labels_, effort[tr], cfg["k"])
        oof[te] = [rk[c] for c in km.predict(embed_with(te_rows, vec, svd))]
    return spearmanr(oof, gen).statistic, oof


def search(rows, groups, effort, gen, grid, base, folds=5, quiet=False):
    keys = list(grid)
    best, results = None, []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        rho, _ = crossfit_rho(rows, groups, effort, gen, cfg, folds)
        results.append((rho, cfg))
        if best is None or rho > best[0]:
            best = (rho, cfg)
        if not quiet:
            print("   rho={:+.4f}  {}".format(
                rho, "  ".join("{}={}".format(k, cfg[k]) for k in keys)))
    return best, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.jsonl")
    ap.add_argument("--stage", choices=["kdim", "tfidf", "all"], default="all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nested", action="store_true", default=True)
    a = ap.parse_args()

    rows = load_rows(a.features)
    gen = np.array([r["gen_tokens"] for r in rows], float)
    groups = np.array([r["workspace"] for r in rows])
    effort = effort_ranks(gen)
    print("tuning on {} tasks, {} workspaces".format(len(rows), len(set(groups))))

    base_rho, _ = crossfit_rho(rows, groups, effort, gen, BASE, a.folds)
    print("shipped config rho = {:+.4f}   {}".format(
        base_rho, "  ".join("{}={}".format(k, v) for k, v in BASE.items())))

    cfg = dict(BASE)
    if a.stage in ("kdim", "all"):
        print("\n=== STAGE 1: K x SVD dims ===")
        best, _ = search(rows, groups, effort, gen, GRID_KDIM, cfg, a.folds)
        cfg = best[1]
        print("   -> best rho={:+.4f} at k={} dims={}".format(best[0], cfg["k"], cfg["dims"]))
    if a.stage in ("tfidf", "all"):
        print("\n=== STAGE 2: TF-IDF, holding K and dims ===")
        best, _ = search(rows, groups, effort, gen, GRID_TFIDF, cfg, a.folds)
        cfg = best[1]
        print("   -> best rho={:+.4f} at ngram={} min_df={} sublinear={}".format(
            best[0], cfg["ngram"], cfg["min_df"], cfg["sublinear"]))

    tuned_rho, _ = crossfit_rho(rows, groups, effort, gen, cfg, a.folds)
    print("\n=== TUNED CONFIG ===")
    print("   {}".format("  ".join("{}={}".format(k, v) for k, v in cfg.items())))
    print("   tuned rho   = {:+.4f}   (OPTIMISTIC - max over the grid)".format(tuned_rho))
    print("   shipped rho = {:+.4f}".format(base_rho))
    print("   apparent gain = {:+.4f}".format(tuned_rho - base_rho))

    if a.nested:
        print("\n=== NESTED CHECK (the honest number) ===")
        print("   outer fold: grid re-searched on train folds only, scored on held-out")
        outer = GroupKFold(n_splits=3)
        picks, scores, base_scores = [], [], []
        for i, (tr, te) in enumerate(outer.split(np.zeros(len(rows)), groups=groups)):
            tr_rows = [rows[j] for j in tr]
            te_rows = [rows[j] for j in te]
            g_tr = groups[tr]
            # inner search on the training portion only
            small = {"k": GRID_KDIM["k"], "dims": [cfg["dims"]]}
            best, _ = search(tr_rows, g_tr, effort_ranks(gen[tr]), gen[tr],
                            small, cfg, folds=3, quiet=True)
            inner_cfg = best[1]
            picks.append(inner_cfg["k"])
            # score the inner winner on the untouched outer fold
            for name, c, bucket in (("tuned", inner_cfg, scores),
                                    ("shipped", BASE, base_scores)):
                vec, svd = build(tr_rows, c)
                km = KMeans(c["k"], n_init=10, random_state=0).fit(
                    embed_with(tr_rows, vec, svd))
                rk = rank_clusters(km.labels_, effort_ranks(gen[tr]), c["k"])
                pred = [rk[q] for q in km.predict(embed_with(te_rows, vec, svd))]
                bucket.append(spearmanr(pred, gen[te]).statistic)
            print("   outer fold {}: inner picked k={:<3d} -> held-out rho tuned {:+.4f} "
                  "vs shipped {:+.4f}".format(i + 1, inner_cfg["k"], scores[-1], base_scores[-1]))
        print("   nested mean: tuned {:+.4f}   shipped {:+.4f}   real gain {:+.4f}"
              .format(np.mean(scores), np.mean(base_scores),
                      np.mean(scores) - np.mean(base_scores)))
        print("   inner picks for k were {} - unstable picks mean the grid is fitting noise"
              .format(picks))
        print("   SELECTION BIAS = tuned rho - nested tuned rho = {:+.4f}"
              .format(tuned_rho - np.mean(scores)))


if __name__ == "__main__":
    main()
