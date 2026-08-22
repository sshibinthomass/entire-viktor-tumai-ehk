#!/usr/bin/env python3
"""Fit the demand model and save a deployable artifact.

This is the only script that looks at outcomes. It learns two things:

    1. WHAT KINDS of task exist - k archetypes clustered from the first user
       message text. No outcome data is used for this.
    2. WHICH archetypes are heavy - the ordering of the k clusters, taken from
       median observed effort. This IS outcome data: exactly k numbers.

The result is saved to a pickle that router.py loads. router.py then needs no
outcomes at all, which is the point of the split: a deployed router cannot know
how much a task will write before it writes it.

TWO FITTING MODES, AND THE DIFFERENCE MATTERS
    save_artifact()  fits on ALL rows. Correct for deployment - you want every
                     row you have informing the model you ship.
    crossfit_bands() refits per fold under GroupKFold on workspace. Correct for
                     REPORTING - it is the only way to state a number that
                     describes an unseen tenant. evaluate.py uses this.
    Never quote a metric produced by the first mode.

Usage:
    python scripts/fit_router.py
    python scripts/fit_router.py -k 16 --out results/router_model.pkl
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

K = 16                 # archetypes; picked over {3,5,8,12,16,20,24}
SYNTHETIC_MAX_ID = 25  # trajectories_v1_00.jsonl is make_synthetic_sample.py filler
ARTIFACT = "results/router_model.pkl"


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_rows(features, keep_synthetic=False):
    rows = load(features)
    if not keep_synthetic:
        rows = [r for r in rows if r["trajectory_id"] > SYNTHETIC_MAX_ID]
    return rows


class TriggerSpace:
    """Vectorise the first user message. Nothing else feeds the demand score.

    Measured: the system-prompt skill list HURTS this (user text alone gives
    Spearman +0.567 against observed effort, adding skills at half weight drops
    it to +0.554, at equal weight +0.368) because the skill set is close to a
    tenant id. The tool list never helped either. So neither is here."""

    def __init__(self, n_dims=80, seed=0):
        self.n_dims, self.seed = n_dims, seed

    def fit(self, rows):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=30000,
                                   sublinear_tf=True, strip_accents="unicode")
        T = self.vec.fit_transform([r["_trigger_text"] for r in rows])
        self.svd = TruncatedSVD(min(self.n_dims, T.shape[1] - 1),
                                random_state=self.seed).fit(T)
        return self

    def transform(self, rows):
        T = self.vec.transform([r["_trigger_text"] for r in rows])
        return normalize(self.svd.transform(T))


def effort_ranks(values):
    """Rank transform, so the ordering is scale-free."""
    return np.argsort(np.argsort(values)).astype(float)


def rank_clusters(labels, effort, k):
    """The k numbers of outcome data. Cluster -> 0..k-1, lightest first."""
    med = {}
    for c in range(k):
        m = labels == c
        med[c] = float(np.median(effort[m])) if m.any() else float(np.median(effort))
    return {c: i for i, c in enumerate(sorted(med, key=lambda c: med[c]))}


def display_band(rank, k):
    """Cluster rank -> a 1..10 number for humans to read.

    DISPLAY ONLY. The router maps cluster RANK to tier directly. Binning 16
    ranks into 10 bands collapsed 6 pairs of clusters into shared bands and made
    the distribution lumpy (one band held 23% of all tasks), so the band is no
    longer in the decision path."""
    return int(np.clip(np.floor(rank / (k - 1) * 9) + 1, 1, 10))


def name_clusters(labels, k, vec, T):
    """Name each cluster by the trigger terms most over-represented inside it."""
    terms = np.array(vec.get_feature_names_out())
    names = {}
    for c in range(k):
        m = labels == c
        if not m.any():
            names[c] = "(empty)"
            continue
        lift = np.asarray(T[m].mean(axis=0)).ravel() - np.asarray(T[~m].mean(axis=0)).ravel()
        names[c] = ", ".join(t for t in terms[np.argsort(-lift)[:7]] if len(t) > 2)[:64]
    return names


def crossfit_ranks(rows, groups, effort, k=K, folds=5, seed=0):
    """Out-of-fold CLUSTER RANKS (0..k-1), for REPORTING only.

    Everything - vectoriser, SVD, clustering, and the cluster ordering - is
    refitted on the training folds and applied to the held-out fold, so no row
    is ever scored using its own outcome. GroupKFold on the workspace
    fingerprint because most workspaces are single-model; a random split lets
    tenant identity leak in and inflates the result (+0.74 vs +0.57)."""
    oof = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(np.zeros(len(rows)), groups=groups):
        tr_rows = [rows[i] for i in tr]
        te_rows = [rows[i] for i in te]
        space = TriggerSpace(seed=seed).fit(tr_rows)
        km = KMeans(k, n_init=15, random_state=seed).fit(embed(tr_rows, space.vec, space.svd))
        rank = rank_clusters(km.labels_, effort[tr], k)
        oof[te] = [rank[c] for c in km.predict(embed(te_rows, space.vec, space.svd))]
    return oof.astype(int)


def embed(rows, vec, svd):
    """Trigger text -> unit vectors. Module-level so the artifact can hold plain
    sklearn objects; a wrapper class pickled from __main__ is unloadable from a
    different entry point."""
    return normalize(svd.transform(vec.transform([r["_trigger_text"] for r in rows])))


def fit_all(rows, effort, k=K, seed=0):
    """Fit on every row. For the artifact you deploy, not for reporting."""
    space = TriggerSpace(seed=seed).fit(rows)
    Z = embed(rows, space.vec, space.svd)
    km = KMeans(k, n_init=25, random_state=seed).fit(Z)
    rank = rank_clusters(km.labels_, effort, k)
    T = space.vec.transform([r["_trigger_text"] for r in rows])
    return {"vec": space.vec, "svd": space.svd, "kmeans": km, "rank": rank, "k": k,
            "names": name_clusters(km.labels_, k, space.vec, T),
            "n_fit_rows": len(rows)}


def save(artifact, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)


def load_artifact(path=ARTIFACT):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.jsonl")
    ap.add_argument("--out", default=ARTIFACT)
    ap.add_argument("-k", type=int, default=K)
    ap.add_argument("--keep-synthetic", action="store_true")
    a = ap.parse_args()

    rows = load_rows(a.features, a.keep_synthetic)
    gen = np.array([r["gen_tokens"] for r in rows], float)
    art = fit_all(rows, effort_ranks(gen), a.k)
    save(art, a.out)

    labels = art["kmeans"].labels_
    print("fitted on {} tasks, k={}".format(len(rows), a.k))
    print("\n{:>3s} {:>5s} {:>5s} {:>9s}  {}".format("c", "n", "band", "med gen", "terms"))
    for c in sorted(range(a.k), key=lambda c: -art["rank"][c]):
        m = labels == c
        if not m.any():
            continue
        print("{:3d} {:5d} {:5d} {:5d} {:9.0f}  {}".format(
            c, int(m.sum()), art["rank"][c], display_band(art["rank"][c], a.k),
            float(np.median(gen[m])), art["names"][c]))
    print("\nwrote {}".format(a.out))
    print("NOTE: this artifact is fitted on ALL rows and is for DEPLOYMENT.")
    print("      Metrics must come from evaluate.py, which cross-fits.")


if __name__ == "__main__":
    main()
