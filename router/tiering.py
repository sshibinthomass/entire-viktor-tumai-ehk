"""Unsupervised tier assignment from routing-time features.

Two methods, both label-free (no logged model, no trajectory outcome):

  score   — each feature is percentile-ranked across tasks (robust to the
            heavy-tailed token counts), group scores are the mean rank of the
            group's features, the composite is a signed weighted mean of group
            scores (signs are part of the routing insight — see
            features.FEATURE_GROUPS), and tiers are cut at configurable
            composite percentiles.
  kmeans  — k-means (k=3) on the same rank-space matrix with group weights
            applied, clusters ordered by mean composite score -> Tier 1/2/3.

Rank space keeps both methods on identical footing, so the dashboard can
re-run them client-side with different hyperparameters.
"""
import numpy as np

from .features import FEATURE_GROUPS

DEFAULT_GROUP_WEIGHTS = {"ask": 1.5, "harness": 0.7, "breadth": 0.8, "midthread": 0.8}
DEFAULT_CUTS = (0.55, 0.85)  # composite percentiles: <=c1 Tier1, <=c2 Tier2, else Tier3


def percentile_ranks(values):
    """Average-rank percentile in [0, 1]; constant columns rank 0."""
    v = np.asarray(values, dtype=float)
    order = v.argsort(kind="stable")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v))
    # average ties so binary flags don't split arbitrarily
    for u in np.unique(v):
        m = v == u
        ranks[m] = ranks[m].mean()
    if len(v) > 1 and v.max() > v.min():
        return ranks / (len(v) - 1)
    return np.zeros(len(v))


def rank_matrix(rows):
    """(matrix, feature_names): percentile ranks per scored feature, row order kept."""
    names = [f for g in FEATURE_GROUPS.values() for f in g["features"]]
    cols = [percentile_ranks([r[f] for r in rows]) for f in names]
    return np.column_stack(cols), names


def composite_scores(ranks, names, group_weights=None):
    """Signed weighted mean of group scores; each group is the mean rank of its
    features. The result is min-max normalized to [0, 1] for display."""
    gw = {**DEFAULT_GROUP_WEIGHTS, **(group_weights or {})}
    total_w = sum(gw.values())
    score = np.zeros(ranks.shape[0])
    for group, spec in FEATURE_GROUPS.items():
        idx = [names.index(f) for f in spec["features"]]
        score += spec["sign"] * gw[group] * ranks[:, idx].mean(axis=1)
    score /= total_w
    lo, hi = score.min(), score.max()
    return (score - lo) / (hi - lo) if hi > lo else score


def tiers_by_cuts(scores, cuts=DEFAULT_CUTS):
    """Cut the composite at dataset percentiles -> 1/2/3."""
    c1, c2 = np.quantile(scores, cuts[0]), np.quantile(scores, cuts[1])
    return np.where(scores <= c1, 1, np.where(scores <= c2, 2, 3)).astype(int)


def tiers_by_kmeans(ranks, names, scores, group_weights=None, seed=13):
    """k-means (k=3) on group-weighted rank space; clusters ordered by mean
    composite. Sign is irrelevant for clustering (a reflection preserves
    distances) — it only orders the clusters."""
    from sklearn.cluster import KMeans
    gw = {**DEFAULT_GROUP_WEIGHTS, **(group_weights or {})}
    w = np.array([gw[g] for g, spec in FEATURE_GROUPS.items()
                  for _ in spec["features"]])
    km = KMeans(n_clusters=3, n_init=10, random_state=seed)
    labels = km.fit_predict(ranks * w)
    order = np.argsort([scores[labels == c].mean() for c in range(3)])
    remap = {int(c): t + 1 for t, c in enumerate(order)}
    return np.array([remap[int(c)] for c in labels], dtype=int)


def route(rows, method="score", group_weights=None, cuts=DEFAULT_CUTS, seed=13):
    """rows: list of feature dicts -> (tiers, composite_scores)."""
    ranks, names = rank_matrix(rows)
    scores = composite_scores(ranks, names, group_weights)
    if method == "kmeans":
        tiers = tiers_by_kmeans(ranks, names, scores, group_weights, seed)
    else:
        tiers = tiers_by_cuts(scores, cuts)
    return tiers, scores
