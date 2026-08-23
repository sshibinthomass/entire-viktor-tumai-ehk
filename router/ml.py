"""Supervised probabilistic router head.

Ordinal logistic regression on the routing-time rank features, trained on the
EVALUATOR's difficulty labels (never the logged model id). Produces cumulative
sufficiency probabilities per task:

    cum1 = P(difficulty <= 1),  cum2 = P(difficulty <= 2)

which the dispatch layer turns into tiers three ways (all benchmarked in
tune_router.py):

    cuts on a blended score  - alpha * rank(ML difficulty) + (1-alpha) * heuristic
                               (balanced winner: improves both per-task AND
                               token-weighted frontiers over the heuristic)
    tau-sufficiency          - cheapest tier with cum >= tau (size-blind;
                               cheapest way to 90% served)
    lambda-Bayes             - argmin_t cost$(t) + lambda * P(D > t)
                               (best tasks-per-dollar; sacrifices heavy tasks)

Honesty: probabilities shipped to results/dashboard are OUT-OF-FOLD
(GroupKFold on the workspace fingerprint), so every task's tier comes from a
model that never saw that task's outcome.
"""
import hashlib
import json
import re

import numpy as np

from .features import FEATURE_GROUPS

FEAT_NAMES = [f for g in FEATURE_GROUPS.values() for f in g["features"]]


def workspace_of(req):
    """Tenant fingerprint (skill set of the system prompt) for grouped CV."""
    sysp = next((i.get("content") for i in req["input"] if i.get("role") == "system"), "") or ""
    if not isinstance(sysp, str):
        sysp = json.dumps(sysp)
    skills = ",".join(sorted(re.findall(r"^\-\s\*\*([a-zA-Z0-9_\- ]+)\*\*", sysp, re.M)))
    return hashlib.md5(skills.encode()).hexdigest()[:8]


def fold_ranks(train_vals, all_vals):
    s = np.sort(np.asarray(train_vals, dtype=float))
    return np.searchsorted(s, np.asarray(all_vals, dtype=float), side="right") / max(len(s), 1)


def _rank_matrix(feat_rows, tr_idx):
    out = np.zeros((len(feat_rows), len(FEAT_NAMES)))
    for j, f in enumerate(FEAT_NAMES):
        vals = [r[f] for r in feat_rows]
        out[:, j] = fold_ranks([vals[i] for i in tr_idx], vals)
    return out


def _fit_ordinal(Xtr, ytr, Xte):
    """Two cumulative binary logits -> class probs, monotone-corrected."""
    from sklearn.linear_model import LogisticRegression
    p2 = LogisticRegression(max_iter=2000).fit(Xtr, (ytr >= 2).astype(int)) \
        .predict_proba(Xte)[:, 1]
    p3 = LogisticRegression(max_iter=2000).fit(Xtr, (ytr >= 3).astype(int)) \
        .predict_proba(Xte)[:, 1]
    p3 = np.minimum(p2, p3)
    return np.column_stack([1 - p2, p2 - p3, p3])


def oof_cumulative_probs(feat_rows, labels, groups, n_splits=5, seed=13):
    """(n, 2) out-of-fold [P(D<=1), P(D<=2)] via GroupKFold on workspace."""
    from sklearn.model_selection import GroupKFold
    n = len(feat_rows)
    labels = np.asarray(labels)
    cum = np.zeros((n, 2))
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(n), groups=groups):
        R = _rank_matrix(feat_rows, tr)
        p = _fit_ordinal(R[tr], labels[tr], R[te])
        cum[te, 0] = p[:, 0]
        cum[te, 1] = p[:, 0] + p[:, 1]
    return cum


def blend_scores(cum, heuristic_scores, alpha=0.5):
    """Balanced router score: alpha * rank(ML difficulty) + (1-alpha) * rank(heuristic)."""
    ml_difficulty = 1 - (cum[:, 0] + cum[:, 1]) / 2
    r_ml = fold_ranks(ml_difficulty, ml_difficulty)
    r_h = fold_ranks(heuristic_scores, heuristic_scores)
    return alpha * r_ml + (1 - alpha) * r_h
