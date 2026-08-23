"""Supervised probabilistic router head.

Winner of the method ladder (experiments.py / exp_text.py, validated with
nested selection in exp_final.py): ORDINAL cumulative logistic regression on
sparse word(1-2) + char_wb(3-5) TF-IDF of the first user message, concatenated
with the numeric rank features. Trained on the EVALUATOR's difficulty labels
(never the logged model id). Produces cumulative sufficiency probabilities:

    cum1 = P(difficulty <= 1),  cum2 = P(difficulty <= 2)

which the dispatch layer turns into tiers three ways (benchmarked in
tune_router.py):

    cuts on a blended score  - alpha * rank(ML difficulty) + (1-alpha) * heuristic
    tau-sufficiency          - cheapest tier with cum >= tau (size-blind;
                               cheapest way to 90% served)
    lambda-Bayes             - argmin_t cost$(t) + lambda * P(D > t)
                               (best tasks-per-dollar; sacrifices heavy tasks)

Honesty: probabilities shipped to results/dashboard are OUT-OF-FOLD
(GroupKFold on the workspace fingerprint), so every task's tier comes from a
model that never saw that task's outcome.
"""
import hashlib
import re

import numpy as np

from rank_utils import ecdf_ranks
from .features import FEATURE_GROUPS, ML_EXTRA_FEATURES, PII_RE, _content

FEAT_NAMES = ([f for g in FEATURE_GROUPS.values() for f in g["features"]]
              + ML_EXTRA_FEATURES)

WORD_TFIDF = dict(ngram_range=(1, 2), min_df=2, max_features=40000,
                  sublinear_tf=True, strip_accents="unicode")
CHAR_TFIDF = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                  max_features=60000, sublinear_tf=True)


def first_user_text(req, limit=12000):
    """PII-collapsed text of the first user message (routing-time input)."""
    fu = next((i for i in req["input"] if i.get("role") == "user"), None)
    c = (fu or {}).get("content")
    txt = c if isinstance(c, str) else "\n".join(
        p.get("text", "") for p in (c or []) if isinstance(p, dict))
    return PII_RE.sub("<E>", txt)[:limit]


def workspace_of(req):
    """Tenant fingerprint (skill set of the system prompt) for grouped CV.

    The system content can be a plain string OR a parts list; joining the parts
    (same logic as the feature extractor) keeps the fingerprint identical across
    both formats. The old version json.dumps'd parts lists, escaped the
    newlines, matched nothing, and collapsed 76% of trajectories into one
    md5("") group — degenerating the GroupKFold folds."""
    sysp = next((_content(i) for i in req["input"] if i.get("role") == "system"), "")
    skills = ",".join(sorted(re.findall(r"^\-\s\*\*([a-zA-Z0-9_\- ]+)\*\*", sysp, re.M)))
    if not skills:  # no skill block: fall back to the system text itself
        skills = sysp
    return hashlib.md5(skills.encode()).hexdigest()[:8]


def fold_ranks(train_vals, all_vals):
    """Canonical rank transform (see rank_utils) — kept as an alias so existing
    imports keep working."""
    return ecdf_ranks(train_vals, all_vals)


def _rank_matrix(feat_rows, tr_idx):
    out = np.zeros((len(feat_rows), len(FEAT_NAMES)))
    for j, f in enumerate(FEAT_NAMES):
        vals = [r[f] for r in feat_rows]
        out[:, j] = fold_ranks([vals[i] for i in tr_idx], vals)
    return out


def _cum_logit(Xtr, ybin, Xte, C=1.0):
    """P(y=1) for one cumulative binary target; constant if the fold has one class."""
    if len(np.unique(ybin)) < 2:
        return np.full(Xte.shape[0], float(ybin[0]))
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=C).fit(Xtr, ybin).predict_proba(Xte)[:, 1]


def _fit_ordinal(Xtr, ytr, Xte, C=1.0):
    """Two cumulative binary logits -> class probs, monotone-corrected."""
    p2 = _cum_logit(Xtr, (ytr >= 2).astype(int), Xte, C)
    p3 = _cum_logit(Xtr, (ytr >= 3).astype(int), Xte, C)
    p3 = np.minimum(p2, p3)
    return np.column_stack([1 - p2, p2 - p3, p3])


def oof_cumulative_probs(feat_rows, labels, groups, texts=None, n_splits=5):
    """(n, 2) out-of-fold [P(D<=1), P(D<=2)] via GroupKFold on workspace.

    With `texts` (first-user texts, PII-collapsed) the head is the full
    word+char TF-IDF + numeric ordinal model; without, numeric-only."""
    from sklearn.model_selection import GroupKFold
    n = len(feat_rows)
    labels = np.asarray(labels)
    cum = np.zeros((n, 2))
    n_splits = min(n_splits, len(set(groups)))  # GroupKFold crashes on fewer groups
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(n), groups=groups):
        R = _rank_matrix(feat_rows, tr)
        if texts is not None:
            from scipy.sparse import hstack as sp_hstack, csr_matrix
            from sklearn.feature_extraction.text import TfidfVectorizer
            mats_tr, mats_te = [], []
            for kw in (WORD_TFIDF, CHAR_TFIDF):
                vec = TfidfVectorizer(**kw)
                mats_tr.append(vec.fit_transform([texts[i] for i in tr]))
                mats_te.append(vec.transform([texts[i] for i in te]))
            mats_tr.append(csr_matrix(R[tr]))
            mats_te.append(csr_matrix(R[te]))
            p = _fit_ordinal(sp_hstack(mats_tr).tocsr(), labels[tr],
                             sp_hstack(mats_te).tocsr())
        else:
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
