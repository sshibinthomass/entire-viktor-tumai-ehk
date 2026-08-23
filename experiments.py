#!/usr/bin/env python3
"""Method ladder for exact agreement + confusion diagonal, all OOF.

Every candidate is scored with the same protocol: GroupKFold(5) on the
workspace fingerprint, evaluator frozen at defaults, tiers assigned by
marginal-matched percentile cuts on the model's expected-difficulty score
unless a dispatch variant says otherwise.

Reported per candidate: exact, balanced accuracy (mean per-class recall — the
honest 'confusion diagonal' number), adjacent, Spearman, D3 recall.

Usage:
  python experiments.py build     # featurize once -> results/exp_cache.npz
  python experiments.py run       # run the ladder from the cache
"""
import json
import sys
from pathlib import Path

import numpy as np

from router.features import extract as extract_router, FEATURE_GROUPS, ML_EXTRA_FEATURES, PII_RE
from router.ml import workspace_of, fold_ranks
from evaluator.metrics import trajectory_metrics
from evaluator.difficulty import grade
from run_pipeline import load_trajectories, DEFAULT_EXPORT

CACHE = Path("results/exp_cache.npz")
BASE_FEATS = [f for g in FEATURE_GROUPS.values() for f in g["features"]]
ALL_FEATS = BASE_FEATS + ML_EXTRA_FEATURES


# ------------------------------------------------------------------ build
def build():
    trajs = load_trajectories(DEFAULT_EXPORT)
    tids = sorted(trajs)
    feats = [extract_router(trajs[t]["first"]) for t in tids]
    mets = [trajectory_metrics(trajs[t]["deepest"], trajs[t]["n_calls"]) for t in tids]
    D, e = grade(mets)
    X = np.array([[float(r[f]) for f in ALL_FEATS] for r in feats])
    groups = np.array([workspace_of(trajs[t]["first"]) for t in tids])
    texts, sys_texts, tool_texts = [], [], []
    for t in tids:
        req = trajs[t]["first"]
        fu = next((i for i in req["input"] if i.get("role") == "user"), None)
        c = (fu or {}).get("content")
        txt = c if isinstance(c, str) else "\n".join(
            p.get("text", "") for p in (c or []) if isinstance(p, dict))
        texts.append(PII_RE.sub("<E>", txt)[:12000])
        sysp = next((i.get("content") for i in req["input"]
                     if i.get("role") == "system"), "") or ""
        if not isinstance(sysp, str):
            sysp = json.dumps(sysp)
        sys_texts.append(PII_RE.sub("<E>", sysp)[:6000])
        tool_texts.append(" ".join(sorted((tl.get("name") or "") for tl in
                                          (req.get("tools") or []))))
    np.savez(CACHE, X=X, D=D, e=e, groups=groups,
             texts=np.array(texts, dtype=object),
             sys_texts=np.array(sys_texts, dtype=object),
             tool_texts=np.array(tool_texts, dtype=object), tids=np.array(tids))
    print(f"cached {X.shape} features -> {CACHE}")


# ------------------------------------------------------------------ scoring
def metrics(tiers, D, e=None, score=None):
    conf = np.array([[(np.array(tiers) == r + 1)[D == c + 1].sum() for c in range(3)]
                     for r in range(3)])
    recalls = [conf[k, k] / max((D == k + 1).sum(), 1) for k in range(3)]
    out = {
        "exact": float((tiers == D).mean()),
        "balanced": float(np.mean(recalls)),
        "adjacent": float((np.abs(tiers - D) <= 1).mean()),
        "d3_recall": float(recalls[2]),
        "conf": conf.tolist(),
    }
    if score is not None and e is not None:
        out["spearman"] = float(np.corrcoef(fold_ranks(score, score), fold_ranks(e, e))[0, 1])
    return out


def cut_dispatch(score, D):
    """Marginal-matched percentile cuts (the label mix decides the tier mix)."""
    p1, p2 = (D == 1).mean(), (D <= 2).mean()
    q1, q2 = np.quantile(score, p1), np.quantile(score, p2)
    return np.where(score <= q1, 1, np.where(score <= q2, 2, 3))


def show(name, tiers, D, e=None, score=None):
    m = metrics(tiers, D, e, score)
    sp = f"  rho {m['spearman']:.3f}" if "spearman" in m else ""
    print(f"{name:38s} exact {m['exact']:.1%}  balanced {m['balanced']:.1%}  "
          f"adj {m['adjacent']:.1%}  D3rec {m['d3_recall']:.1%}{sp}")
    return m


# ------------------------------------------------------------------ models
def rank_fold(X, tr):
    R = np.zeros_like(X)
    for j in range(X.shape[1]):
        R[:, j] = fold_ranks(X[tr, j], X[:, j])
    return R


def oof_probs(X, D, folds, fit, texts=None):
    """fit(Xtr, ytr, Xte, [Ttr, Tte]) -> (m,3) class probs; returns OOF (n,3)."""
    P = np.zeros((len(D), 3))
    for tr, te in folds:
        R = rank_fold(X, tr)
        if texts is not None:
            P[te] = fit(R[tr], D[tr], R[te], [texts[i] for i in tr], [texts[i] for i in te])
        else:
            P[te] = fit(R[tr], D[tr], R[te])
    return P


def align(m, X):
    p = m.predict_proba(X)
    out = np.zeros((X.shape[0], 3))
    for j, c in enumerate(m.classes_):
        out[:, int(c) - 1] = p[:, j]
    return out


def f_ordlog(bal=False):
    from sklearn.linear_model import LogisticRegression
    cw = "balanced" if bal else None
    def fit(Xtr, ytr, Xte):
        p2 = LogisticRegression(max_iter=2000, class_weight=cw).fit(
            Xtr, (ytr >= 2).astype(int)).predict_proba(Xte)[:, 1]
        p3 = LogisticRegression(max_iter=2000, class_weight=cw).fit(
            Xtr, (ytr >= 3).astype(int)).predict_proba(Xte)[:, 1]
        p3 = np.minimum(p2, p3)
        return np.column_stack([1 - p2, p2 - p3, p3])
    return fit


def f_hgb(bal=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    def fit(Xtr, ytr, Xte):
        sw = None
        if bal:
            cnt = np.bincount(ytr, minlength=4)[1:]
            sw = (len(ytr) / (3 * cnt))[ytr - 1]
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=300,
                                           learning_rate=0.06, random_state=13)
        m.fit(Xtr, ytr, sample_weight=sw)
        return align(m, Xte)
    return fit


def f_lgbm(bal=False):
    import lightgbm as lgb
    def fit(Xtr, ytr, Xte):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                               num_leaves=15, subsample=0.9, colsample_bytree=0.8,
                               class_weight="balanced" if bal else None,
                               random_state=13, verbose=-1)
        m.fit(Xtr, ytr)
        return align(m, Xte)
    return fit


def f_xgb():
    from xgboost import XGBClassifier
    def fit(Xtr, ytr, Xte):
        m = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                          subsample=0.9, colsample_bytree=0.8, random_state=13,
                          verbosity=0)
        m.fit(Xtr, ytr - 1)
        return m.predict_proba(Xte)
    return fit


def f_cat():
    from catboost import CatBoostClassifier
    def fit(Xtr, ytr, Xte):
        m = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4,
                               random_seed=13, verbose=False)
        m.fit(Xtr, ytr)
        return align(m, Xte)
    return fit


def f_mord():
    from mord import LogisticAT
    def fit(Xtr, ytr, Xte):
        m = LogisticAT(alpha=1.0).fit(Xtr, ytr)
        return m.predict_proba(Xte)
    return fit


def f_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from sklearn.linear_model import LogisticRegression
    def fit(Xtr, ytr, Xte, Ttr, Tte):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000,
                              sublinear_tf=True, strip_accents="unicode")
        A = vec.fit_transform(Ttr)
        svd = TruncatedSVD(min(80, A.shape[1] - 1), random_state=0).fit(A)
        Ztr = np.hstack([Xtr, normalize(svd.transform(A))])
        Zte = np.hstack([Xte, normalize(svd.transform(vec.transform(Tte)))])
        m = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
        return align(m, Zte)
    return fit


# ------------------------------------------------------------------ run
def run():
    z = np.load(CACHE, allow_pickle=True)
    X, D, e, groups, texts = z["X"], z["D"], z["e"], z["groups"], list(z["texts"])
    n = len(D)
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(5).split(np.zeros(n), groups=groups))
    base_idx = [ALL_FEATS.index(f) for f in BASE_FEATS]

    results = {}

    def eval_probs(name, P, score=None):
        s = score if score is not None else 1 - (P[:, 0] + (P[:, 0] + P[:, 1])) / 2
        results[name] = {"P": P, "score": s,
                         "m": show(name, cut_dispatch(s, D), D, e, s)}

    print(f"n={n}  label mix {[int((D==k).sum()) for k in (1,2,3)]}  "
          f"always-T1 exact = {(D==1).mean():.1%}\n")

    eval_probs("ordlog/base-feats", oof_probs(X[:, base_idx], D, folds, f_ordlog()))
    eval_probs("ordlog/v2-feats", oof_probs(X, D, folds, f_ordlog()))
    eval_probs("hgb/v2", oof_probs(X, D, folds, f_hgb()))
    eval_probs("hgb/v2-balanced", oof_probs(X, D, folds, f_hgb(bal=True)))
    for nm, mk in [("lgbm/v2", f_lgbm), ("xgb/v2", f_xgb), ("cat/v2", f_cat),
                   ("mord/v2", f_mord)]:
        try:
            eval_probs(nm, oof_probs(X, D, folds, mk() if nm != "lgbm/v2" else mk()))
        except ImportError as ex:
            print(f"{nm:38s} skipped ({ex})")
    try:
        eval_probs("lgbm/v2-balanced", oof_probs(X, D, folds, f_lgbm(bal=True)))
    except ImportError:
        pass
    eval_probs("tfidf+lr/v2", oof_probs(X, D, folds, f_tfidf(), texts=texts))

    # ---- stack: meta-logistic on nested-OOF level-0 probs
    from sklearn.linear_model import LogisticRegression
    level0 = [k for k in ("ordlog/v2-feats", "hgb/v2", "lgbm/v2", "tfidf+lr/v2",
                          "cat/v2", "mord/v2") if k in results]
    fitters = {"ordlog/v2-feats": (f_ordlog(), None), "hgb/v2": (f_hgb(), None),
               "lgbm/v2": None, "tfidf+lr/v2": (f_tfidf(), texts),
               "cat/v2": None, "mord/v2": None}
    try: fitters["lgbm/v2"] = (f_lgbm(), None)
    except Exception: pass
    try: fitters["cat/v2"] = (f_cat(), None)
    except Exception: pass
    try: fitters["mord/v2"] = (f_mord(), None)
    except Exception: pass
    P_stack = np.zeros((n, 3))
    for tr, te in folds:
        inner = list(GroupKFold(4).split(np.zeros(len(tr)), groups=groups[tr]))
        Z_tr = []  # level-0 OOF probs inside the outer-train
        Z_te = []
        for k in level0:
            if fitters.get(k) is None:
                continue
            fit, txt = fitters[k]
            Pin = np.zeros((len(tr), 3))
            for itr, ite in inner:
                Xi = X[tr]
                R = rank_fold(Xi, itr)
                if txt is not None:
                    Pin[ite] = fit(R[itr], D[tr][itr], R[ite],
                                   [txt[tr[i]] for i in itr], [txt[tr[i]] for i in ite])
                else:
                    Pin[ite] = fit(R[itr], D[tr][itr], R[ite])
            Z_tr.append(Pin[:, :2])
            R = rank_fold(X, tr)
            if txt is not None:
                Pte = fit(R[tr], D[tr], R[te], [txt[i] for i in tr], [txt[i] for i in te])
            else:
                Pte = fit(R[tr], D[tr], R[te])
            Z_te.append(Pte[:, :2])
        meta = LogisticRegression(max_iter=2000).fit(np.hstack(Z_tr), D[tr])
        P_stack[te] = align(meta, np.hstack(Z_te))
    eval_probs("STACK/nested", P_stack)

    # ---- dispatch variants on the best-balanced candidate
    best = max(results, key=lambda k: results[k]["m"]["exact"])
    print(f"\nbest by exact: {best} — dispatch variants on it:")
    P = results[best]["P"]
    show("  expected-L1-cost dispatch",
         np.array([np.argmin([sum(abs(t - d) * P[i, d - 1] for d in (1, 2, 3))
                              for t in (1, 2, 3)]) + 1 for i in range(n)]), D, e,
         results[best]["score"])
    show("  argmax dispatch", P.argmax(1) + 1, D, e, results[best]["score"])
    # blend with heuristic (50/50 in rank space)
    from router.tiering import composite_scores, rank_matrix
    feats_dicts = None  # heuristic score from base ranks
    Rb = np.zeros((n, len(BASE_FEATS)))
    for j in range(len(BASE_FEATS)):
        Rb[:, j] = fold_ranks(X[:, base_idx[j]], X[:, base_idx[j]])
    h = composite_scores(Rb, BASE_FEATS)
    sb = 0.5 * fold_ranks(results[best]["score"], results[best]["score"]) + \
         0.5 * fold_ranks(h, h)
    show("  50/50 heuristic blend + cuts", cut_dispatch(sb, D), D, e, sb)

    with open("results/experiments_report.json", "w", encoding="utf-8") as f:
        json.dump({k: v["m"] for k, v in results.items()}, f, indent=2)
    np.savez("results/exp_probs.npz",
             **{k.replace("/", "_"): v["P"] for k, v in results.items()})
    print("\nwrote results/experiments_report.json + results/exp_probs.npz")


if __name__ == "__main__":
    (build if "build" in sys.argv else run)()
