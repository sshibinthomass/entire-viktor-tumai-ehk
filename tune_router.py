#!/usr/bin/env python3
"""Benchmark + tune the router for COST-TO-PERFORMANCE.

Objective: the cost-quality frontier — served share (router tier >= observed
difficulty) vs estimated cost relative to sending everything to Tier 3. A
method is better when its frontier dominates; the headline scalar is the
frontier AUC over cost-share in [0.25, 1], plus served@50%-budget,
cost@90%-served, and the knee max(served - cost_share).

Rigor rules baked in:
  - The EVALUATOR IS FROZEN at its defaults while the router is tuned — tuning
    both would be circular. A sensitivity pass perturbs evaluator weights at
    the end to check the winner is not an artifact of one weighting.
  - GroupKFold on the workspace fingerprint (hash of the skill set): most
    workspaces run one kind of task, so random splits leak tenant identity.
  - Everything quoted is OUT-OF-FOLD. The weight search additionally prints
    its in-search (optimistic) score next to the OOF score — the gap is the
    selection bias.
  - Percentile ranks are refit per fold on train rows only.
  - Costs are chars/4 estimates under an assumed price sheet; the cache-aware
    model bills the replayed prefix at the cached rate (linear-growth
    approximation) and is the primary axis. Routing whole tasks (not single
    calls) means a policy never pays the cache-reset penalty.
  - lambda-Bayes comes in TWO flavors and they are not the same claim:
    "lambda-oracle" plugs each task's REALIZED cost into the dispatch rule —
    unknowable at routing time, so it is an upper bound, never a router
    result. "lambda-pred" uses an OOF routing-time cost prediction and is the
    deployable rule.

Usage: python tune_router.py [export.jsonl] [--fast]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from router.features import extract as extract_router, FEATURE_GROUPS
from router.tiering import DEFAULT_GROUP_WEIGHTS
from router.ml import workspace_of
from rank_utils import ecdf_ranks
from evaluator.metrics import trajectory_metrics
from evaluator.difficulty import (DEFAULT_WEIGHTS as EV_WEIGHTS,
                                  DEFAULT_CUTS as EV_CUTS,
                                  DEFAULT_OVERRIDES)
from run_pipeline import (load_trajectories, TIER_PRICES, DEFAULT_EXPORT,
                          oof_predicted_costs)

RNG = np.random.default_rng(7)
COST_GRID = np.linspace(0.25, 1.0, 76)   # frontier integration grid
CUT_SWEEP = [(c1, c2) for c2 in np.arange(0.20, 1.001, 0.02)
             for c1 in [c2 * r for r in (0.45, 0.6, 0.75)]]
TAU_SWEEP = np.concatenate([np.arange(0.30, 0.995, 0.02), [0.995]])
LAM_SWEEP = np.geomspace(1e-4, 30, 40)


# ---------------------------------------------------------------- data
def fold_ranks(train_vals, all_vals):
    return ecdf_ranks(train_vals, all_vals)


def build_dataset(export):
    trajs = load_trajectories(export)
    tids = sorted(trajs)
    feats = [extract_router(trajs[t]["first"]) for t in tids]
    mets = [trajectory_metrics(trajs[t]["deepest"], trajs[t]["n_calls"]) for t in tids]
    # workspace fingerprint for grouped CV — the SAME parts-aware function the
    # ML head uses (router.ml.workspace_of); the old duplicate here broke on
    # parts-list system content and collapsed 76% of rows into one group
    groups = [workspace_of(trajs[t]["first"]) for t in tids]
    # first-user text for the TF-IDF variant — the SAME extractor and length
    # limit the pipeline head uses (router.ml.first_user_text), so benchmark
    # and shipped numbers stay comparable
    from router.ml import first_user_text
    texts = [first_user_text(trajs[t]["first"]) for t in tids]
    return tids, feats, mets, np.array(groups), texts


def frozen_evaluator(mets):
    """Difficulty labels/scores with the evaluator frozen at its defaults
    (full-dataset ranks: the evaluator is the yardstick, not a learner)."""
    total = sum(EV_WEIGHTS.values())
    score = np.zeros(len(mets))
    for k, w in EV_WEIGHTS.items():
        score += w * fold_ranks([m[k] for m in mets], [m[k] for m in mets])
    score /= total
    c1, c2 = np.quantile(score, EV_CUTS[0]), np.quantile(score, EV_CUTS[1])
    d = np.where(score <= c1, 1, np.where(score <= c2, 2, 3)).astype(int)
    for i, m in enumerate(mets):
        if (m["n_tool_errors"] >= DEFAULT_OVERRIDES["errors_t3"]
                or m["n_tool_calls"] >= DEFAULT_OVERRIDES["tool_calls_t3"]):
            d[i] = 3
    return d, score


def evaluator_variant(mets, rng):
    """Perturbed evaluator (weights jittered x0.5..x2, cuts +-5pts) for the
    sensitivity pass."""
    w = {k: v * float(rng.uniform(0.5, 2.0)) for k, v in EV_WEIGHTS.items()}
    total = sum(w.values())
    score = np.zeros(len(mets))
    for k, wk in w.items():
        score += wk * fold_ranks([m[k] for m in mets], [m[k] for m in mets])
    score /= total
    cuts = np.clip([EV_CUTS[0] + rng.uniform(-.05, .05),
                    EV_CUTS[1] + rng.uniform(-.05, .05)], .3, .95)
    c1, c2 = np.quantile(score, cuts[0]), np.quantile(score, max(cuts[0] + .02, cuts[1]))
    d = np.where(score <= c1, 1, np.where(score <= c2, 2, 3)).astype(int)
    for i, m in enumerate(mets):
        if (m["n_tool_errors"] >= DEFAULT_OVERRIDES["errors_t3"]
                or m["n_tool_calls"] >= DEFAULT_OVERRIDES["tool_calls_t3"]):
            d[i] = 3
    return d


# ---------------------------------------------------------------- costs
def costs_matrix(mets, cache_aware=True):
    """(n, 3) $ estimate per trajectory per tier. Cache-aware: every input token
    is paid uncached once; the growing prefix is replayed by later calls at the
    cached rate (linear-growth approx: replay ~= T*(n-1)/2)."""
    out = np.zeros((len(mets), 3))
    for i, m in enumerate(mets):
        T, g, n = m["context_tokens"], m["gen_tokens"], max(m["n_llm_calls"], 1)
        for t in (1, 2, 3):
            pin, pc, pout = TIER_PRICES[t]
            if cache_aware:
                out[i, t - 1] = (T * pin + T * (n - 1) / 2 * pc + g * pout) / 1e6
            else:
                out[i, t - 1] = (T * pin + g * pout) / 1e6
    return out


# ---------------------------------------------------------------- frontier
def frontier_metrics(points, y=1):
    """points: list of (cost_share, served, weighted_served). y picks the
    quality column (1 = per-task served, 2 = token-weighted served)."""
    pts = sorted(set(points))
    xs = np.array([p[0] for p in pts]); ys = np.array([p[y] for p in pts])
    # upper envelope on the integration grid (best served at cost <= x)
    env = np.array([ys[xs <= x].max() if (xs <= x).any() else 0.0 for x in COST_GRID])
    env = np.maximum.accumulate(env)
    auc = float(np.trapezoid(env, COST_GRID) / (COST_GRID[-1] - COST_GRID[0]))
    at50 = float(ys[xs <= 0.5].max()) if (xs <= 0.5).any() else float("nan")
    c90 = float(xs[ys >= 0.9].min()) if (ys >= 0.9).any() else float("nan")
    knee_i = int(np.argmax(ys - xs))
    return {"auc": auc, "served@50%budget": at50, "cost@90%served": c90,
            "knee": {"cost": float(xs[knee_i]), "served": float(ys[knee_i]),
                     "gap": float(ys[knee_i] - xs[knee_i])}}


def eval_tiers(tiers, D, C):
    served = tiers >= D
    cost = C[np.arange(len(tiers)), tiers - 1].sum()
    top = C[:, 2].sum()
    w = C[:, 2]  # token weight ~ what the task would cost on the top tier
    return cost / top, float(served.mean()), float((served * w).sum() / w.sum())


def sweep_cuts(scores, D, C):
    pts = []
    for c1, c2 in CUT_SWEEP:
        q1, q2 = np.quantile(scores, min(c1, 1)), np.quantile(scores, min(c2, 1))
        tiers = np.where(scores <= q1, 1, np.where(scores <= q2, 2, 3))
        pts.append(eval_tiers(tiers, D, C))
    return pts


def sweep_tau(cum, D, C):
    """cum: (n,2) = P(D<=1), P(D<=2). Cheapest tier whose sufficiency >= tau."""
    pts = []
    for tau in TAU_SWEEP:
        tiers = np.where(cum[:, 0] >= tau, 1, np.where(cum[:, 1] >= tau, 2, 3))
        pts.append(eval_tiers(tiers, D, C))
    return pts


def sweep_lambda(cum, D, C, C_dispatch=None):
    """Bayes rule with task-specific costs: argmin_t cost$(t) + lam*P(D>t).

    C_dispatch is what the RULE sees; C is what the EVALUATION prices. Passing
    the realized C as dispatch input makes this an ORACLE (the rule knows each
    task's hindsight size); pass an OOF routing-time prediction for the
    deployable version."""
    Cd = C if C_dispatch is None else C_dispatch
    p_over = np.stack([1 - cum[:, 0], 1 - cum[:, 1], np.zeros(len(cum))], axis=1)
    pts = []
    for lam in LAM_SWEEP:
        tiers = np.argmin(Cd + lam * p_over, axis=1) + 1
        pts.append(eval_tiers(tiers, D, C))
    return pts


def sweep_lambda_sized(cum, D, C, C_dispatch=None):
    """Size-weighted Bayes rule: the miss penalty scales with the task's own
    top-tier cost — argmin_t cost$(t) + lam*C(3)*P(D>t). Bayes-optimal for the
    token-weighted quality objective. Same C_dispatch caveat as sweep_lambda."""
    Cd = C if C_dispatch is None else C_dispatch
    p_over = np.stack([1 - cum[:, 0], 1 - cum[:, 1], np.zeros(len(cum))], axis=1)
    pts = []
    for lam in np.geomspace(0.02, 50, 40):
        tiers = np.argmin(Cd + lam * Cd[:, 2:3] * p_over, axis=1) + 1
        pts.append(eval_tiers(tiers, D, C))
    return pts


# ---------------------------------------------------------------- scores
FEAT_NAMES = [f for g in FEATURE_GROUPS.values() for f in g["features"]]
G_SLICES, _off = {}, 0
for _g, _spec in FEATURE_GROUPS.items():
    G_SLICES[_g] = (_spec["sign"], slice(_off, _off + len(_spec["features"])))
    _off += len(_spec["features"])


def rank_mat(feats, tr_idx, te_idx):
    """Per-fold rank matrix: ranks fit on train rows, applied to all rows."""
    out = np.zeros((len(feats), len(FEAT_NAMES)))
    for j, f in enumerate(FEAT_NAMES):
        vals = [r[f] for r in feats]
        out[:, j] = fold_ranks([vals[i] for i in tr_idx], vals)
    return out


def group_score(R, gw):
    s = np.zeros(R.shape[0]); tw = 0.0
    for g, (sign, sl) in G_SLICES.items():
        s += sign * gw[g] * R[:, sl].mean(axis=1); tw += gw[g]
    return s / tw


def search_weights(R, D, C, idx, n_draws):
    """Random search over group weights maximizing frontier AUC on `idx` rows."""
    cands = [DEFAULT_GROUP_WEIGHTS] + [
        {g: float(w) for g, w in zip(G_SLICES, draw)}
        for draw in RNG.uniform(0.0, 2.0, size=(n_draws, len(G_SLICES)))]
    best, best_auc = None, -1
    for gw in cands:
        if sum(gw.values()) < 1e-9:
            continue
        s = group_score(R[idx], gw)
        auc = frontier_metrics(sweep_cuts(s, D[idx], C[idx]))["auc"]
        if auc > best_auc:
            best, best_auc = gw, auc
    return best, best_auc


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?", default=DEFAULT_EXPORT)
    ap.add_argument("--fast", action="store_true", help="fewer search draws")
    a = ap.parse_args()
    n_draws = 60 if a.fast else 300

    print("building dataset ...", flush=True)
    tids, feats, mets, groups, texts = build_dataset(a.export)
    n = len(tids)
    D, e_score = frozen_evaluator(mets)
    C = costs_matrix(mets, cache_aware=True)
    C_naive = costs_matrix(mets, cache_aware=False)
    print(f"n={n}  workspaces={len(set(groups))}  "
          f"difficulty mix: {[int((D==k).sum()) for k in (1,2,3)]}")

    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=5).split(np.zeros(n), groups=groups))

    results, frontiers = {}, {}

    def record(name, pts, note=""):
        fm = frontier_metrics(pts)
        fw = frontier_metrics(pts, y=2)
        results[name] = {**fm, "weighted_auc": fw["auc"],
                         "weighted@50%budget": fw["served@50%budget"], "note": note}
        frontiers[name] = sorted(set(tuple(round(v, 4) for v in p) for p in pts))
        k = fm["knee"]
        print(f"{name:22s} AUC {fm['auc']:.3f}  @50%budget {fm['served@50%budget']:.3f}  "
              f"cost@90% {fm['cost@90%served']:.3f}  "
              f"knee {k['served']:.2f}served/{k['cost']:.2f}cost  "
              f"| wAUC {fw['auc']:.3f} w@50% {fw['served@50%budget']:.3f}  {note}")

    # ---- reference points
    oracle_cost, _, _ = eval_tiers(D, D, C)
    print(f"\noracle (tier = difficulty): served 100% at {oracle_cost:.1%} of all-Tier3 cost")
    for t in (1, 2, 3):
        cs, sv, wsv = eval_tiers(np.full(n, t), D, C)
        print(f"always-T{t}: cost {cs:.1%}  served {sv:.1%}  weighted-served {wsv:.1%}")

    print(f"\n--- out-of-fold frontiers (cache-aware costs, evaluator frozen) ---")

    # ---- 1. default score (no training; per-fold ranks for parity)
    s_def = np.zeros(n)
    for tr, te in folds:
        R = rank_mat(feats, tr, te)
        s_def[te] = group_score(R, DEFAULT_GROUP_WEIGHTS)[te]
    record("score/default", sweep_cuts(s_def, D, C))

    # ---- 2. tuned score weights (search on train folds only)
    s_tuned = np.zeros(n)
    fold_ws, search_aucs = [], []
    for tr, te in folds:
        R = rank_mat(feats, tr, te)
        gw, in_auc = search_weights(R, D, C, tr, n_draws)
        fold_ws.append(gw); search_aucs.append(in_auc)
        s_tuned[te] = group_score(R, gw)[te]
    record("score/tuned", sweep_cuts(s_tuned, D, C),
           f"(in-search AUC {np.mean(search_aucs):.3f} -> gap = selection bias)")
    consensus = {g: round(float(np.mean([w[g] for w in fold_ws])), 2) for g in G_SLICES}
    print(f"  consensus tuned weights: {consensus}  (per-fold: {fold_ws})")

    # ---- 3/4. unsupervised clustering (single operating points)
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    for name, mk in [("kmeans", lambda: KMeans(3, n_init=10, random_state=13)),
                     ("gmm", lambda: GaussianMixture(3, n_init=3, random_state=13))]:
        tiers = np.zeros(n, dtype=int)
        for tr, te in folds:
            R = rank_mat(feats, tr, te)
            w = np.concatenate([np.full(sl.stop - sl.start, DEFAULT_GROUP_WEIGHTS[g])
                                for g, (_, sl) in G_SLICES.items()])
            m = mk().fit(R[tr] * w)
            lab_tr, lab_te = m.predict(R[tr] * w), m.predict(R[te] * w)
            ref = group_score(R, DEFAULT_GROUP_WEIGHTS)
            order = np.argsort([ref[tr][lab_tr == c].mean() if (lab_tr == c).any() else 9
                                for c in range(3)])
            remap = {int(c): t + 1 for t, c in enumerate(order)}
            tiers[te] = [remap[int(c)] for c in lab_te]
        record(name, [eval_tiers(tiers, D, C)], "(single point, no knob)")

    # ---- 5/6. supervised probabilistic: OOF P(D<=t), two dispatch rules
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    def oof_cum(fit_fn):
        cum = np.zeros((n, 2))
        for tr, te in folds:
            R = rank_mat(feats, tr, te)
            p = fit_fn(R[tr], D[tr], R[te])          # (len(te), 3) class probs
            cum[te, 0] = p[:, 0]
            cum[te, 1] = p[:, 0] + p[:, 1]
        return cum

    def fit_logreg(Xtr, ytr, Xte):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        return align_probs(m, Xte)

    def fit_hgb(Xtr, ytr, Xte):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                           learning_rate=0.06,
                                           random_state=13).fit(Xtr, ytr)
        return align_probs(m, Xte)

    def align_probs(m, X):
        p = m.predict_proba(X)
        out = np.zeros((len(X), 3))
        for j, cls in enumerate(m.classes_):
            out[:, int(cls) - 1] = p[:, j]
        return out

    def fit_ordlog(Xtr, ytr, Xte):
        # ordinal: two cumulative binary models P(D>=2), P(D>=3)
        out = np.zeros((len(Xte), 3))
        p2 = LogisticRegression(max_iter=2000).fit(Xtr, (ytr >= 2).astype(int)) \
            .predict_proba(Xte)[:, 1]
        p3 = LogisticRegression(max_iter=2000).fit(Xtr, (ytr >= 3).astype(int)) \
            .predict_proba(Xte)[:, 1]
        p3 = np.minimum(p2, p3)  # enforce monotone cumulative
        out[:, 0] = 1 - p2; out[:, 1] = p2 - p3; out[:, 2] = p3
        return out

    # routing-time cost prediction for the deployable lambda rules (OOF)
    print("predicting routing-time costs (OOF regressors) ...", flush=True)
    C_hat = oof_predicted_costs(feats, mets, np.asarray(groups))

    ORACLE = "(oracle: dispatch saw realized cost — upper bound, NOT a router)"
    cum_lr = oof_cum(fit_logreg)
    record("logreg/tau", sweep_tau(cum_lr, D, C))
    record("logreg/lambda-pred", sweep_lambda(cum_lr, D, C, C_dispatch=C_hat))
    record("logreg/lambda-oracle", sweep_lambda(cum_lr, D, C), ORACLE)
    cum_ord = oof_cum(fit_ordlog)
    record("ordlog/tau", sweep_tau(cum_ord, D, C))
    record("ordlog/lambda-pred", sweep_lambda(cum_ord, D, C, C_dispatch=C_hat))
    record("ordlog/lambda-oracle", sweep_lambda(cum_ord, D, C), ORACLE)
    record("ordlog/lambda-sized-pred", sweep_lambda_sized(cum_ord, D, C, C_dispatch=C_hat))
    record("ordlog/lambda-sized-oracle", sweep_lambda_sized(cum_ord, D, C), ORACLE)

    # ---- cost-weighted training: be right where being wrong is expensive
    def oof_cum_weighted(fit_fn):
        cum = np.zeros((n, 2))
        sw_all = C[:, 2] / C[:, 2].mean()
        for tr, te in folds:
            R = rank_mat(feats, tr, te)
            p = fit_fn(R[tr], D[tr], R[te], sw_all[tr])
            cum[te, 0] = p[:, 0]
            cum[te, 1] = p[:, 0] + p[:, 1]
        return cum

    def fit_logreg_sw(Xtr, ytr, Xte, sw):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr, sample_weight=sw)
        return align_probs(m, Xte)

    cum_wlr = oof_cum_weighted(fit_logreg_sw)
    record("wlogreg/tau", sweep_tau(cum_wlr, D, C))
    record("wlogreg/lambda-sized-pred", sweep_lambda_sized(cum_wlr, D, C, C_dispatch=C_hat))
    cum_hgb = oof_cum(fit_hgb)
    record("hgb/tau", sweep_tau(cum_hgb, D, C))
    record("hgb/lambda-pred", sweep_lambda(cum_hgb, D, C, C_dispatch=C_hat))

    # ---- 7. text: TF-IDF + SVD + numeric ranks -> logistic
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    cum_tx = np.zeros((n, 2))
    for tr, te in folds:
        R = rank_mat(feats, tr, te)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000,
                              sublinear_tf=True, strip_accents="unicode")
        Ttr = vec.fit_transform([texts[i] for i in tr])
        svd = TruncatedSVD(min(80, Ttr.shape[1] - 1), random_state=0).fit(Ttr)
        Xtr = np.hstack([R[tr], normalize(svd.transform(Ttr))])
        Xte = np.hstack([R[te], normalize(svd.transform(
            vec.transform([texts[i] for i in te])))])
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, D[tr])
        p = align_probs(m, Xte)
        cum_tx[te, 0] = p[:, 0]; cum_tx[te, 1] = p[:, 0] + p[:, 1]
    record("tfidf+logreg/tau", sweep_tau(cum_tx, D, C))
    record("tfidf+logreg/lambda-pred", sweep_lambda(cum_tx, D, C, C_dispatch=C_hat))
    record("tfidf+logreg/lambda-oracle", sweep_lambda(cum_tx, D, C), ORACLE)

    # ---- 8. random baseline: 100 permutations, not one lucky draw
    rand_aucs, rand_w50 = [], []
    rand_pts_first = None
    for i in range(100):
        pts = sweep_cuts(RNG.permutation(s_def), D, C)
        if rand_pts_first is None:
            rand_pts_first = pts
        fm = frontier_metrics(pts)
        rand_aucs.append(fm["auc"])
        rand_w50.append(fm["served@50%budget"])
    record("random", rand_pts_first,
           f"(100 perms: AUC {np.mean(rand_aucs):.3f} +/- {1.96 * np.std(rand_aucs):.3f})")
    results["random"]["auc_mean_100perm"] = float(np.mean(rand_aucs))
    results["random"]["auc_ci95_100perm"] = [
        float(np.percentile(rand_aucs, 2.5)), float(np.percentile(rand_aucs, 97.5))]
    # honest context: most of any method's AUC is the 55%-D1 floor — quote the
    # margin over random, not the absolute AUC
    print(f"  random AUC over 100 permutations: {np.mean(rand_aucs):.3f} "
          f"(95% CI [{results['random']['auc_ci95_100perm'][0]:.3f}, "
          f"{results['random']['auc_ci95_100perm'][1]:.3f}])")

    # ---- winner under the cache-blind cost model too (comparability).
    # Oracle rows are upper bounds, not routers — excluded from the winner pick.
    best = max((k for k in results if k != "random" and "oracle" not in k),
               key=lambda k: results[k]["auc"])
    print(f"\nwinner by OOF frontier AUC (oracle rows excluded): {best}")
    cums = {"logreg": cum_lr, "ordlog": cum_ord, "hgb": cum_hgb,
            "tfidf+logreg": cum_tx, "wlogreg": cum_wlr}

    def run_method(name, Dv, C_eval):
        head, _, rule = name.partition("/")
        if head in cums:
            cum = cums[head]
            if rule == "tau":
                return sweep_tau(cum, Dv, C_eval)
            if rule.startswith("lambda-sized"):
                return sweep_lambda_sized(cum, Dv, C_eval,
                                          C_dispatch=C_hat if rule.endswith("pred") else None)
            return sweep_lambda(cum, Dv, C_eval,
                                C_dispatch=C_hat if rule.endswith("pred") else None)
        return sweep_cuts(s_tuned if name == "score/tuned" else s_def, Dv, C_eval)

    fmn = frontier_metrics(run_method(best, D, C_naive))
    print(f"  same winner under cache-blind costs: AUC {fmn['auc']:.3f}, "
          f"knee {fmn['knee']['served']:.2f}served/{fmn['knee']['cost']:.2f}cost")

    # ---- evaluator sensitivity for the winner (is the win an artifact?)
    print("\nevaluator sensitivity (30 perturbed evaluators, winner's served@50%budget):")
    rng = np.random.default_rng(11)
    vals, label_agreement = [], []
    for _ in range(30):
        Dv = evaluator_variant(mets, rng)
        label_agreement.append(float((Dv == D).mean()))
        vals.append(frontier_metrics(run_method(best, Dv, C))["served@50%budget"])
    print(f"  mean {np.mean(vals):.3f}  sd {np.std(vals):.3f}  "
          f"min {np.min(vals):.3f} (frozen-evaluator value: "
          f"{results[best]['served@50%budget']:.3f})")
    print(f"  perturbed evaluators agree with the frozen labels "
          f"{np.mean(label_agreement):.1%} on average — that agreement bounds any router")

    out = {
        "objective": "cost-to-performance frontier, OOF, GroupKFold(workspace), cache-aware costs",
        "oracle_cost_share": oracle_cost,
        "results": results,
        "frontiers": frontiers,
        "tuned_group_weights_consensus": consensus,
        "winner": best,
        "lambda_note": "lambda-oracle rows dispatch on realized (hindsight) cost: "
                       "upper bounds, not router results; lambda-pred rows use an "
                       "OOF routing-time cost prediction and are deployable",
        "evaluator_sensitivity_served@50": {
            "mean": float(np.mean(vals)), "sd": float(np.std(vals)),
            "min": float(np.min(vals))},
        "evaluator_label_agreement": {
            "mean": float(np.mean(label_agreement)),
            "min": float(np.min(label_agreement)),
            "n_variants": 30,
            "note": "exact agreement of perturbed evaluator configs (weights x0.5-2, "
                    "cuts +/-5pts) with the frozen default labels"},
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/tuning_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    np.savez("results/tuning_arrays.npz", s_def=s_def, s_tuned=s_tuned,
             cum_lr=cum_lr, cum_ord=cum_ord, cum_hgb=cum_hgb, cum_tx=cum_tx,
             cum_wlr=cum_wlr, D=D, C=C, C_naive=C_naive, e_score=e_score,
             tids=np.array(tids))
    print("\nwrote results/tuning_report.json + results/tuning_arrays.npz")


if __name__ == "__main__":
    main()
