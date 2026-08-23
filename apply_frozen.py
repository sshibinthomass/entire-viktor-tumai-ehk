#!/usr/bin/env python3
"""Route a HELD-OUT chunk (or one task) with the frozen router — zero refitting.

Loads the artifact freeze_router.py produced from chunk 01 and applies it:
ECDFs, TF-IDF vocabularies, logit coefficients, cut values and the evaluator
yardstick are all fixed. Nothing about the new data changes any transform, so
these are true held-out numbers — report them SEPARATELY from the public-split
numbers (the submission requires that separation).

Modes:
  chunk:  python apply_frozen.py export_linked/trajectories_v1_02.jsonl
          -> routes every trajectory; if the chunk is enriched it also grades
             it with the FROZEN evaluator and prints held-out agreement/served
  task:   python apply_frozen.py --one request.json
          -> routes a single request line (the single-task inference path);
             prints tier + P(D<=1), P(D<=2)

Usage: python apply_frozen.py [export.jsonl] [--frozen results/frozen_router.pkl]
                              [--method blend|score|ml-tau|ml-lambda] [--one FILE]
                              [--out results/heldout_report.json]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack as sp_hstack, csr_matrix

from router.features import extract as extract_router
from router.ml import first_user_text
from evaluator.metrics import trajectory_metrics
from run_pipeline import load_trajectories, compare, costs_matrix


def ecdf_apply(store, x):
    x = np.asarray(x, dtype=float)
    if len(store) == 0 or store[0] == store[-1]:
        return np.full(x.shape, 0.5)
    return np.searchsorted(store, x, side="right") / len(store)


def route_requests(art, first_reqs):
    """-> dict of scores/probs/tiers for a list of first-call requests."""
    from router.features import FEATURE_GROUPS
    feats = [extract_router(r) for r in first_reqs]
    texts = [first_user_text(r) for r in first_reqs]
    rank_cols = {f: ecdf_apply(art["feat_ecdfs"][f], [row[f] for row in feats])
                 for f in art["feat_names"]}
    gw, total_w = art["group_weights"], sum(art["group_weights"].values())
    h = np.zeros(len(feats))
    for g, spec in FEATURE_GROUPS.items():
        cols = np.column_stack([rank_cols[f] for f in spec["features"]])
        h += spec["sign"] * gw[g] * cols.mean(axis=1)
    h /= total_w

    R = np.column_stack([rank_cols[f] for f in art["feat_names"]])
    Z = sp_hstack([v.transform(texts) for v in art["tfidf_vecs"]]
                  + [csr_matrix(R)]).tocsr()
    p2 = art["lr2"].predict_proba(Z)[:, 1]
    p3 = np.minimum(p2, art["lr3"].predict_proba(Z)[:, 1])
    cum = np.column_stack([1 - p2, 1 - p3])  # P(D<=1), P(D<=2)
    ml_difficulty = 1 - (cum[:, 0] + cum[:, 1]) / 2

    blend = art["alpha"] * ecdf_apply(art["ml_ecdf"], ml_difficulty) \
        + (1 - art["alpha"]) * ecdf_apply(art["h_ecdf"], h)
    c1, c2 = art["blend_cut_vals"]
    tiers_blend = np.where(blend <= c1, 1, np.where(blend <= c2, 2, 3)).astype(int)
    hc1, hc2 = art["h_cut_vals"]
    tiers_score = np.where(h <= hc1, 1, np.where(h <= hc2, 2, 3)).astype(int)
    tiers_tau = np.where(cum[:, 0] >= art["tau"], 1,
                         np.where(cum[:, 1] >= art["tau"], 2, 3)).astype(int)
    # lambda: predicted routing-time costs from the frozen regressors
    T = np.maximum(np.expm1(art["cost_regs"]["context_tokens"].predict(R)), 0)
    g = np.maximum(np.expm1(art["cost_regs"]["gen_tokens"].predict(R)), 0)
    nc = np.maximum(np.expm1(art["cost_regs"]["n_llm_calls"].predict(R)), 1)
    C_hat = np.zeros((len(feats), 3))
    for t in (1, 2, 3):
        pin, pc, pout = art["tier_prices"][t]
        C_hat[:, t - 1] = (T * pin + T * (nc - 1) / 2 * pc + g * pout) / 1e6
    p_over = np.stack([1 - cum[:, 0], 1 - cum[:, 1], np.zeros(len(cum))], axis=1)
    tiers_lam = (np.argmin(C_hat + art["lambda"] * p_over, axis=1) + 1).astype(int)

    return {"cum": cum, "heuristic": h, "blend": blend,
            "tiers": {"blend": tiers_blend, "score": tiers_score,
                      "ml-tau": tiers_tau, "ml-lambda": tiers_lam}}


def frozen_grade(art, mets):
    ev = art["evaluator"]
    total = sum(ev["weights"].values())
    score = np.zeros(len(mets))
    for k, w in ev["weights"].items():
        score += w * ecdf_apply(ev["ecdfs"][k], [m[k] for m in mets])
    score /= total
    c1, c2 = ev["cut_vals"]
    D = np.where(score <= c1, 1, np.where(score <= c2, 2, 3)).astype(int)
    ov = ev["overrides"]
    for i, m in enumerate(mets):
        if (m["n_tool_errors"] >= ov.get("errors_t3", 10**9)
                or m["n_tool_calls"] >= ov.get("tool_calls_t3", 10**9)):
            D[i] = 3
    return D, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?", default=None)
    ap.add_argument("--frozen", default="results/frozen_router.pkl")
    ap.add_argument("--method", default="blend",
                    choices=["blend", "score", "ml-tau", "ml-lambda"])
    ap.add_argument("--one", default=None, help="route a single request JSON file")
    ap.add_argument("--out", default="results/heldout_report.json")
    a = ap.parse_args()

    fz = Path(a.frozen)
    if not fz.exists():
        sys.exit(f"{fz} not found — run 'python freeze_router.py' on chunk 01 first")
    with open(fz, "rb") as f:
        art = pickle.load(f)
    print(f"frozen router: fit on {art['fit_source']} (n={art['n_fit']})")

    if a.one:
        req = json.loads(Path(a.one).read_text(encoding="utf-8"))
        r = route_requests(art, [req])
        t = r["tiers"][a.method][0]
        print(f"single task -> Tier {t}  "
              f"P(D<=1)={r['cum'][0, 0]:.3f}  P(D<=2)={r['cum'][0, 1]:.3f}  "
              f"(method={a.method}; every transform frozen from {art['fit_source']})")
        return

    if not a.export:
        sys.exit("pass a held-out chunk (enriched jsonl) or --one request.json")
    trajs = load_trajectories(a.export)
    tids = sorted(trajs)
    print(f"routing {len(tids)} held-out trajectories from {Path(a.export).name}")
    r = route_requests(art, [trajs[t]["first"] for t in tids])
    tiers = r["tiers"][a.method]
    print("tier mix: " + " ".join(f"T{k}={int((tiers == k).sum())}" for k in (1, 2, 3)))

    # grade with the FROZEN evaluator (chunk-01 yardstick, no refit on this chunk)
    mets = [trajectory_metrics(trajs[t]["deepest"], trajs[t]["n_calls"]) for t in tids]
    D, e_scores = frozen_grade(art, mets)
    C = costs_matrix(mets, cache_aware=True, tier_prices=art["tier_prices"])
    rep = compare(tiers, D, r["blend"], e_scores, C)
    rep["method"] = a.method
    rep["heldout_source"] = Path(a.export).name
    rep["frozen_fit_source"] = art["fit_source"]
    rep["note"] = ("HELD-OUT run: every transform (ECDFs, TF-IDF, logits, cuts, "
                   "evaluator yardstick) frozen from the fit chunk. Report these "
                   "numbers separately from public-split numbers.")
    Path("results").mkdir(exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"held-out exact {rep['exact_agreement']:.1%}  adjacent "
          f"{rep['adjacent_agreement']:.1%}  served {rep['served']:.1%}  "
          f"savings {rep['est_savings_pct']}%")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
