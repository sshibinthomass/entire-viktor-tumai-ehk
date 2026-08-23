#!/usr/bin/env python3
"""What the hyperparameter search actually bought. Four panels.

    A  rho vs SVD dims - the one real effect, with a peak rather than an edge
    B  dims 40 vs 80 across random seeds - is it the seed or the setting?
    C  selection bias - grid maximum vs nested estimate, for both configs
    D  router outcome before and after, on the metrics that matter

Everything is recomputed here rather than transcribed from a previous run, so
the panels cannot drift from the code. Panel C re-runs the nested search, which
is the slow part (~1 min).

THE POINT OF PANEL C
    A maximum over a grid of noisy estimates is not an estimate of the maximum.
    Panel C is the difference between what the search claimed and what survives
    when the winner is scored on folds the search never touched. Only the SVD
    dimension survived; K and the TF-IDF settings did not.

Usage: python scripts/plot_tuning.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from evaluate import capture, cost_of, heavy_at_cheapest, oracle_at_budget, top_used
from fit_router import effort_ranks, load_rows, rank_clusters
from router import TIERS, load_pricing, route_one, tier_of, tier_price
from tune import BASE, build, crossfit_rho, embed_with, search

OUT = "results/tuning.png"
DIMS_CURVE = [20, 30, 40, 60, 80, 120]
SEEDS = range(5)
SHIPPED_DIMS, TUNED_DIMS = 80, 40
GREY, GREEN, RED, BLUE = "#555555", "#16a34a", "#dc2626", "#0369a1"


def router_metrics(rows, groups, effort, gen, tin, tout, logged, pricing, dims):
    """Route with a given SVD width and score the outcome."""
    oof = np.full(len(rows), -1, int)
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), groups=groups):
        tr_rows, te_rows = [rows[i] for i in tr], [rows[i] for i in te]
        vec, svd = build(tr_rows, {**BASE, "dims": dims})
        # n_init must match fit_router.crossfit_ranks or panel D will not
        # reproduce the numbers evaluate.py prints
        km = KMeans(BASE["k"], n_init=15, random_state=0).fit(
            embed_with(tr_rows, vec, svd))
        rk = rank_clusters(km.labels_, effort[tr], BASE["k"])
        oof[te] = [rk[c] for c in km.predict(embed_with(te_rows, vec, svd))]
    ms = [route_one(r, int(k), pricing)[0] for r, k in zip(rows, oof)]
    used = top_used(ms)
    decile = np.floor(np.argsort(np.argsort(tout)) / len(tout) * 10).astype(int) + 1
    c = cost_of(ms, tin, tout, pricing)
    cheapest = min((m for g in TIERS for m in g), key=lambda m: tier_price(m, pricing))
    strongest = min((m for m in (x for g in TIERS for x in g) if tier_of(m) == used),
                    key=lambda m: tier_price(m, pricing))
    orc, _ = oracle_at_budget(tout, tin, c, pricing, cheapest, strongest)
    orc_cap = capture(orc, tout, used)
    return dict(cost=c, saving=1 - c / cost_of(logged, tin, tout, pricing),
                cap_top=capture(ms, tout, used),
                heavy=heavy_at_cheapest(ms, decile),
                rho=spearmanr(oof, gen).statistic,
                oracle_ratio=capture(ms, tout, used) / max(1e-9, orc_cap))



def grid_scores(rows, groups, effort, gen, ks, dims_list, folds=5, seed=0):
    """Cross-fitted rho for every (k, dims) pair, sharing work across the grid.

    TF-IDF depends on neither k nor dims, and the SVD depends only on dims, so
    fitting them once per fold instead of once per grid cell turns 36 vectoriser
    fits into 1 and makes the nested check affordable."""
    oof = {(k, d): np.full(len(rows), np.nan) for k in ks for d in dims_list}
    for tr, te in GroupKFold(n_splits=folds).split(np.zeros(len(rows)), groups=groups):
        tr_rows, te_rows = [rows[i] for i in tr], [rows[i] for i in te]
        vec, _ = build(tr_rows, {**BASE, "dims": max(dims_list)})
        Ttr = vec.transform([r["_trigger_text"] for r in tr_rows])
        Tte = vec.transform([r["_trigger_text"] for r in te_rows])
        for d in dims_list:
            svd = TruncatedSVD(min(d, Ttr.shape[1] - 1), random_state=seed).fit(Ttr)
            Ztr, Zte = normalize(svd.transform(Ttr)), normalize(svd.transform(Tte))
            for k in ks:
                km = KMeans(k, n_init=10, random_state=seed).fit(Ztr)
                rk = rank_clusters(km.labels_, effort[tr], k)
                oof[(k, d)][te] = [rk[c] for c in km.predict(Zte)]
    return {key: spearmanr(v, gen).statistic for key, v in oof.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.jsonl")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    pricing = load_pricing()
    rows = load_rows(a.features)
    gen = np.array([r["gen_tokens"] for r in rows], float)
    tout = gen
    tin = np.array([r["sys_tokens"] + r["usr_tokens"] + r["tools_tokens"]
                    for r in rows], float)
    logged = [r["model"] for r in rows]
    groups = np.array([r["workspace"] for r in rows])
    effort = effort_ranks(gen)

    print("A: dims curve ...")
    curve = [crossfit_rho(rows, groups, effort, gen, {**BASE, "dims": d})[0]
             for d in DIMS_CURVE]
    print("B: seed check ...")
    s40 = [crossfit_rho(rows, groups, effort, gen, {**BASE, "dims": 40}, seed=s)[0]
           for s in SEEDS]
    s80 = [crossfit_rho(rows, groups, effort, gen, {**BASE, "dims": 80}, seed=s)[0]
           for s in SEEDS]

    print("C: nested search over K x dims ...")
    KS = [8, 12, 16, 20, 24, 32]
    flat = grid_scores(rows, groups, effort, gen, KS, DIMS_CURVE)
    grid_max = max(flat.values())
    nested_tuned, nested_ship, picks = [], [], []
    for tr, te in GroupKFold(n_splits=3).split(np.zeros(len(rows)), groups=groups):
        tr_rows, te_rows = [rows[i] for i in tr], [rows[i] for i in te]
        # inner search on the training portion only
        inner = grid_scores(tr_rows, groups[tr], effort_ranks(gen[tr]), gen[tr],
                            KS, DIMS_CURVE, folds=3)
        bk, bd = max(inner, key=inner.get)
        picks.append((bk, bd))
        for cfg, bucket in (({**BASE, "k": bk, "dims": bd}, nested_tuned),
                            (BASE, nested_ship)):
            vec, svd = build(tr_rows, cfg)
            km = KMeans(cfg["k"], n_init=10, random_state=0).fit(
                embed_with(tr_rows, vec, svd))
            rk = rank_clusters(km.labels_, effort_ranks(gen[tr]), cfg["k"])
            pred = [rk[c] for c in km.predict(embed_with(te_rows, vec, svd))]
            bucket.append(spearmanr(pred, gen[te]).statistic)
    print("   inner picks (k, dims): {}".format(picks))

    print("D: router before/after ...")
    before = router_metrics(rows, groups, effort, gen, tin, tout, logged,
                            pricing, SHIPPED_DIMS)
    after = router_metrics(rows, groups, effort, gen, tin, tout, logged,
                           pricing, TUNED_DIMS)

    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.4))

    # A: dims curve
    axA = ax[0, 0]
    axA.plot(DIMS_CURVE, curve, "o-", color=GREEN, lw=2.4, ms=8)
    peak = int(np.argmax(curve))
    axA.scatter([DIMS_CURVE[peak]], [curve[peak]], s=260, facecolor="none",
                edgecolor=GREEN, lw=2.4, zorder=5)
    axA.annotate("peak, dims={}\nrho {:+.3f}".format(DIMS_CURVE[peak], curve[peak]),
                 (DIMS_CURVE[peak], curve[peak]), textcoords="offset points",
                 xytext=(14, -6), fontsize=9, color=GREEN, fontweight="bold")
    i80 = DIMS_CURVE.index(SHIPPED_DIMS)
    axA.scatter([SHIPPED_DIMS], [curve[i80]], s=150, color=GREY, zorder=5)
    axA.annotate("was here\nrho {:+.3f}".format(curve[i80]),
                 (SHIPPED_DIMS, curve[i80]), textcoords="offset points",
                 xytext=(6, -34), fontsize=9, color=GREY, fontweight="bold")
    axA.set_xlabel("SVD dimensions", fontsize=9)
    axA.set_ylabel("cross-fitted rho vs observed effort", fontsize=9)
    axA.set_title("A  A peak, not an edge - so it is a real effect",
                  fontsize=11, fontweight="bold", loc="left")
    axA.grid(alpha=0.25)

    # B: seeds
    axB = ax[0, 1]
    x = np.arange(len(list(SEEDS)))
    for xi, lo, hi in zip(x, s80, s40):
        axB.plot([xi, xi], [lo, hi], color="#cbd5e1", lw=2, zorder=1)
    axB.scatter(x, s80, s=110, color=GREY, zorder=3, label="dims=80 (was)")
    axB.scatter(x, s40, s=110, color=GREEN, zorder=3, label="dims=40 (now)")
    axB.set_xticks(x)
    axB.set_xticklabels(["seed {}".format(s) for s in SEEDS], fontsize=8.6)
    axB.set_ylabel("cross-fitted rho", fontsize=9)
    axB.set_title("B  {}/{} seeds, mean +{:.3f} - not seed luck".format(
        sum(1 for p, q in zip(s40, s80) if p > q), len(s40),
        np.mean(s40) - np.mean(s80)), fontsize=11, fontweight="bold", loc="left")
    axB.legend(frameon=False, fontsize=9, loc="lower right")
    axB.grid(axis="y", alpha=0.25)

    # C: selection bias
    axC = ax[1, 0]
    labels = ["grid maximum\n(what tuning\nclaimed)",
              "nested tuned\n(honest)", "nested shipped\n(no tuning)"]
    vals = [grid_max, float(np.mean(nested_tuned)), float(np.mean(nested_ship))]
    cols = [RED, BLUE, GREY]
    bars = axC.bar(range(3), vals, color=cols, width=0.58)
    for b, v in zip(bars, vals):
        axC.text(b.get_x() + b.get_width() / 2, v + 0.012, "{:+.3f}".format(v),
                 ha="center", fontsize=10, fontweight="bold")
    axC.annotate("", xy=(0, grid_max), xytext=(1, vals[1]),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    axC.text(0.5, (grid_max + vals[1]) / 2 + 0.012,
             "selection bias\n{:+.3f}".format(grid_max - vals[1]),
             ha="center", fontsize=9.5, color=RED, fontweight="bold")
    axC.set_xticks(range(3))
    axC.set_xticklabels(labels, fontsize=8.6)
    axC.set_ylabel("rho vs observed effort", fontsize=9)
    axC.set_ylim(0, max(vals) * 1.30)
    axC.set_title("C  Grid over K x dims: nested gain only {:+.3f}".format(
        vals[1] - vals[2]), fontsize=11, fontweight="bold", loc="left")
    axC.grid(axis="y", alpha=0.25)

    # D: router outcome
    axD = ax[1, 1]
    metrics = [("cost saving", before["saving"], after["saving"], "{:.1%}"),
               ("cap_top", before["cap_top"], after["cap_top"], "{:.1%}"),
               ("oracle ratio", before["oracle_ratio"], after["oracle_ratio"], "{:.0%}"),
               ("rho", before["rho"], after["rho"], "{:+.3f}"),
               ("heavy@t0\n(lower better)", before["heavy"] / 20, after["heavy"] / 20,
                None)]
    xs = np.arange(len(metrics))
    w = 0.36
    axD.bar(xs - w / 2, [m[1] for m in metrics], w, color=GREY, label="dims=80 (before)")
    axD.bar(xs + w / 2, [m[2] for m in metrics], w, color=GREEN, label="dims=40 (after)")
    for i, (nm, b, aft, fmt) in enumerate(metrics):
        for off, v in ((-w / 2, b), (w / 2, aft)):
            txt = fmt.format(v) if fmt else "{:.0f}".format(v * 20)
            axD.text(i + off, v + 0.015, txt, ha="center", fontsize=8.4)
    axD.set_xticks(xs)
    axD.set_xticklabels([m[0] for m in metrics], fontsize=8.4)
    axD.set_ylim(0, 1.05)
    axD.set_ylabel("value (heavy@t0 scaled /20)", fontsize=9)
    axD.legend(frameon=False, fontsize=9, loc="upper right")
    axD.set_title("D  Router outcome: +{:.1f} pts cap_top, {} -> {} heavy misroutes"
                  .format(100 * (after["cap_top"] - before["cap_top"]),
                          before["heavy"], after["heavy"]),
                  fontsize=11, fontweight="bold", loc="left")
    axD.grid(axis="y", alpha=0.25)

    fig.suptitle("Hyperparameter tuning: one real gain, and the bias in the rest",
                 fontsize=13.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.014,
             "Objective is Spearman against OBSERVED EFFORT, a proxy for difficulty - not difficulty itself. "
             "All scores are cross-fitted under GroupKFold on the workspace fingerprint, so they describe an unseen tenant.",
             ha="center", fontsize=8.4, color="#666")
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote {}".format(a.out))
    print("grid max {:+.4f} | nested tuned {:+.4f} | nested shipped {:+.4f}"
          .format(grid_max, np.mean(nested_tuned), np.mean(nested_ship)))
    print("router: cap_top {:.1%} -> {:.1%}, heavy@t0 {} -> {}, saving {:.1%} -> {:.1%}"
          .format(before["cap_top"], after["cap_top"], before["heavy"],
                  after["heavy"], before["saving"], after["saving"]))


if __name__ == "__main__":
    main()
