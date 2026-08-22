#!/usr/bin/env python3
"""The frontier: every router configuration plotted in cost vs targeting space.

THIS IS NOT A COST-QUALITY FRONTIER, AND CALLING IT ONE WOULD BE A LIE.
    The export has no outputs and no quality labels, so no axis here measures
    whether an answer was good. The y-axis is TARGETING: what share of the work
    that actually turned out heavy was routed to a capable tier. A router can
    score 100% on it and still produce garbage.
    A real quality axis needs judge-model scores on a stratified sample - the
    challenge brief names that as expected work, and it is not done here.

    (The repo's own scripts/plot_frontier.py plots a placeholder "quality" =
    fraction of calls kept on the logged model. That is agreement with the log,
    not quality either. This script at least labels the axis for what it is.)

WHAT IS PLOTTED
    One dot per configuration: every rank->tier boundary crossed with every
    async-discount setting. Red dots breach the safety rule (more than 10 of the
    true heaviest-two-decile tasks sent to the cheapest tier); those are cheap
    for a bad reason and are excluded from the usable front.
    The grey dashed curve is the ORACLE ceiling - the best targeting buyable at
    each budget by something that already knows the answer.
    Left panel scores on the top tier alone, right panel on the top two tiers.
    They disagree, and that is the point: a tier relabel moves one and not the
    other, which is how a definition change can masquerade as an improvement.

Usage: python scripts/plot_router_frontier.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import (BASELINE_SMALL, capture, cost_of, heavy_at_cheapest,
                      oracle_at_budget, top_used)
from fit_router import K, crossfit_ranks, effort_ranks, load_rows
from router import (RANK_TO_TIER, TIERS, load_pricing, route_one, tier_of,
                    tier_price)

OUT = "results/router_frontier.png"
SAFE_MAX = 10  # heavy@t0 above this is not a usable configuration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.jsonl")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    pricing = load_pricing()
    rows = load_rows(a.features)
    n = len(rows)
    tin = np.array([r["sys_tokens"] + r["usr_tokens"] + r["tools_tokens"]
                    for r in rows], float)
    tout = np.array([r["gen_tokens"] for r in rows], float)
    logged = [r["model"] for r in rows]
    groups = np.array([r["workspace"] for r in rows])
    decile = np.floor(np.argsort(np.argsort(tout)) / n * 10).astype(int) + 1
    rank = crossfit_ranks(rows, groups, effort_ranks(tout), K)

    all_models = [m for g in TIERS for m in g]
    cheapest = min(all_models, key=lambda m: tier_price(m, pricing))
    routed_now = [route_one(r, int(k), pricing)[0] for r, k in zip(rows, rank)]
    used_top = top_used(routed_now)
    strongest = min((m for m in all_models if tier_of(m) == used_top),
                    key=lambda m: tier_price(m, pricing))
    c_log = cost_of(logged, tin, tout, pricing)
    mid_floor = max(1, used_top - 1)

    # ---------------- the configuration grid ----------------
    def mk(lo, hi):
        return {**{r: 0 for r in range(0, lo)}, **{r: 1 for r in range(lo, hi)},
                **{r: 2 for r in range(hi, K)}}

    DISCOUNTS = [((0, -1), 1, "off"),
                 ((11, 12), 1, "r11-12>t1"), ((11, 13), 1, "r11-13>t1"),
                 ((10, 15), 1, "r10-15>t1"), ((11, 15), 1, "r11-15>t1"),
                 ((7, 10), 0, "r7-10>t0"), ((0, 15), 0, "all>t0")]
    pts = []
    for lo in range(4, 11):
        for hi in range(lo + 2, min(lo + 6, K)):
            for dr, fl, dtag in DISCOUNTS:
                ms = [route_one(r, int(k), pricing, True, mk(lo, hi), dr, fl)[0]
                      for r, k in zip(rows, rank)]
                pts.append(dict(cost=cost_of(ms, tin, tout, pricing),
                                cap_top=capture(ms, tout, used_top),
                                cap_mid=capture(ms, tout, mid_floor),
                                heavy=heavy_at_cheapest(ms, decile),
                                tag="{}-{}/{}".format(lo, hi - 1, dtag)))
    print("swept {} configurations".format(len(pts)))

    # ---------------- oracle ceiling ----------------
    lo_cost = cost_of([cheapest] * n, tin, tout, pricing)
    hi_cost = cost_of([strongest] * n, tin, tout, pricing)
    orc = []
    for b in np.linspace(lo_cost, hi_cost, 40):
        ms, spent = oracle_at_budget(tout, tin, b, pricing, cheapest, strongest)
        orc.append((spent, capture(ms, tout, used_top), capture(ms, tout, mid_floor)))
    orc = np.array(orc)

    baseline = [(("claude-sonnet-5" if m.startswith("claude") else cheapest)
                 if t < BASELINE_SMALL else m) for m, t in zip(logged, tin)]
    REFS = [("logged", logged, "#555555", "o"),
            ("baseline_router", baseline, "#0369a1", "s"),
            ("all -> " + cheapest, [cheapest] * n, "#dc2626", "X"),
            ("router.py (current)", routed_now, "#16a34a", "*")]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    for ax, key, floor, label in [(axes[0], "cap_top", used_top, "top tier only"),
                                  (axes[1], "cap_mid", mid_floor, "top two tiers")]:
        safe = [p for p in pts if p["heavy"] <= SAFE_MAX]
        risky = [p for p in pts if p["heavy"] > SAFE_MAX]
        ax.scatter([p["cost"] for p in risky], [p[key] for p in risky],
                   s=26, color="#dc2626", alpha=0.40, linewidths=0,
                   label="breaches safety (>{} heavy on cheapest)".format(SAFE_MAX))
        ax.scatter([p["cost"] for p in safe], [p[key] for p in safe],
                   s=30, color="#94a3b8", alpha=0.75, linewidths=0,
                   label="usable configurations")
        sp = sorted(safe, key=lambda p: p["cost"])
        front, best = [], -1
        for p in sp:
            if p[key] > best:
                front.append(p)
                best = p[key]
        ax.plot([p["cost"] for p in front], [p[key] for p in front],
                color="#0f766e", lw=2.2, label="usable Pareto front")
        ax.plot(orc[:, 0], orc[:, 1 if key == "cap_top" else 2], "--",
                color="#888", lw=1.8, label="oracle ceiling (knows the answer)")
        for nm, ms, colr, mrk in REFS:
            x = cost_of(ms, tin, tout, pricing)
            y = capture(ms, tout, floor)
            ax.scatter([x], [y], s=260 if mrk == "*" else 130, color=colr,
                       marker=mrk, zorder=6, edgecolor="white", linewidth=1.4)
            ax.annotate(nm, (x, y), textcoords="offset points", xytext=(10, -13),
                        fontsize=8.6, color=colr, fontweight="bold")
        ax.axvline(c_log, color="#555", ls=":", lw=1.2)
        ax.set_xlabel("estimated cost, USD  (assumed prices, est. tokens)", fontsize=9)
        ax.set_ylabel("share of heavy work on a capable tier", fontsize=9)
        ax.set_title("scored on the {}".format(label), fontsize=11,
                     fontweight="bold", loc="left")
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.05, 1.03)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Router frontier - {} configurations over {} real tasks. "
                 "TARGETING, not quality.".format(len(pts), n),
                 fontsize=13, fontweight="bold", y=0.975)
    fig.text(0.5, 0.045,
             "The y-axis is NOT quality. The export has no outputs and no quality labels, so nothing here shows a "
             "cheap model would have answered well -",
             ha="center", fontsize=8.4, color="#666")
    fig.text(0.5, 0.022,
             "it measures whether effort went where the work turned out to be. Tokens estimated (chars/4); "
             "prices and tier order assumed; token counts held fixed across policies.",
             ha="center", fontsize=8.4, color="#666")
    fig.tight_layout(rect=[0, 0.075, 1, 0.945])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote {}".format(a.out))

    cur_cost = cost_of(routed_now, tin, tout, pricing)
    cur_cap = capture(routed_now, tout, used_top)
    print("current: ${:.2f}  cap_top {:.1%}".format(cur_cost, cur_cap))
    dom = [p for p in pts if p["heavy"] <= SAFE_MAX
           and p["cost"] <= cur_cost and p["cap_top"] > cur_cap]
    if dom:
        b = max(dom, key=lambda p: p["cap_top"])
        print("DOMINATED: {} is ${:.2f} with cap_top {:.1%} (heavy@t0 {})"
              .format(b["tag"], b["cost"], b["cap_top"], b["heavy"]))
    else:
        print("current config is on the usable Pareto front")


if __name__ == "__main__":
    main()
