#!/usr/bin/env python3
"""Plot what evaluate.py reports. Six panels.

    A  cost vs heavy-work capture - the Pareto view, four policies
    B  per-band cost delta - where the router wins and where it loses
    C  break-even - how much extra output kills the saving
    D  where the money is - input composition vs output, plus cache headroom
    E  band calibration - does a higher band mean more work?
    F  bootstrap - the interval, not the point estimate

Recomputed with the same functions evaluate.py uses, so the chart cannot drift
from the numbers. Tokens are estimates (chars/4), prices are assumed, and token
counts are held fixed across policies - read ratios, not dollars.

Usage: python scripts/plot_evaluation.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import BASELINE_SMALL, capture_of, cost_of
from fit_router import K, crossfit_ranks, effort_ranks, load_rows
from router import load_pricing, price_of, route_one, tier_of

OUT = "results/evaluation.png"
GREY, BLUE, RED, GREEN = "#555555", "#0369a1", "#dc2626", "#16a34a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.jsonl")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()

    pricing = load_pricing()
    rows = load_rows(a.features)
    n = len(rows)
    sys_t = np.array([r["sys_tokens"] for r in rows], float)
    usr_t = np.array([r["usr_tokens"] for r in rows], float)
    tls_t = np.array([r["tools_tokens"] for r in rows], float)
    tin = sys_t + usr_t + tls_t
    tout = np.array([r["gen_tokens"] for r in rows], float)
    logged = [r["model"] for r in rows]
    groups = np.array([r["workspace"] for r in rows])

    rank = crossfit_ranks(rows, groups, effort_ranks(tout), K)
    routed = [route_one(r, int(k), pricing)[0] for r, k in zip(rows, rank)]
    baseline = [(("claude-sonnet-5" if m.startswith("claude") else "gpt-5.6-luna")
                 if t < BASELINE_SMALL else m) for m, t in zip(logged, tin)]
    allcheap = ["gpt-5.6-luna"] * n
    c_log = cost_of(logged, tin, tout, pricing)

    fig = plt.figure(figsize=(17, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26,
                          left=0.055, right=0.985, top=0.885, bottom=0.075)

    # ---------------- A: Pareto ----------------
    axA = fig.add_subplot(gs[0, 0])
    pts = [("logged\n(what ran)", logged, GREY, (10, -26)),
           ("baseline_router", baseline, BLUE, (10, 8)),
           ("all -> luna", allcheap, RED, (12, 4)),
           ("router.py", routed, GREEN, (-4, 14))]
    for name, ms, colr, off in pts:
        x, y = cost_of(ms, tin, tout, pricing), capture_of(ms, tout)[1]
        axA.scatter([x], [y], s=200, color=colr, zorder=5,
                    edgecolor="white", linewidth=1.6)
        axA.annotate("{}\n${:.0f}  {:.0%}".format(name, x, y), (x, y),
                     textcoords="offset points", xytext=off,
                     fontsize=8.8, color=colr, fontweight="bold")
    axA.set_xlim(-8, 128)
    axA.set_ylim(-0.10, 0.86)
    axA.set_xlabel("estimated cost, USD", fontsize=9)
    axA.set_ylabel("share of heavy work on a strong model", fontsize=9)
    axA.set_title("A  Up and to the left is better", fontsize=11,
                  fontweight="bold", loc="left", pad=8)
    axA.grid(alpha=0.25)

    # ---------------- B: per-band delta ----------------
    axB = fig.add_subplot(gs[0, 1])
    bands, deltas, counts = [], [], []
    for b in range(K):
        idx = np.where(rank == b)[0]
        if not len(idx):
            continue
        cl = cost_of([logged[i] for i in idx], tin[idx], tout[idx], pricing)
        cr = cost_of([routed[i] for i in idx], tin[idx], tout[idx], pricing)
        bands.append(b)
        deltas.append(cr / cl - 1 if cl else 0.0)
        counts.append(len(idx))
    cols = [GREEN if d < 0 else RED for d in deltas]
    axB.bar(bands, deltas, color=cols, width=0.74)
    axB.axhline(0, color="#333", lw=1)
    for b, d, c in zip(bands, deltas, counts):
        axB.text(b, d + (0.06 if d >= 0 else -0.12), "n={}".format(c),
                 ha="center", fontsize=7.2, color="#444")
    axB.set_xticks(bands[::2])
    axB.set_ylim(min(deltas) - 0.25, max(deltas) + 0.30)
    axB.set_xlabel("archetype rank (0 = lightest, 15 = heaviest)", fontsize=9)
    axB.set_ylabel("router cost vs logged", fontsize=9)
    axB.yaxis.set_major_formatter(lambda v, _: "{:+.0%}".format(v))
    axB.set_title("B  Wins on the light archetypes, loses on the heavy ones",
                  fontsize=11, fontweight="bold", loc="left", pad=8)
    axB.grid(axis="y", alpha=0.25)

    # ---------------- C: break-even ----------------
    axC = fig.add_subplot(gs[0, 2])
    xs = np.linspace(0, 6, 40)
    ys = []
    for x in xs:
        infl = np.array([1 + x if tier_of(m) < tier_of(l) else 1.0
                         for m, l in zip(routed, logged)])
        ys.append(cost_of(routed, tin, tout * infl, pricing))
    ys = np.array(ys)
    axC.plot(xs * 100, ys, color=GREEN, lw=2.6)
    axC.axhline(c_log, color=GREY, ls="--", lw=1.5)
    axC.annotate("logged  ${:.0f}".format(c_log), xy=(10, c_log),
                 xytext=(10, c_log + 2), fontsize=8.8, color=GREY)
    cross = np.interp(c_log, ys, xs * 100) if ys[-1] > c_log else None
    if cross is not None:
        axC.axvline(cross, color=RED, ls=":", lw=1.5)
        axC.annotate("break-even\n+{:.0f}% output".format(cross),
                     xy=(cross, ys.min()), xytext=(cross - 175, ys.min() + 4),
                     fontsize=8.8, color=RED, fontweight="bold")
    axC.set_xlabel("extra output burned by downgraded tasks", fontsize=9)
    axC.set_ylabel("estimated cost, USD", fontsize=9)
    axC.xaxis.set_major_formatter(lambda v, _: "+{:.0f}%".format(v))
    axC.set_title("C  The saving survives a lot of slop", fontsize=11,
                  fontweight="bold", loc="left", pad=8)
    axC.grid(alpha=0.25)

    # ---------------- D: where the money is ----------------
    axD = fig.add_subplot(gs[1, 0])
    def bill(ms, toks, out=False):
        j = 2 if out else 0
        return sum(t * price_of(m, pricing)[j] / 1e6 for m, t in zip(ms, toks))
    labels = ["logged", "router", "router\n+ cached prefix"]
    comp = []
    for ms in (logged, routed):
        comp.append([bill(ms, sys_t), bill(ms, usr_t), bill(ms, tls_t),
                     bill(ms, tout, out=True)])
    cached_in = sum(t * price_of(m, pricing)[1] / 1e6 for m, t in zip(routed, tin))
    comp.append([cached_in, 0, 0, bill(routed, tout, out=True)])
    comp = np.array(comp)
    parts = ["system prompt", "first user msg", "tool defs", "output"]
    shades = ["#1e3a8a", "#3b82f6", "#93c5fd", "#f59e0b"]
    bot = np.zeros(3)
    for j, (pname, shade) in enumerate(zip(parts, shades)):
        axD.bar(range(3), comp[:, j], bottom=bot, color=shade, width=0.6, label=pname)
        bot += comp[:, j]
    for i, tot in enumerate(bot):
        axD.text(i, tot + 1.5, "${:.0f}".format(tot), ha="center",
                 fontsize=9, fontweight="bold")
    axD.set_xticks(range(3))
    axD.set_xticklabels(labels, fontsize=8.6)
    axD.set_ylabel("estimated cost, USD", fontsize=9)
    axD.set_ylim(0, bot.max() * 1.20)
    axD.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper right")
    axD.set_title("D  The prompt costs more than the model choice",
                  fontsize=11, fontweight="bold", loc="left", pad=8)
    axD.grid(axis="y", alpha=0.25)

    # ---------------- E: band calibration ----------------
    axE = fig.add_subplot(gs[1, 1])
    bs, meds, ns = [], [], []
    for b in range(K):
        m = rank == b
        if m.sum():
            bs.append(b); meds.append(np.median(tout[m])); ns.append(int(m.sum()))
    cmap = plt.get_cmap("YlOrRd")
    axE.bar(bs, meds, color=[cmap(0.25 + 0.7 * b / (K - 1)) for b in bs], width=0.74)
    for b, v, c in zip(bs, meds, ns):
        axE.text(b, v * 1.04, "n={}".format(c), ha="center", fontsize=7.2, color="#444")
    axE.set_xticks(bs[::2])
    axE.set_ylim(0, max(meds) * 1.18)
    axE.set_xlabel("archetype rank (out-of-fold, unseen workspaces)", fontsize=9)
    axE.set_ylabel("median observed generated tokens", fontsize=9)
    axE.set_title("E  Higher rank really does mean more work", fontsize=11,
                  fontweight="bold", loc="left", pad=8)
    axE.grid(axis="y", alpha=0.25)

    # ---------------- F: bootstrap ----------------
    axF = fig.add_subplot(gs[1, 2])
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(a.bootstrap):
        s = rng.integers(0, n, n)
        cl = cost_of([logged[i] for i in s], tin[s], tout[s], pricing)
        cr = cost_of([routed[i] for i in s], tin[s], tout[s], pricing)
        ratios.append(cr / cl - 1)
    ratios = np.array(ratios) * 100
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    axF.hist(ratios, bins=40, color=GREEN, alpha=0.75)
    for v, style in ((lo, ":"), (hi, ":"), (ratios.mean(), "-")):
        axF.axvline(v, color="#166534", ls=style, lw=1.8)
    axF.axvline(0, color=GREY, ls="--", lw=1.4)
    axF.annotate("mean {:+.1f}%\n95% CI [{:+.1f}%, {:+.1f}%]"
                 .format(ratios.mean(), lo, hi),
                 xy=(0.03, 0.95), xycoords="axes fraction", va="top",
                 fontsize=9, fontweight="bold", color="#166534")
    axF.set_xlabel("router cost vs logged", fontsize=9)
    axF.set_ylabel("resamples", fontsize=9)
    axF.xaxis.set_major_formatter(lambda v, _: "{:+.0f}%".format(v))
    axF.set_title("F  {} bootstrap resamples".format(a.bootstrap), fontsize=11,
                  fontweight="bold", loc="left", pad=8)
    axF.grid(axis="y", alpha=0.25)

    fig.suptitle("Router evaluation - {} real tasks, cross-fitted on unseen workspaces"
                 .format(n), fontsize=13, fontweight="bold", y=0.958)
    fig.text(0.5, 0.012,
             "Tokens estimated (chars/4); prices and tier order assumed; token counts held FIXED "
             "across policies (panel C bounds that). No quality label exists in the export - "
             "'heavy work on a strong model' measures targeting, not whether output was good.",
             ha="center", fontsize=8.3, color="#666")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote {}".format(a.out))
    print("cost {:+.1%} [{:+.1f}%, {:+.1f}%]   capture {:.1%}"
          .format(cost_of(routed, tin, tout, pricing) / c_log - 1, lo, hi,
                  capture_of(routed, tout)[1]))


if __name__ == "__main__":
    main()
