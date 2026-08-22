#!/usr/bin/env python3
"""Evaluate the router against the log. Every reported number lives here.

Reports, in order:
    1  policy comparison        cost, BOTH capture definitions, and safety
    2  oracle ceiling           best capture buyable at the router's own cost
    3  demand quality           cross-fitted cluster rank, unseen workspaces
    4  rank -> tier sweep       what the mapping and discount cap cost
    5  break-even               how much extra output kills the saving
    6  price sensitivity        do the conclusions survive a different sheet?
    7  where the money is       input vs output, and prompt composition
    8  cache headroom           what a stable prefix would be worth
    9  per-rank                 pooled numbers can hide a loss
   10  bootstrap intervals      point estimates are not results

TWO CAPTURE DEFINITIONS, BOTH REPORTED
    capture_top  share of observed generated tokens on the TOP tier only
    capture_mid  share on the top tier OR the one below
    These move in opposite directions when tier boundaries change, and quoting
    only one is how a tier edit can look like a routing improvement. An earlier
    version reported capture_top alone; a tier reshuffle then made the logged
    baseline drop 61% -> 21% with no change to any routing decision.

SAFETY IS A FIRST-CLASS METRIC
    heavy@cheapest = tasks in the true heaviest two deciles routed to the
    cheapest tier. Cost alone always prefers routing everything to the cheapest
    model, so a cost table without this column cannot be read.

HONEST BASIS - applies to every dollar figure below
    Tokens are estimated as chars/4; the export has no `usage` field.
    Prices in pricing.json are assumed; the model ids are anonymized (section 6
    tests how much that matters). Token counts are held FIXED across policies
    and only the price is swapped - section 5 bounds that.
    There are no outputs and no quality labels, so capture is a TARGETING
    measure. Nothing here shows a cheap model would have answered well.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --bootstrap 2000
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

from fit_router import K, crossfit_ranks, display_band, effort_ranks, load_rows
from router import (DISCOUNT_FLOOR_TIER, DISCOUNT_RANKS, RANK_TO_TIER, TIERS,
                    latency_class, load_pricing, price_of, route_one, tier_of,
                    tier_price)

BASELINE_SMALL = 15_000  # baseline_router.py's threshold
TOP = len(TIERS) - 1     # top tier index


def cost_of(models, tin, tout, pricing):
    return sum((i * price_of(m, pricing)[0] + o * price_of(m, pricing)[2]) / 1e6
               for m, i, o in zip(models, tin, tout))


def top_used(models):
    """Highest tier this policy actually selects. TIERS may define a rung that
    nothing maps to (fable-5), and 'capture on the top tier' must mean the top
    tier IN USE or every policy scores 0."""
    return max((tier_of(m) for m in models), default=0)


def capture(models, tout, floor):
    m = np.array([tier_of(x) >= floor for x in models])
    return tout[m].sum() / tout.sum() if tout.sum() else 0.0


def heavy_at_cheapest(models, decile):
    """Tasks in the true heaviest two deciles routed to tier 0."""
    return int(((decile >= 9) & np.array([tier_of(m) == 0 for m in models])).sum())


def describe(mapping):
    """Label a rank->tier mapping from its own contents, so a label can never go
    stale when the mapping is edited. Bit me once already."""
    bounds, prev = [], None
    for r in sorted(mapping):
        if mapping[r] != prev:
            bounds.append((r, mapping[r]))
            prev = mapping[r]
    return "/".join("{}+->t{}".format(r, t) for r, t in bounds)


def oracle_at_budget(tout, tin, budget, pricing, cheap, strong):
    """Max capture buyable for `budget`, knowing true effort. Greedy: upgrade the
    heaviest tasks first until the money runs out. This is the ceiling any
    router is competing against at that price."""
    order = np.argsort(-tout)
    models = [cheap] * len(tout)
    spent = cost_of(models, tin, tout, pricing)
    for i in order:
        trial = spent \
            - (tin[i] * price_of(cheap, pricing)[0] + tout[i] * price_of(cheap, pricing)[2]) / 1e6 \
            + (tin[i] * price_of(strong, pricing)[0] + tout[i] * price_of(strong, pricing)[2]) / 1e6
        if trial > budget:
            continue
        models[i] = strong
        spent = trial
    return models, spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.jsonl")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
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
    effort = effort_ranks(tout)
    decile = np.floor(np.argsort(np.argsort(tout)) / n * 10).astype(int) + 1

    # Cross-fitted cluster ranks: the ONLY basis for a reported number.
    rank = crossfit_ranks(rows, groups, effort, K, a.folds)
    routed = [route_one(r, int(k), pricing)[0] for r, k in zip(rows, rank)]
    baseline = [(("claude-sonnet-5" if m.startswith("claude") else "gpt-5.6-luna")
                 if t < BASELINE_SMALL else m) for m, t in zip(logged, tin)]
    cheapest = min((m for g in TIERS for m in g), key=lambda m: tier_price(m, pricing))
    allcheap = [cheapest] * n
    c_log = cost_of(logged, tin, tout, pricing)
    used_top = top_used(routed)

    print("tasks={}  est. input tok={:,.0f}  est. generated tok={:,.0f}"
          .format(n, tin.sum(), tout.sum()))
    print("tiers in use: {} of {} defined; top tier in use = tier {}"
          .format(used_top + 1, len(TIERS), used_top + 1))

    POLICIES = [("logged (what actually ran)", logged),
                ("baseline_router.py", baseline),
                ("all -> " + cheapest, allcheap),
                ("router.py", routed)]

    print("\n1. POLICY COMPARISON  (cross-fitted ranks, unseen workspaces)")
    print("   {:<28s} {:>9s} {:>10s} {:>10s} {:>10s} {:>9s}".format(
        "policy", "cost$", "vs logged", "cap_top", "cap_mid", "heavy@t0"))
    for name, route in POLICIES:
        c = cost_of(route, tin, tout, pricing)
        print("   {:<28s} {:9.4f} {:+10.1%} {:10.1%} {:10.1%} {:9d}".format(
            name, c, c / c_log - 1, capture(route, tout, used_top),
            capture(route, tout, max(1, used_top - 1)),
            heavy_at_cheapest(route, decile)))
    print("   cap_top = work on tier {}; cap_mid = work on tier {} or {}."
          .format(used_top + 1, used_top, used_top + 1))
    print("   heavy@t0 = tasks in the true heaviest 2 deciles sent to tier 1.")

    print("\n2. ORACLE CEILING  (what the router's own budget could have bought)")
    c_rt = cost_of(routed, tin, tout, pricing)
    strongest = min((m for g in TIERS for m in g if tier_of(m) == used_top),
                    key=lambda m: tier_price(m, pricing))
    orc, orc_cost = oracle_at_budget(tout, tin, c_rt, pricing, cheapest, strongest)
    print("   router      ${:8.4f}  cap_top {:.1%}".format(
        c_rt, capture(routed, tout, used_top)))
    print("   oracle      ${:8.4f}  cap_top {:.1%}   <- same money, knows the answer"
          .format(orc_cost, capture(orc, tout, used_top)))
    gap = capture(orc, tout, used_top) - capture(routed, tout, used_top)  # noqa
    print("   the router captures {:.0%} of what perfect foresight buys at this price"
          .format(capture(routed, tout, used_top) / max(1e-9, capture(orc, tout, used_top))))
    print("   remaining headroom: {:+.1f} points of capture".format(100 * gap))

    print("\n3. DEMAND QUALITY  (does the cluster rank track real effort?)")
    print("   spearman(rank, generated tokens) = {:+.3f}".format(
        spearmanr(rank, tout).statistic))
    print("   spearman(rank, tool calls)       = {:+.3f}".format(
        spearmanr(rank, [r["tool_calls"] for r in rows]).statistic))
    print("   {:>5s} {:>5s} {:>5s} {:>12s} {:>10s}".format(
        "rank", "band", "n", "med gen tok", "p90"))
    for k in range(K):
        m = rank == k
        if m.sum():
            print("   {:5d} {:5d} {:5d} {:12.0f} {:10.0f}".format(
                k, display_band(k, K), int(m.sum()),
                np.median(tout[m]), np.percentile(tout[m], 90)))

    print("\n4. RANK -> TIER SWEEP  (labels derived from the mappings themselves)")

    def mk(lo, hi):
        return {**{r: 0 for r in range(0, lo)}, **{r: 1 for r in range(lo, hi)},
                **{r: 2 for r in range(hi, K)}}

    OPTS = [(RANK_TO_TIER, DISCOUNT_RANKS, DISCOUNT_FLOOR_TIER, "CURRENT"),
            (RANK_TO_TIER, (0, -1), 1, "discount off"),
            (RANK_TO_TIER, (11, 13), 1, ""),
            (RANK_TO_TIER, (11, K), 1, ""),
            (RANK_TO_TIER, (7, 10), 0, "unsafe: reaches tier 0"),
            (mk(8, 12), DISCOUNT_RANKS, 1, ""),
            (mk(6, 10), DISCOUNT_RANKS, 1, "")]
    bad = [t for m, _, _, _ in OPTS for t in m.values() if t >= len(TIERS)]
    if bad:
        raise SystemExit("a mapping targets tier {} but only {} are defined"
                         .format(max(bad), len(TIERS)))
    print("   {:<26s} {:>12s} {:>9s} {:>9s} {:>9s} {:>9s} {:>8s}".format(
        "mapping", "discount", "cost$", "vs logged", "cap_top", "cap_mid", "heavy@t0"))
    for m, dr, fl, note in OPTS:
        r = [route_one(x, int(k), pricing, True, m, dr, fl)[0]
             for x, k in zip(rows, rank)]
        c = cost_of(r, tin, tout, pricing)
        tag = "off" if dr[1] < dr[0] else "r{}-{}>t{}".format(dr[0], min(dr[1], K - 1), fl)
        print("   {:<26s} {:>12s} {:9.4f} {:+10.1%} {:9.1%} {:9.1%} {:8d}  {}".format(
            describe(m), tag, c, c / c_log - 1,
            capture(r, tout, used_top), capture(r, tout, max(1, used_top - 1)),
            heavy_at_cheapest(r, decile), note))

    print("\n5. BREAK-EVEN  (fixed-token assumption is the biggest soft spot)")
    for x in (0.0, 0.5, 1.0, 2.0, 4.0):
        infl = np.array([1 + x if tier_of(m) < tier_of(l) else 1.0
                         for m, l in zip(routed, logged)])
        c = cost_of(routed, tin, tout * infl, pricing)
        print("   {:+5.0%} more output on downgraded tasks -> ${:8.4f}  ({:+.1%})"
              .format(x, c, c / c_log - 1))

    print("\n6. PRICE SENSITIVITY  (every conclusion rests on an assumed sheet)")
    print("   {:<30s} {:>10s} {:>10s}".format("price sheet", "cost$", "vs logged"))
    for label, f in [("as posted", lambda p: p),
                     ("cheap/strong gap halved", lambda p: [p[0] ** 0.5 * 2.24, p[1], p[2] ** 0.5 * 5]),
                     ("output rates doubled", lambda p: [p[0], p[1], p[2] * 2]),
                     ("input rates doubled", lambda p: [p[0] * 2, p[1] * 2, p[2]])]:
        alt = {k: (v if k == "_default" else f(v)) for k, v in pricing.items()}
        cl2 = cost_of(logged, tin, tout, alt)
        cr2 = cost_of(routed, tin, tout, alt)
        print("   {:<30s} {:10.4f} {:+10.1%}".format(label, cr2, cr2 / cl2 - 1))
    print("   the router's ADVANTAGE should survive all of these; the dollar figure")
    print("   should not be quoted without saying which sheet produced it.")

    print("\n7. WHERE THE MONEY IS")
    for name, ms in [("logged", logged), ("router", routed)]:
        ci = sum(i * price_of(m, pricing)[0] / 1e6 for m, i in zip(ms, tin))
        co = sum(o * price_of(m, pricing)[2] / 1e6 for m, o in zip(ms, tout))
        print("   {:<7s} input ${:7.2f} ({:.0f}%)   output ${:7.2f} ({:.0f}%)"
              .format(name, ci, 100 * ci / (ci + co), co, 100 * co / (ci + co)))
    print("   prompt composition - identical whatever you route to:")
    for name, v in [("system prompt", sys_t), ("first user msg", usr_t),
                    ("tool definitions", tls_t)]:
        billed = sum(i * price_of(m, pricing)[0] / 1e6 for m, i in zip(routed, v))
        print("     {:<18s} {:>11,.0f} tok  {:4.0f}% of input  ${:6.2f}"
              .format(name, v.sum(), 100 * v.sum() / tin.sum(), billed))

    print("\n8. CACHE HEADROOM  (not realisable today - see below)")
    unc = sum(i * price_of(m, pricing)[0] / 1e6 for m, i in zip(routed, tin))
    cac = sum(i * price_of(m, pricing)[1] / 1e6 for m, i in zip(routed, tin))
    print("   input uncached ${:.2f} vs cached ${:.2f}  ({:.1f}x)".format(unc, cac, unc / cac))
    print("   route cost would fall ${:.2f} -> ${:.2f}  ({:+.1%} vs logged)"
          .format(c_rt, c_rt - unc + cac, (c_rt - unc + cac) / c_log - 1))
    print("   BUT: 37 of 47 consecutive call-pairs in the export share ZERO prefix,")
    print("   because the system prompt is re-rendered between calls. Upstream fix.")

    print("\n9. PER-RANK  (pooled numbers can hide a loss)")
    by = defaultdict(list)
    for i, k in enumerate(rank):
        by[int(k)].append(i)
    print("   {:>5s} {:>5s} {:>10s} {:>10s} {:>9s} {:>9s}".format(
        "rank", "n", "logged$", "router$", "delta", "heavy@t0"))
    for k in sorted(by):
        idx = by[k]
        cl = cost_of([logged[i] for i in idx], tin[idx], tout[idx], pricing)
        cr = cost_of([routed[i] for i in idx], tin[idx], tout[idx], pricing)
        h = heavy_at_cheapest([routed[i] for i in idx], decile[idx])
        print("   {:5d} {:5d} {:10.3f} {:10.3f} {:+8.1%} {:9d}"
              .format(k, len(idx), cl, cr, cr / cl - 1 if cl else 0, h))

    print("\n10. BOOTSTRAP  ({} resamples, task-level)".format(a.bootstrap))
    rng = np.random.default_rng(0)
    ratios, caps, heavies = [], [], []
    for _ in range(a.bootstrap):
        s = rng.integers(0, n, n)
        cl = cost_of([logged[i] for i in s], tin[s], tout[s], pricing)
        cr = cost_of([routed[i] for i in s], tin[s], tout[s], pricing)
        ratios.append(cr / cl - 1)
        caps.append(capture([routed[i] for i in s], tout[s], used_top))
        heavies.append(heavy_at_cheapest([routed[i] for i in s], decile[s]))
    for label, arr, fmt in [("cost vs logged", np.array(ratios), "signed"),
                            ("cap_top", np.array(caps), "share"),
                            ("heavy@t0 count", np.array(heavies, float), "num")]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        if fmt == "signed":
            print("   {:<16s} {:+.1%}   95% CI [{:+.1%}, {:+.1%}]"
                  .format(label, arr.mean(), lo, hi))
        elif fmt == "share":
            print("   {:<16s} {:.1%}    95% CI [{:.1%}, {:.1%}]"
                  .format(label, arr.mean(), lo, hi))
        else:
            print("   {:<16s} {:.1f}     95% CI [{:.0f}, {:.0f}]"
                  .format(label, arr.mean(), lo, hi))

    print("\nlatency mix: {}".format(
        dict(Counter(latency_class(r) for r in rows).most_common())))
    print("REMINDER: estimated tokens, assumed prices, fixed token counts across")
    print("          policies, and no quality label exists in the export.")


if __name__ == "__main__":
    main()
