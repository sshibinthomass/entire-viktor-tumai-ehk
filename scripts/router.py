#!/usr/bin/env python3
"""The router. Picks a model for a task. Reads no outcomes.

INPUT CONTRACT - four fields per task, all knowable before dispatch:
    _trigger_text   the first user message (Viktor's composed envelope)
    trig_is_dm      0/1  channel id looks like a DM
    trig_slack      0/1  trigger envelope is a Slack message
    trig_teams      0/1  trigger envelope is a Teams message

That is the whole input. Not the system prompt, not the tool list, not the
logged model, and not how much the task went on to write.

TWO SIGNALS
    cluster rank      which of the k archetypes the trigger text resembles,
                      ranked 0 (lightest) to k-1 (heaviest). Maps straight to a
                      tier - no 1-10 band in the decision path, because binning
                      16 ranks into 10 bands collapsed 6 pairs of clusters and
                      left one band holding 23% of all tasks. The artifact comes
                      from fit_router.py.
    latency class     who is waiting. 744 of 953 tasks in the export are
                      cron/system-triggered with nobody on the other end, and a
                      downgrade there needs no claim about quality - which is
                      the one claim this dataset cannot support.

NO CAPABILITY GATE, DELIBERATELY
    A vision/context gate was built and measured to be inert: every model in the
    export demonstrably served image-bearing requests, and none shows evidence
    of a smaller context window than any other - the largest single request
    seen, 204,065 tokens, was served by the CHEAPEST model. A gate that never
    fires implies a correctness guarantee that was never earned. Reinstate one
    only against a real published spec sheet.

Usage:
    python scripts/router.py                          # route results/features.jsonl
    python scripts/router.py --explain 560            # why one task routed as it did
    python scripts/router.py --no-async-discount --out results/routes_nodisc.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fit_router import (ARTIFACT, K, display_band, embed, load_artifact,
                        load_rows)

# Price rungs. Index is 0-based here; comments give the 1-based names used in
# discussion. Selection is CHEAPEST-IN-TIER, which forces one design rule:
#
#   A TIER MUST BE PRICE-HOMOGENEOUS.
#
# Any price spread inside a tier makes its expensive members unreachable. An
# earlier version spanned $2-$5 in one tier, which silently retired opus-5 - the
# single most-used model in the log (327 calls). Tiers are cost rungs; the
# boundary is where you actually start paying more.
#
# Rungs are $0.20 / $2 / $5 / $10 - steps of 10x, 2.5x, 2x.
#
# Older generations sit at the SAME price as their current sibling (opus-4-8 and
# opus-4-6 are both $5.00/$25.00, identical to opus-5), so paying for them is
# strictly worse. They stay listed as fallbacks and the tiebreak never picks them.
TIERS = [
    # tier 1 - $0.20 / $1.20
    ["gpt-5.6-luna"],
    # tier 2 - $2.00 / $10-12   -> selects sonnet-5 ($10 out beats terra's $12)
    ["claude-sonnet-5", "gpt-5.6-terra", "claude-sonnet-4-6"],
    # tier 3 - $5.00 / $25-30   -> selects opus-5 ($25 out beats sol's $30)
    ["claude-opus-5", "gpt-5.6-sol", "claude-opus-4-8", "claude-opus-4-6"],
    # tier 4 - $10.00 / $50.00
    ["claude-fable-5"],
]
# claude-sonnet-4-6 ($3.00/$15.00) has one log call and does not deserve a rung
# of its own; it sits in tier 2 as an alternate. Every priced model must appear
# somewhere - tier_of() labels an unlisted id top-tier, which silently corrupts
# the capture figure for the logged baseline.


# Cluster RANK (0 = lightest archetype .. K-1 = heaviest) -> minimum tier.
#
# Rank maps to tier DIRECTLY. An earlier version binned 16 ranks into 10 bands
# first, which collapsed 6 pairs of clusters into shared bands and left one band
# holding 23% of all tasks. Two lossy cuts where one will do.
#
# Nothing maps to tier 4 (fable-5). Measured: moving the heaviest archetypes there
# cost +102% on those tasks for ZERO change in capture - they already had a strong
# model. It is a quality bet this export has no labels to support. One edit away.
#
# Boundaries chosen by sweep (evaluate.py section 4). Measured, everything else
# held fixed, against the four-tier TIERS above:
#     0-6 / 7-10 / 11+   $78.23  -25.4%  cap_top 66.1%  cap_mid 91.8%  heavy@t0  5
#     0-7 / 8-11 / 12+   $67.56  -35.6%  cap_top 56.1%  cap_mid 90.8%  heavy@t0  5
#     0-8 / 9-12 / 13+   $61.37  -41.5%  cap_top 47.5%  cap_mid 89.2%  heavy@t0  9
#     0-5 / 6-9  / 10+   $96.49   -8.0%  cap_top 85.1%  cap_mid 91.9%  heavy@t0  5
#
# 0-6/7-10/11+ is the pick: 10 points dearer than the wider mapping but it holds
# 10 more points of cap_top, and it is the only setting where the router beats the
# logged route on cap_top (66.1% vs 61.3%) rather than trailing it.
#
# An earlier note here claimed 0-7/8-11/12+ was "free" at the same safety. That
# was measured under the previous three-tier TIERS and did not survive the retier
# - under the current table it costs 10 points of cap_top. Re-run the sweep after
# any change to TIERS; the two are coupled.
RANK_TO_TIER = {**{r: 0 for r in range(0, 7)},
                **{r: 1 for r in range(7, 11)},
                **{r: 2 for r in range(11, K)}}

# --- async discount: two knobs, both needed ---------------------------------
#
# DISCOUNT_RANKS      inclusive rank window the discount may apply to
# DISCOUNT_FLOOR_TIER the discount may never take a task BELOW this tier
#
# The floor is the important one. A single rank cap could not express the rule we
# actually want, and an earlier version set the cap to 6 while RANK_TO_TIER also
# cut at 6 - so every eligible rank already sat at tier 0, `need > 0` was false
# everywhere, and the discount fired on ZERO tasks while looking active.
#
# With FLOOR_TIER = 1 the discount is structurally unable to put anything on the
# cheapest model, so heavy@t0 cannot get worse no matter how wide the window.
# Measured, everything else fixed:
#
#   window   floor  fired   cost     vs logged  cap_top  cap_mid  heavy@t0
#   (off)      -        0   $78.23     -25.4%    66.1%    91.8%        5
#   11-12      t1     137   $67.68     -35.5%    53.3%    91.8%        5   <- default
#   11-13      t1     151   $66.05     -37.0%    49.6%    91.8%        5
#   11-15      t1     205   $56.37     -46.2%    25.8%    91.8%        5
#   7-10       t0     337   $65.07     -37.9%    66.1%    70.6%       46
#   0-15       t0     542   $43.21     -58.8%    25.8%    70.6%       46
#
# Any window reaching into ranks 7-10 with floor t0 lets heavy work fall to the
# cheapest model - heavy@t0 jumps 5 -> 46. Those rows are cheap and not defensible
# without quality labels.
#
# The default trades 10 points of cost for 13 points of cap_top, at unchanged
# cap_mid and unchanged heavy@t0. Widen the window if cost matters more than
# keeping work on the top tier; the safety metrics will not move.
DISCOUNT_RANKS = (11, 12)
DISCOUNT_FLOOR_TIER = 1

HERE = Path(__file__).resolve().parent


def load_pricing():
    return json.loads((HERE / "pricing.json").read_text())


def price_of(model, pricing):
    if model in pricing:
        return pricing[model]
    for pre in sorted(pricing, key=len, reverse=True):
        if pre != "_default" and model.startswith(pre):
            return pricing[pre]
    return pricing["_default"]


def tier_price(model, pricing):
    """Sort key for within-tier selection: input price first, then output.

    Input alone leaves exact ties (opus-5 / sol / opus-4-8 / opus-4-6 are all
    $5.00 in) and min() then returns whichever happens to be listed first. The
    output rate is the real differentiator: opus-5 $25 vs sol $30."""
    p = price_of(model, pricing)
    return (p[0], p[2])


def tier_of(model):
    for t, group in enumerate(TIERS):
        if model in group:
            return t
    return len(TIERS) - 1  # unknown id: assume it is an expensive one


def latency_class(task):
    """Who is waiting? Read off the trigger envelope."""
    if task["trig_is_dm"]:
        return "interactive_dm"
    if task["trig_slack"] or task["trig_teams"]:
        return "interactive_channel"
    return "async_cron"


def demand_ranks(tasks, artifact):
    """Archetype -> cluster rank, using the fitted artifact. No outcomes read."""
    Z = embed(tasks, artifact["vec"], artifact["svd"])
    clusters = artifact["kmeans"].predict(Z)
    rank = artifact["rank"]
    return np.array([rank[c] for c in clusters]), clusters


def route_one(task, rank, pricing, allow_async_discount=True,
              rank_to_tier=None, discount_ranks=DISCOUNT_RANKS,
              discount_floor_tier=DISCOUNT_FLOOR_TIER):
    """The decision. Returns (model, human-readable reason)."""
    rank_to_tier = rank_to_tier or RANK_TO_TIER
    need = rank_to_tier[rank]
    lat = latency_class(task)
    reason = "rank {} -> tier {}".format(rank, need)

    # The discount buys latency tolerance, not capability. The floor makes it
    # structurally unable to reach the cheapest tier, so heavy work can never
    # land there through a discount - see DISCOUNT_RANKS for the measured sweep.
    lo, hi = discount_ranks
    if (allow_async_discount and lat == "async_cron"
            and need > discount_floor_tier and lo <= rank <= hi):
        need -= 1
        reason += ", async discount -> tier {}".format(need)

    pick = min(TIERS[need], key=lambda m: tier_price(m, pricing))
    return pick, reason + ", {} -> {}".format(lat, pick)


def route_all(tasks, artifact, pricing, allow_async_discount=True,
              rank_to_tier=None):
    ranks, clusters = demand_ranks(tasks, artifact)
    k = artifact["k"]
    out = []
    for task, rank, cluster in zip(tasks, ranks, clusters):
        model, reason = route_one(task, int(rank), pricing,
                                  allow_async_discount, rank_to_tier)
        out.append({"routed_model": model, "cluster_rank": int(rank),
                    "demand_band": display_band(int(rank), k),  # display only
                    "cluster": int(cluster), "latency": latency_class(task),
                    "reason": reason})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.jsonl")
    ap.add_argument("--artifact", default=ARTIFACT)
    ap.add_argument("--out", default="results/router_routes.jsonl")
    ap.add_argument("--explain", type=int, default=None, help="trajectory_id to explain")
    ap.add_argument("--no-async-discount", action="store_true")
    a = ap.parse_args()

    if not Path(a.artifact).exists():
        raise SystemExit("no artifact at {} - run scripts/fit_router.py first"
                         .format(a.artifact))
    art = load_artifact(a.artifact)
    pricing = load_pricing()
    rows = load_rows(a.features)
    decisions = route_all(rows, art, pricing, not a.no_async_discount)

    if a.explain is not None:
        i = next((i for i, r in enumerate(rows)
                  if r["trajectory_id"] == a.explain), None)
        if i is None:
            raise SystemExit("trajectory_id {} not in {}".format(a.explain, a.features))
        d = decisions[i]
        print("task {}  logged={}  routed={}".format(
            a.explain, rows[i]["model"], d["routed_model"]))
        print("  latency     : {}".format(d["latency"]))
        print("  archetype   : c{}  {}".format(d["cluster"], art["names"][d["cluster"]]))
        print("  cluster rank: {} of {}   (display band {}/10)".format(
            d["cluster_rank"], art["k"] - 1, d["demand_band"]))
        print("  decision    : {}".format(d["reason"]))
        # outcome shown for context only - the router never saw it
        print("  (actual outcome, NOT an input: {:,} generated tokens, {} tool calls)"
              .format(rows[i]["gen_tokens"], rows[i]["tool_calls"]))
        return

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r, d in zip(rows, decisions):
            fh.write(json.dumps({"trajectory_id": r["trajectory_id"],
                                 "logged_model": r["model"], **d}) + "\n")

    print("routed {} tasks -> {}".format(len(rows), a.out))
    print("model mix  : {}".format(dict(Counter(d["routed_model"] for d in decisions).most_common())))
    print("latency mix: {}".format(dict(Counter(d["latency"] for d in decisions).most_common())))
    print("rank mix   : {}".format(dict(sorted(Counter(d["cluster_rank"] for d in decisions).items()))))
    print("\nfor cost and capture numbers: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
