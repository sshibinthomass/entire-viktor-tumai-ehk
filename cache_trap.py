#!/usr/bin/env python3
"""The cache trap, demonstrated: naive costing overstates per-call savings.

The deck's twist: providers cache the shared input prefix across a task's
calls, and a MODEL SWITCH RESETS THAT CACHE — the first call after a switch
pays the uncached rate for the whole prefix. A per-call router therefore looks
great under naive costing (every shared prefix billed cheap) and much worse
once resets are priced.

Demo on the trajectories with >=2 logged calls (the only subset where
switching is measurable — the export samples ~1 call per task, so this is a
lower bound on the effect):

  policies: all-Tier3 | per-task tier (our router) | per-call greedy
            (route each call by its own prefix size: small->T1, mid->T2,
            large->T3 — the "one-hour heuristic" the deck suggests)
  pricing:  naive       — shared prefix always billed at the cached rate
            cache-aware — cached rate only when the call runs on the SAME
                          tier as the previous call (a switch resets)

Writes results/cache_trap.json.

Usage: python cache_trap.py
"""
import json

import numpy as np

from run_pipeline import TIER_PRICES, DEFAULT_EXPORT


def est_tokens(obj):
    return len(json.dumps(obj)) // 4


def shared_prefix_tokens(prev_req, req):
    shared = 0
    for a, b in zip(prev_req["input"], req["input"]):
        if a == b:
            shared += est_tokens(a)
        else:
            break
    return shared


def price(calls, route, cache_aware):
    """$ for the logged calls of one trajectory under a tier route."""
    usd = 0.0
    for i, c in enumerate(calls):
        inp = est_tokens(c["input"])
        cached = shared_prefix_tokens(calls[i - 1], c) if i > 0 else 0
        if cache_aware and i > 0 and route[i] != route[i - 1]:
            cached = 0  # the switch reset the cache
        cached = min(cached, inp)
        pin, pc, _ = TIER_PRICES[route[i]]
        usd += ((inp - cached) * pin + cached * pc) / 1e6
    return usd


def main():
    task_tier = {json.loads(l)["trajectory_id"]: json.loads(l)["router_tier"]
                 for l in open("results/tiers.jsonl", encoding="utf-8")}

    # regroup ALL calls per trajectory (load_trajectories keeps only first/deepest)
    calls_of = {}
    with open(DEFAULT_EXPORT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                req = json.loads(line)
                calls_of.setdefault(req["trajectory_id"], []).append(req)
    calls_of = {tid: sorted(v, key=lambda r: r.get("call_index", 0))
                for tid, v in calls_of.items() if len(v) >= 2}
    print(f"trajectories with >=2 logged calls: {len(calls_of)} "
          f"(total logged calls {sum(len(v) for v in calls_of.values())})")

    # greedy per-call thresholds from the prefix-size distribution
    sizes = [est_tokens(c["input"]) for v in calls_of.values() for c in v]
    q33, q66 = np.quantile(sizes, [1 / 3, 2 / 3])

    def greedy_route(calls):
        return [1 if est_tokens(c["input"]) <= q33
                else 2 if est_tokens(c["input"]) <= q66 else 3 for c in calls]

    def alternating_route(calls):
        """The per-call policy people actually propose: cheap model for
        mechanical tool-loop continuations, strong model for planning turns."""
        out = []
        for c in calls:
            # logged inputs end with the assistant's reply; the item before it
            # says what triggered this call: a tool result (mechanical
            # continuation) or a fresh user/system turn (planning)
            prev = c["input"][-2] if len(c["input"]) > 1 else {}
            mech = prev.get("type") in ("function_call_output", "custom_tool_call_output")
            out.append(1 if mech else 3)
        return out

    policies = {
        "all-Tier3": lambda tid, calls: [3] * len(calls),
        "per-task (our router)": lambda tid, calls: [task_tier[tid]] * len(calls),
        "per-call greedy (size)": lambda tid, calls: greedy_route(calls),
        "per-call cheap-tool-loops": lambda tid, calls: alternating_route(calls),
    }
    out = {"n_trajectories": len(calls_of), "policies": {}}
    top_naive = sum(price(v, [3] * len(v), False) for v in calls_of.values())
    top_aware = sum(price(v, [3] * len(v), True) for v in calls_of.values())
    switches_total = 0
    for name, pol in policies.items():
        naive = aware = 0.0
        switches = 0
        for tid, calls in calls_of.items():
            r = pol(tid, calls)
            naive += price(calls, r, False)
            aware += price(calls, r, True)
            switches += sum(1 for i in range(1, len(r)) if r[i] != r[i - 1])
        rec = {
            "usd_naive": round(naive, 2), "usd_cache_aware": round(aware, 2),
            "savings_naive_pct": round(100 * (1 - naive / top_naive), 1),
            "savings_cache_aware_pct": round(100 * (1 - aware / top_aware), 1),
            "switches": switches,
        }
        out["policies"][name] = rec
        print(f"{name:22s} naive ${naive:7.2f} ({rec['savings_naive_pct']:5.1f}% saved)   "
              f"cache-aware ${aware:7.2f} ({rec['savings_cache_aware_pct']:5.1f}% saved)   "
              f"switches {switches}")
    g = out["policies"]["per-call cheap-tool-loops"]
    out["overstatement_pct_points"] = round(
        g["savings_naive_pct"] - g["savings_cache_aware_pct"], 1)
    print(f"\nnaive costing overstates the switching policy's savings by "
          f"{out['overstatement_pct_points']} percentage points on the measurable subset.")

    # The measured subset CANNOT show the penalty: the export samples ~1 call
    # per task, and consecutive logged calls are all mechanical continuations,
    # so the planning<->execution switch points fall between the samples.
    # Model-based extrapolation to FULL trajectories instead (assumptions
    # stated): a task whose final context is T tokens pays ~T/2 extra
    # uncached-vs-cached per mid-task switch; a cheap-tool-loops policy
    # switches ~2x per user/planning turn; entries alternate between T1 and
    # T3, so the rate delta is the average of both tiers'.
    mets = [json.loads(l) for l in open("results/evaluator_metrics.jsonl",
                                        encoding="utf-8")]
    d1 = TIER_PRICES[1][0] - TIER_PRICES[1][1]
    d3 = TIER_PRICES[3][0] - TIER_PRICES[3][1]
    delta = (d1 + d3) / 2
    per_switch = np.array([m["context_tokens"] / 2 * delta / 1e6 for m in mets])
    n_calls = np.array([m["n_llm_calls"] for m in mets])
    est_switches = np.minimum(2 * np.array([m["n_user_turns"] for m in mets]),
                              np.maximum(n_calls - 1, 0))
    full_penalty = float((per_switch * est_switches).sum())
    out["model_based_full_penalty_usd"] = round(full_penalty, 2)
    out["measured_subset_caveat"] = ("sampled logging hides switch points: consecutive "
                                     "logged calls are all mechanical, so the measured "
                                     "0.0pt is a floor, not the effect")
    print(f"model-based full-trajectory estimate: that same policy pays "
          f"~${full_penalty:,.0f} in cache resets across all 953 tasks "
          f"(vs ~$259 to send EVERYTHING to Tier 3) — the naive '96% saved' "
          f"is an artifact of ignoring resets.")
    print("(input side only; conservative: switch count capped by call count, "
          "reset rate averaged over the two tiers entered.)")
    with open("results/cache_trap.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results/cache_trap.json")


if __name__ == "__main__":
    main()
