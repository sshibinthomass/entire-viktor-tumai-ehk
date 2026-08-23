#!/usr/bin/env python3
"""The cache trap: naive costing overstates per-call switching savings.

The deck's twist: providers cache the shared input prefix across a task's
calls, and a MODEL SWITCH RESETS THAT CACHE — the first call after a switch
pays the uncached rate for the whole prefix. A per-call router therefore looks
great under naive costing (every shared prefix billed cheap) and much worse
once resets are priced.

TWO layers, clearly separated (they were previously blurred):

  MEASURED (trajectories with >=2 logged calls): on this subset the cheap-
  tool-loops policy produces ~zero switches — the export samples ~1 call per
  task and consecutive logged calls are all mechanical continuations, so the
  planning<->execution switch points fall BETWEEN the samples. The measured
  overstatement is a floor, not the effect. A chunk with NO multi-call
  trajectory (v1_01 and v1_02 are both one-call-per-task) has no call pair to
  measure at all: the layer is then reported unavailable, not faked.

  MODELED (all trajectories): an extrapolation under three stated assumptions:
    (a) a mid-task switch re-pays ~T/2 uncached-vs-cached (T = final context)
    (b) a cheap-tool-loops policy switches ~2x per user/planning turn,
        capped by the call count
    (c) entries alternate between T1 and T3, so the per-token rate delta is
        the average of the two tiers' uncached-minus-cached rates
  Every number derived from this is labeled MODELED, never 'demonstrated'.

Trigger-item note (checked against the export, not assumed): every logged
call's input ends with an assistant message — input[-1] is role=assistant in
47/47 later calls — so the item that says what TRIGGERED the call is
input[-2] (function_call_output = mechanical continuation vs message =
planning turn).

Writes results/cache_trap.json.

Usage: python cache_trap.py
"""
import json
from pathlib import Path

import numpy as np

from run_pipeline import TIER_PRICES, TIER3_FABLE, DEFAULT_EXPORT


def per_call_token_profile(calls):
    """[(input_tokens, tokens_shared_with_previous_call)] — items serialized
    ONCE per call (the old version re-serialized the whole growing input for
    every call and again per policy/pricing combination)."""
    prev_ser = None
    out = []
    for c in calls:
        ser = [json.dumps(it) for it in c["input"]]
        toks = [len(s) // 4 for s in ser]
        inp = sum(toks) // 1  # per-item sum (close to whole-blob chars/4)
        shared = 0
        if prev_ser is not None:
            for a, b, t in zip(prev_ser, ser, toks):
                if a == b:
                    shared += t
                else:
                    break
        out.append((inp, min(shared, inp)))
        prev_ser = ser
    return out


def price(profile, route, cache_aware):
    """$ for the logged calls of one trajectory under a tier route."""
    usd = 0.0
    for i, (inp, shared) in enumerate(profile):
        cached = shared if i > 0 else 0
        if cache_aware and i > 0 and route[i] != route[i - 1]:
            cached = 0  # the switch reset the cache
        pin, pc, _ = TIER_PRICES[route[i]]
        usd += ((inp - cached) * pin + cached * pc) / 1e6
    return usd


def main():
    task_tier = {json.loads(l)["trajectory_id"]: json.loads(l)["router_tier"]
                 for l in open("results/tiers.jsonl", encoding="utf-8")}

    # regroup ALL calls per trajectory (load_trajectories keeps only first/deepest)
    p = Path(DEFAULT_EXPORT)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    calls_of = {}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    req = json.loads(line)
                    calls_of.setdefault(req["trajectory_id"], []).append(req)
    calls_of = {tid: sorted(v, key=lambda r: r.get("call_index", 0))
                for tid, v in calls_of.items() if len(v) >= 2}
    print(f"trajectories with >=2 logged calls: {len(calls_of)} "
          f"(total logged calls {sum(len(v) for v in calls_of.values())})")

    out = {"n_trajectories": len(calls_of), "policies": {}}
    g = measured_layer(calls_of, task_tier, out) if calls_of else None
    if g is None:
        out["measured_available"] = False
        out["measured_note"] = (
            "MEASURED layer unavailable: no trajectory in this export has >=2 logged "
            "calls, so there are no intra-task call pairs whose prefix overlap could "
            "be measured. Chunks that ship one sampled call per task (v1_01, v1_02) "
            "are like this; v1_00 ships whole trajectories. The MODELED layer below "
            "needs only the per-trajectory metrics and still runs."
        )
        print(out["measured_note"])
    modeled_layer(out, g)


def measured_layer(calls_of, task_tier, out):
    """Price the per-call policies on the trajectories that DO have >=2 logged
    calls. Returns the cheap-tool-loops record (the one the correction uses)."""
    profiles = {tid: per_call_token_profile(v) for tid, v in calls_of.items()}

    # greedy per-call thresholds from the input-size distribution
    sizes = [inp for prof in profiles.values() for inp, _ in prof]
    q33, q66 = np.quantile(sizes, [1 / 3, 2 / 3])

    def greedy_route(tid, calls):
        return [1 if inp <= q33 else 2 if inp <= q66 else 3
                for inp, _ in profiles[tid]]

    def alternating_route(tid, calls):
        """The per-call policy people actually propose: cheap model for
        mechanical tool-loop continuations, strong model for planning turns.
        input[-1] is always the assistant's reply (verified above), so
        input[-2] is the trigger item."""
        out = []
        for c in calls:
            prev = c["input"][-2] if len(c["input"]) > 1 else {}
            mech = prev.get("type") in ("function_call_output", "custom_tool_call_output")
            out.append(1 if mech else 3)
        return out

    policies = {
        "all-Tier3": lambda tid, calls: [3] * len(calls),
        "per-task (our router)": lambda tid, calls: [task_tier[tid]] * len(calls),
        "per-call greedy (size)": greedy_route,
        "per-call cheap-tool-loops": alternating_route,
    }
    top_naive = sum(price(p, [3] * len(p), False) for p in profiles.values())
    top_aware = sum(price(p, [3] * len(p), True) for p in profiles.values())
    for name, pol in policies.items():
        naive = aware = 0.0
        switches = 0
        for tid, calls in calls_of.items():
            r = pol(tid, calls)
            naive += price(profiles[tid], r, False)
            aware += price(profiles[tid], r, True)
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
    out["measured_overstatement_pct_points"] = round(
        g["savings_naive_pct"] - g["savings_cache_aware_pct"], 1)
    print(f"\nMEASURED subset: overstatement {out['measured_overstatement_pct_points']} pts "
          f"with {g['switches']} switches — the sampling hides switch points, so this "
          f"is a floor, not the effect.")
    out["measured_available"] = True
    return g


def modeled_layer(out, g):
    """The extrapolation (assumptions a/b/c in the docstring). Needs only the
    per-trajectory metrics, so it runs on one-call-per-task chunks too; the
    naive-claim correction needs the measured layer and is skipped without it."""
    # ---------------- MODELED extrapolation (assumptions a/b/c in the docstring)
    mets = [json.loads(l) for l in open("results/evaluator_metrics.jsonl",
                                        encoding="utf-8")]

    def modeled(tier3_prices):
        d1 = TIER_PRICES[1][0] - TIER_PRICES[1][1]
        d3 = tier3_prices[0] - tier3_prices[1]
        delta = (d1 + d3) / 2
        per_switch = np.array([m["context_tokens"] / 2 * delta / 1e6 for m in mets])
        n_calls = np.array([m["n_llm_calls"] for m in mets])
        est_switches = np.minimum(2 * np.array([m["n_user_turns"] for m in mets]),
                                  np.maximum(n_calls - 1, 0))
        penalty = float((per_switch * est_switches).sum())
        # all-Tier3 INPUT-ONLY budget under the whole-trajectory model:
        # first pass uncached + linear-growth replay at the cached rate
        first_pass = sum(m["context_tokens"] * tier3_prices[0] for m in mets) / 1e6
        replay = sum(m["context_tokens"] * (max(m["n_llm_calls"], 1) - 1) / 2
                     * tier3_prices[1] for m in mets) / 1e6
        return penalty, first_pass, replay

    penalty, first_pass, replay = modeled(TIER_PRICES[3])
    input_budget = first_pass + replay
    naive_claim = g["savings_naive_pct"] if g else None
    corrected = None if g is None else \
        round(naive_claim - 100 * penalty / input_budget, 1)
    out["modeled_full_trajectory"] = {
        "label": "MODELED, not measured — extrapolation under the three stated assumptions",
        "assumptions": [
            "a switch re-pays ~T/2 uncached-vs-cached (T = final context tokens)",
            "cheap-tool-loops switches ~2x per user/planning turn, capped by call count",
            "rate delta averaged over the two tiers entered (T1 and T3)",
            "input side only; token counts are chars/4 estimates",
        ],
        "cache_reset_penalty_usd": round(penalty, 2),
        "all_tier3_input_budget_usd": round(input_budget, 2),
        "all_tier3_budget_decomposition": {
            "first_pass_input_usd": round(first_pass, 2),
            "cached_replay_usd": round(replay, 2),
            "note": "input-only — output cost is NOT in this denominator",
        },
        "penalty_share_of_input_budget_pct": round(100 * penalty / input_budget, 1),
        "naive_claimed_savings_pct": naive_claim,
        "corrected_savings_pct": corrected,
    }
    # tier-price sensitivity: same computation with fable-5-priced Tier 3
    pen_f, fp_f, rp_f = modeled(TIER3_FABLE)
    out["modeled_full_trajectory"]["tier3_price_sensitivity"] = {
        "opus_priced": {"penalty_usd": round(penalty, 2),
                        "input_budget_usd": round(input_budget, 2),
                        "penalty_share_pct": round(100 * penalty / input_budget, 1)},
        "fable_priced": {"penalty_usd": round(pen_f, 2),
                         "input_budget_usd": round(fp_f + rp_f, 2),
                         "penalty_share_pct": round(100 * pen_f / (fp_f + rp_f), 1)},
    }
    print(f"\nMODELED (all {len(mets)} tasks, input-only, assumptions stated): the same "
          f"policy pays ~${penalty:,.0f} in cache resets vs a ${input_budget:,.0f} "
          f"all-Tier3 INPUT budget ({100 * penalty / input_budget:.0f}% of it).")
    if g:
        print(f"Its 'naive {naive_claim:.0f}% saved' corrects to ~{corrected:.0f}% saved "
              f"once resets are priced — per-task routing pays zero resets by construction.")
    else:
        print("(no measured per-call claim on this export to correct — the naive-vs-"
              "corrected pair needs trajectories with >=2 logged calls.)")
    print(f"(fable-priced Tier 3: penalty ${pen_f:,.0f} on a ${fp_f + rp_f:,.0f} budget)")
    with open("results/cache_trap.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results/cache_trap.json")


if __name__ == "__main__":
    main()
