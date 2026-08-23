#!/usr/bin/env python3
"""Cache-aware cost model on ESTIMATED tokens.

The export has no `usage` field: every token count here is estimated as
serialized chars / 4 (see load_trajectories.est_tokens). The cache trap still
applies: providers cache the shared input prefix across consecutive calls of a
task, and a model switch resets that cache — the first call after a switch
pays the uncached rate for the whole prefix.

Cached share of call i is estimated as the token size of the item-level prefix
that call i shares with call i-1 of the same trajectory.

Model ids in the export are anonymized (families claude-opus/-sonnet/-fable and
gpt-5.6-* across generations, e.g. claude-opus-5, claude-opus-4-8, gpt-5.6-terra),
so no public price sheet exists. DEFAULT_PRICING is an ASSUMPTION for relative
comparisons; unknown ids fall back to their family rate by prefix match — if a price sheet is posted in the
challenge Discord, put it in scripts/pricing.json; either way, state your
pricing assumption in the writeup.
"""
import json
from functools import lru_cache
from pathlib import Path
from load_trajectories import est_tokens

# Family fallbacks CONSISTENT with scripts/pricing.json (the old file priced
# claude-fable at $0.80 here while pricing.json says $10 — the posted sheet
# wins; these are only fallbacks when pricing.json is absent).
DEFAULT_PRICING = {  # per 1M est. tokens: [uncached_input, cached_input, output] — ASSUMED, not official
    "claude-opus": [5.00, 0.50, 25.00],    # family prefixes: match any generation
    "claude-sonnet": [2.00, 0.20, 10.00],
    "claude-fable": [10.00, 1.00, 50.00],
    "gpt-5.6": [2.00, 0.20, 10.00],
    "_default": [2.00, 0.20, 10.00],
}

def load_pricing():
    p = Path(__file__).parent / "pricing.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else DEFAULT_PRICING

@lru_cache(maxsize=None)
def _prefixes_by_len(pricing_key):
    return sorted(pricing_key, key=len, reverse=True)

def price_of(model, pricing):
    if model in pricing: return pricing[model]
    for prefix in _prefixes_by_len(tuple(pricing)):  # longest family prefix wins (sort hoisted)
        if prefix != "_default" and model.startswith(prefix): return pricing[prefix]
    return pricing["_default"]

def call_token_profiles(calls):
    """[(input_tokens, tokens_shared_with_previous_call)] per call — each item
    serialized ONCE (the old path re-serialized every shared item per pair)."""
    prev_ser, out = None, []
    for c in calls:
        ser = [json.dumps(it) for it in c["input"]]
        toks = [len(s) // 4 for s in ser]
        inp = sum(toks)
        shared = 0
        if prev_ser is not None:
            for a, b, t in zip(prev_ser, ser, toks):
                if a == b: shared += t
                else: break
        out.append((inp, min(shared, inp)))
        prev_ser = ser
    return out

def shared_prefix_tokens(prev_req, req):
    """Estimated tokens of the item-level input prefix shared with the previous call."""
    shared = 0
    for a, b in zip(prev_req["input"], req["input"]):
        if a == b: shared += est_tokens(a)
        else: break
    return shared

def trajectory_cost(calls, route, pricing=None, profiles=None):
    """Cost of a reconstructed trajectory (calls ordered by input length) if call i
    had been served by route[i]. Cache-aware: the shared prefix is billed at the
    cached rate only when route[i] == route[i-1]. Output tokens are unknowable
    (no outputs in the export) and are NOT included — say so when you quote numbers.
    Pass `profiles` (call_token_profiles) to avoid re-serializing across routes.
    Returns (usd, uncached_input_tokens_est)."""
    pricing = pricing or load_pricing()
    profiles = profiles or call_token_profiles(calls)
    usd, uncached_total = 0.0, 0
    for i, (inp, shared) in enumerate(profiles):
        cached = shared if (i > 0 and route[i] == route[i - 1]) else 0
        uncached = inp - cached
        pu, pc, _ = price_of(route[i], pricing)
        usd += (uncached * pu + cached * pc) / 1e6
        uncached_total += uncached
    return usd, uncached_total

def logged_route(calls): return [c["model"] for c in calls]
