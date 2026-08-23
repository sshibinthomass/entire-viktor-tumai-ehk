#!/usr/bin/env python3
"""Matched cross-model comparison — the off-policy validation of 'served'.

Our quality proxy assumes: a task is fine if its tier >= its observed
difficulty. This script tests the premise behind that with the natural
experiment the deck points at: tasks of SIMILAR routing-time difficulty
(matched into quintile buckets by the router's OOF score, which never saw the
model id) were served by DIFFERENT models. If low-tier models struggle more
than high-tier models on matched hard tasks — more tool errors, longer retry
streaks — then under-routing has a real, measurable cost and the served
penalty is evidence, not assumption.

Tier groups come from infer_tiers.py (results/model_tiers.json).

Writes results/matching_check.json.

Usage: python matching_check.py
"""
import json
from pathlib import Path

import numpy as np


def main():
    tiers_rows = [json.loads(l) for l in open("results/tiers.jsonl", encoding="utf-8")]
    mets = {r["trajectory_id"]: r for r in
            (json.loads(l) for l in open("results/evaluator_metrics.jsonl", encoding="utf-8"))}
    data = json.loads(Path("dashboard/data.js").read_text(encoding="utf-8")
                      .removeprefix("window.VIKTOR_DATA = ").rstrip().rstrip(";"))
    model_of = {r["id"]: r["model"] for r in data["rows"]}
    tier_of = {m["model"]: m["tier"] for m in
               json.load(open("results/model_tiers.json", encoding="utf-8"))["models"]}

    rows = []
    for t in tiers_rows:
        m = mets[t["trajectory_id"]]
        rows.append({
            "tier": tier_of[model_of[t["trajectory_id"]]],
            "score": t["router_score"],
            "err_rate": m["n_tool_errors"] / max(m["n_tool_calls"], 1),
            "streak": m["max_repeat_streak"],
        })

    scores = np.array([r["score"] for r in rows])
    hard = scores >= np.quantile(scores, 0.6)      # top-2 difficulty quintiles
    easy = ~hard

    def cell(mask, tg):
        sel = [r for i, r in enumerate(rows) if mask[i] and r["tier"] in tg]
        if len(sel) < 10:
            return None
        return {"n": len(sel),
                "err_rate": round(float(np.mean([r["err_rate"] for r in sel])), 4),
                "retry_streak": round(float(np.mean([r["streak"] for r in sel])), 2)}

    low, high = (1, 2), (3,)
    table = {
        "easy_low_tier": cell(easy, low), "easy_high_tier": cell(easy, high),
        "hard_low_tier": cell(hard, low), "hard_high_tier": cell(hard, high),
    }
    for k, v in table.items():
        print(f"{k:16s} {v}")

    # the interaction is the finding: extra struggle of low-tier models ON HARD
    # TASKS beyond their baseline gap on easy tasks (difference-in-differences)
    out = {"table": table}
    if all(table.values()):
        did_err = ((table["hard_low_tier"]["err_rate"] - table["hard_high_tier"]["err_rate"])
                   - (table["easy_low_tier"]["err_rate"] - table["easy_high_tier"]["err_rate"]))
        did_st = ((table["hard_low_tier"]["retry_streak"] - table["hard_high_tier"]["retry_streak"])
                  - (table["easy_low_tier"]["retry_streak"] - table["easy_high_tier"]["retry_streak"]))
        # bootstrap CI on the error-rate interaction
        rng = np.random.default_rng(7)
        boots = []
        idx_all = np.arange(len(rows))
        for _ in range(500):
            pick = rng.choice(idx_all, len(idx_all))
            def bcell(mask, tg):
                v = [rows[i]["err_rate"] for i in pick if mask[i] and rows[i]["tier"] in tg]
                return np.mean(v) if len(v) >= 5 else np.nan
            b = ((bcell(hard, low) - bcell(hard, high))
                 - (bcell(easy, low) - bcell(easy, high)))
            if b == b:
                boots.append(b)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["interaction"] = {
            "extra_err_rate_of_low_tier_on_hard": round(float(did_err), 4),
            "err_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "extra_retry_streak_of_low_tier_on_hard": round(float(did_st), 3),
            "significant": bool(lo > 0 or hi < 0),
        }
        print(f"\ninteraction (difference-in-differences): low-tier models show "
              f"{did_err:+.4f} extra error rate on hard tasks (95% CI "
              f"[{lo:+.4f}, {hi:+.4f}]) and {did_st:+.2f} extra retry streak.")
        verdict = ("SUPPORTED: under-routing has a measurable cost"
                   if lo > 0 else
                   ("REFUTED-DIRECTION" if hi < 0 else
                    "INCONCLUSIVE at n=953 — the served penalty stays an assumption; "
                    "state it as such"))
        out["verdict"] = verdict
        print("verdict:", verdict)
    with open("results/matching_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results/matching_check.json")


if __name__ == "__main__":
    main()
