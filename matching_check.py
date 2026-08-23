#!/usr/bin/env python3
"""Matched cross-model comparison — the off-policy validation of 'served'.

Our quality proxy assumes: a task is fine if its tier >= its observed
difficulty. This script tests the premise behind that with the natural
experiment the deck points at: tasks of SIMILAR routing-time difficulty
(hard = top-2 quintiles of the router's OOF score, which never saw the model
id) were served by DIFFERENT models. If low-tier models struggle more than
high-tier models on matched hard tasks — more tool errors, longer retry
streaks — then under-routing has a real, measurable cost.

Design guards (both were audit findings):
  - CIRCULARITY: the tier map comes from infer_tiers.py's SPLIT-HALF fit
    (models_half_a, even-indexed trajectories); this script evaluates the
    ODD-indexed half only, so the map is never validated on the rows that
    produced it. The signals are still the same KIND the inference used —
    stated here and in the writeup.
  - SIGNIFICANCE: BOTH difference-in-differences interactions (error rate AND
    retry streak) get bootstrap CIs (5000 resamples, computed from the
    unrounded per-trajectory arrays); the verdict reads both.

Writes results/matching_check.json.

Usage: python matching_check.py
"""
import json
from pathlib import Path

import numpy as np

N_BOOT = 5000


def main():
    tiers_rows = [json.loads(l) for l in open("results/tiers.jsonl", encoding="utf-8")]
    tiers_rows.sort(key=lambda r: r["trajectory_id"])
    mets = {r["trajectory_id"]: r for r in
            (json.loads(l) for l in open("results/evaluator_metrics.jsonl", encoding="utf-8"))}
    data = json.loads(Path("dashboard/data.js").read_text(encoding="utf-8")
                      .removeprefix("window.VIKTOR_DATA = ").rstrip().rstrip(";"))
    model_of = {r["id"]: r["model"] for r in data["rows"]}
    mt = json.load(open("results/model_tiers.json", encoding="utf-8"))
    half_a = mt.get("models_half_a")
    tier_src = half_a if half_a else mt["models"]
    tier_of = {m["model"]: m["tier"] for m in tier_src}
    print("tier map: " + ("split-half fit (models_half_a) — validating on the "
                          "held-out odd half" if half_a else
                          "FULL-data fit — circular with this check; rerun infer_tiers.py"))

    # validation rows: the ODD-index half only (the map was fit on the even half)
    val_rows = tiers_rows[1::2] if half_a else tiers_rows
    rows = []
    for t in val_rows:
        m = mets[t["trajectory_id"]]
        rows.append({
            "tier": tier_of[model_of[t["trajectory_id"]]],
            "score": t["router_score"],
            "err_rate": m["n_tool_errors"] / max(m["n_tool_calls"], 1),
            "streak": float(m["max_repeat_streak"]),
        })
    print(f"validation rows: {len(rows)}")

    scores = np.array([r["score"] for r in rows])
    hard = scores >= np.quantile(scores, 0.6)      # top-2 difficulty quintiles
    low_t = np.array([r["tier"] in (1, 2) for r in rows])
    err = np.array([r["err_rate"] for r in rows])
    stk = np.array([r["streak"] for r in rows])

    def cell(mask_hard, mask_low):
        sel = mask_hard & mask_low
        if sel.sum() < 10:
            return None
        return {"n": int(sel.sum()),
                "err_rate": round(float(err[sel].mean()), 4),
                "retry_streak": round(float(stk[sel].mean()), 2)}

    table = {
        "easy_low_tier": cell(~hard, low_t), "easy_high_tier": cell(~hard, ~low_t),
        "hard_low_tier": cell(hard, low_t), "hard_high_tier": cell(hard, ~low_t),
    }
    for k, v in table.items():
        print(f"{k:16s} {v}")

    def did(values, h, lo):
        """Difference-in-differences from the UNROUNDED arrays."""
        return ((values[h & lo].mean() - values[h & ~lo].mean())
                - (values[~h & lo].mean() - values[~h & ~lo].mean()))

    out = {"table": table,
           "design": "tier map fit on even-half trajectories (infer_tiers.py), "
                     "validated on this odd half only; both interactions "
                     f"bootstrapped with {N_BOOT} resamples on unrounded arrays"}
    if all(table.values()):
        did_err = did(err, hard, low_t)
        did_st = did(stk, hard, low_t)
        rng = np.random.default_rng(7)
        n = len(rows)
        boots_err, boots_st = [], []
        for _ in range(N_BOOT):
            pick = rng.integers(0, n, n)
            h, lo = hard[pick], low_t[pick]
            if min((h & lo).sum(), (h & ~lo).sum(), (~h & lo).sum(), (~h & ~lo).sum()) < 5:
                continue
            boots_err.append(did(err[pick], h, lo))
            boots_st.append(did(stk[pick], h, lo))
        e_lo, e_hi = np.percentile(boots_err, [2.5, 97.5])
        s_lo, s_hi = np.percentile(boots_st, [2.5, 97.5])
        sig_err = bool(e_lo > 0 or e_hi < 0)
        sig_st = bool(s_lo > 0 or s_hi < 0)
        out["interaction"] = {
            "extra_err_rate_of_low_tier_on_hard": round(float(did_err), 4),
            "err_ci95": [round(float(e_lo), 4), round(float(e_hi), 4)],
            "err_significant": sig_err,
            "extra_retry_streak_of_low_tier_on_hard": round(float(did_st), 3),
            "streak_ci95": [round(float(s_lo), 3), round(float(s_hi), 3)],
            "streak_significant": sig_st,
            "n_boot": N_BOOT,
        }
        print(f"\ninteraction (difference-in-differences, on the held-out half):")
        print(f"  error rate: {did_err:+.4f} (95% CI [{e_lo:+.4f}, {e_hi:+.4f}]) "
              f"{'SIGNIFICANT' if sig_err else 'not significant'}")
        print(f"  retry streak: {did_st:+.2f} (95% CI [{s_lo:+.2f}, {s_hi:+.2f}]) "
              f"{'SIGNIFICANT' if sig_st else 'not significant'}")
        if sig_err and sig_st and did_err > 0 and did_st > 0:
            verdict = "SUPPORTED on both metrics: under-routing has a measurable cost"
        elif (sig_err and did_err > 0) or (sig_st and did_st > 0):
            # name the OTHER metric's actual sign — the old wording said
            # "directionally positive" even when it had flipped negative
            if sig_err and did_err > 0:
                which, other, o_did, o_sig = "error-rate", "retry-streak", did_st, sig_st
            else:
                which, other, o_did, o_sig = "retry-streak", "error-rate", did_err, sig_err
            sign = "positive" if o_did > 0 else "negative" if o_did < 0 else "flat"
            tail = ("and significant with the OPPOSITE sign — inspect before claiming"
                    if o_sig else "but its CI includes zero")
            verdict = (f"PARTIALLY SUPPORTED: the {which} interaction is positive and "
                       f"significant; the {other} interaction is directionally {sign} "
                       f"{tail}")
        elif (sig_err and did_err < 0) or (sig_st and did_st < 0):
            verdict = "REFUTED-DIRECTION on at least one metric — inspect before claiming"
        else:
            verdict = (f"INCONCLUSIVE at n={len(rows)} — the served penalty stays an "
                       f"assumption; state it as such")
        out["verdict"] = verdict
        print("verdict:", verdict)
    with open("results/matching_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results/matching_check.json")


if __name__ == "__main__":
    main()
