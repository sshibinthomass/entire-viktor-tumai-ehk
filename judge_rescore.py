#!/usr/bin/env python3
"""Judge-model rescoring of matched call pairs (deck starter idea 3).

The strongest possible off-policy signal: take pairs of calls with SIMILAR
routing-time difficulty served by DIFFERENT tiers, recover the model's actual
reply from the NEXT call's input (call i's output appears inside call i+1's
input), and ask a judge model which reply better advances the task. This
breaks the tier-inference circularity with a quality signal that is
independent of the effort metrics.

Requires the team's own ANTHROPIC_API_KEY (allowed by the rules; everything
else in this repo runs offline). Without a key it stops after building and
saving the matched pairs so the sampling is reproducible.

Usage: python judge_rescore.py [export_linked/trajectories_v1_01.jsonl] [--pairs 50]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

JUDGE_MODEL = "claude-sonnet-5"
PROMPT = """You are judging two AI assistant replies to the SAME kind of workplace task.
Task A opening (redacted): {task_a}
Reply A (by model X): {reply_a}
Task B opening (redacted): {task_b}
Reply B (by model Y): {reply_b}
The two tasks were matched to be of similar difficulty. Which reply more
competently advances its task — fewer flailing tool calls, more coherent plan?
Answer with exactly one word: A, B, or TIE."""


def recovered_reply(calls, i):
    """Model output of call i = the items call i+1's input adds beyond call i's."""
    if i + 1 >= len(calls):
        return None
    prev, nxt = calls[i]["input"], calls[i + 1]["input"]
    added = nxt[len(prev):]
    model_items = [it for it in added
                   if it.get("role") == "assistant"
                   or it.get("type") in ("function_call", "custom_tool_call")]
    return json.dumps(model_items)[:4000] if model_items else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?", default="export_linked")
    ap.add_argument("--pairs", type=int, default=50)
    a = ap.parse_args()

    tiers = {json.loads(l)["trajectory_id"]: json.loads(l)
             for l in open("results/tiers.jsonl", encoding="utf-8")}
    mt = json.load(open("results/model_tiers.json", encoding="utf-8"))
    tier_of_model = {m["model"]: m["tier"] for m in mt["models"]}

    p = Path(a.export)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    calls_of = defaultdict(list)
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    req = json.loads(line)
                    calls_of[req["trajectory_id"]].append(req)
    for tid in calls_of:
        calls_of[tid].sort(key=lambda r: r.get("call_index", 0))

    # candidates: multi-call trajectories with a recoverable reply
    cands = []
    for tid, calls in calls_of.items():
        if len(calls) < 2 or tid not in tiers:
            continue
        rep = recovered_reply(calls, 0)
        if not rep:
            continue
        first_user = ""
        for it in calls[0]["input"]:
            if it.get("role") == "user":
                c = it.get("content")
                first_user = c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in (c or []) if isinstance(p, dict))
                break
        cands.append({"tid": tid, "score": tiers[tid]["router_score"],
                      "model": calls[0]["model"],
                      "model_tier": tier_of_model.get(calls[0]["model"], 2),
                      "task": first_user[:600], "reply": rep})
    print(f"{len(cands)} trajectories with a recoverable first reply")

    # match low-tier vs high-tier candidates by routing score
    low = sorted([c for c in cands if c["model_tier"] <= 2], key=lambda c: c["score"])
    high = sorted([c for c in cands if c["model_tier"] == 3], key=lambda c: c["score"])
    pairs = []
    used = set()
    for c in low:
        best, bd = None, 1e9
        for h in high:
            if h["tid"] in used:
                continue
            d = abs(h["score"] - c["score"])
            if d < bd:
                best, bd = h, d
        if best and bd <= 0.05:
            used.add(best["tid"])
            pairs.append({"low": c, "high": best, "score_gap": round(bd, 4)})
        if len(pairs) >= a.pairs:
            break
    print(f"built {len(pairs)} matched pairs (|routing-score gap| <= 0.05)")
    Path("results").mkdir(exist_ok=True)
    # pair METADATA only is saved (ids, scores, tiers) — the verbatim task/reply
    # text stays out of results/ because the dataset is no-redistribution
    with open("results/judge_pairs.json", "w", encoding="utf-8") as f:
        json.dump([{"low_tid": p["low"]["tid"], "high_tid": p["high"]["tid"],
                    "low_model": p["low"]["model"], "high_model": p["high"]["model"],
                    "score_gap": p["score_gap"]} for p in pairs], f, indent=2)
    print("wrote results/judge_pairs.json (pair metadata)")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY not set — stopping before the judge calls. "
              "Set the key and rerun to score the pairs.")
        return
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic to run the judge")
    client = anthropic.Anthropic()
    votes = {"A": 0, "B": 0, "TIE": 0}
    for p in pairs:
        msg = client.messages.create(
            model=JUDGE_MODEL, max_tokens=5,
            messages=[{"role": "user", "content": PROMPT.format(
                task_a=p["low"]["task"], reply_a=p["low"]["reply"],
                task_b=p["high"]["task"], reply_b=p["high"]["reply"])}])
        v = msg.content[0].text.strip().upper()
        votes[v if v in votes else "TIE"] += 1
    n = sum(votes.values())
    out = {"judge_model": JUDGE_MODEL, "n_pairs": n, "votes": votes,
           "high_tier_win_rate": votes["B"] / max(n, 1),
           "note": "A = low-tier reply, B = high-tier reply, matched on routing score"}
    with open("results/judge_rescore.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"judge votes: {votes} -> high-tier win rate "
          f"{out['high_tier_win_rate']:.0%}; wrote results/judge_rescore.json")


if __name__ == "__main__":
    main()
