#!/usr/bin/env python3
"""Compare deterministic complexity routing with the starter baseline.

Both policies make one sticky model choice per structurally clean trajectory and
use the same cache-aware estimated-input cost model. This is a routing/cost test,
not a quality evaluation: the export has no counterfactual outputs.

Usage:
    python scripts/compare_routers.py export/

Writes:
    results/router_comparison.json
    results/router_comparison_routes.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from baseline_router import route_trajectory as baseline_route
from cost_model import load_pricing, logged_route, trajectory_cost
from deterministic_router import evaluate_route
from extract_router_metrics import metrics_from_request
from load_trajectories import group_trajectories, is_generated_synthetic, iter_requests


# Test-only ladder. It is an explicit assumption, not a discovered tier order.
MODEL_LADDERS = {
    "claude": {
        "economical": "claude-fable-5",
        "balanced": "claude-sonnet-5",
        "strongest": "claude-opus-5",
    },
    "gpt": {
        "economical": "gpt-5.6-sol",
        "balanced": "gpt-5.6-terra",
        "strongest": "gpt-5.6-luna",
    },
}

def family_of(model: str) -> str:
    if model.startswith("claude"):
        return "claude"
    if model.startswith("gpt"):
        return "gpt"
    raise ValueError(f"no test ladder configured for model {model!r}")


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def pct_change(value: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return round(value / reference - 1, 6)


def main() -> None:
    export_dir = sys.argv[1] if len(sys.argv) > 1 else "export"
    pricing = load_pricing()
    requests = [r for _, _, r in iter_requests(export_dir)]
    synthetic_requests = [r for r in requests if is_generated_synthetic(r)]
    real_requests = [r for r in requests if not is_generated_synthetic(r)]
    groups = group_trajectories(real_requests)
    clean = {
        key: calls
        for key, calls in groups.items()
        if len({call["model"] for call in calls}) == 1
    }
    excluded_mixed = len(groups) - len(clean)

    totals = {"logged": 0.0, "baseline": 0.0, "deterministic": 0.0}
    scores: list[int] = []
    tier_counts: Counter[str] = Counter()
    logged_model_counts: Counter[str] = Counter()
    deterministic_model_counts: Counter[str] = Counter()
    component_totals: Counter[str] = Counter()
    agreement = Counter()
    family_totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "trajectories": 0,
            "logged_cost": 0.0,
            "baseline_cost": 0.0,
            "deterministic_cost": 0.0,
        }
    )

    Path("results").mkdir(exist_ok=True)
    route_path = Path("results/router_comparison_routes.jsonl")
    with route_path.open("w", encoding="utf-8") as output:
        for trajectory_id, calls in clean.items():
            logged = logged_route(calls)
            logged_model = logged[0]
            family = family_of(logged_model)
            ladder = MODEL_LADDERS[family]

            metrics = metrics_from_request(calls[0])
            decision = evaluate_route(
                {
                    "metrics": asdict(metrics),
                    "mode": "initial",
                    "tier_model_map": ladder,
                }
            )
            selected = decision["selected_model_id"]
            if selected is None:
                raise RuntimeError(f"no selected model for trajectory {trajectory_id}")
            deterministic = [selected] * len(calls)
            baseline = baseline_route(calls)

            logged_cost, _ = trajectory_cost(calls, logged, pricing)
            baseline_cost, _ = trajectory_cost(calls, baseline, pricing)
            deterministic_cost, _ = trajectory_cost(calls, deterministic, pricing)
            totals["logged"] += logged_cost
            totals["baseline"] += baseline_cost
            totals["deterministic"] += deterministic_cost

            score = int(decision["complexity_score"])
            scores.append(score)
            tier_counts[decision["target_tier"]] += 1
            logged_model_counts[logged_model] += 1
            deterministic_model_counts[selected] += 1
            component_totals.update(decision["component_scores"])

            baseline_changed = baseline != logged
            deterministic_changed = deterministic != logged
            agreement["same_full_route"] += int(baseline == deterministic)
            agreement["baseline_changed"] += int(baseline_changed)
            agreement["deterministic_changed"] += int(deterministic_changed)
            agreement["both_changed"] += int(baseline_changed and deterministic_changed)
            agreement["only_baseline_changed"] += int(baseline_changed and not deterministic_changed)
            agreement["only_deterministic_changed"] += int(deterministic_changed and not baseline_changed)

            family_row = family_totals[family]
            family_row["trajectories"] += 1
            family_row["logged_cost"] += logged_cost
            family_row["baseline_cost"] += baseline_cost
            family_row["deterministic_cost"] += deterministic_cost

            record = {
                "trajectory": trajectory_id,
                "family": family,
                "n_calls": len(calls),
                "logged_model": logged_model,
                "baseline_model": baseline[0],
                "deterministic_model": selected,
                "complexity_score": score,
                "component_scores": decision["component_scores"],
                "target_tier": decision["target_tier"],
                "matched_rules": decision["matched_rules"],
                "cost_logged_usd": round(logged_cost, 9),
                "cost_baseline_usd": round(baseline_cost, 9),
                "cost_deterministic_usd": round(deterministic_cost, 9),
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")

    n = len(clean)
    for family_row in family_totals.values():
        for key in ("logged_cost", "baseline_cost", "deterministic_cost"):
            family_row[key] = round(float(family_row[key]), 6)
        family_row["baseline_vs_logged"] = pct_change(
            float(family_row["baseline_cost"]), float(family_row["logged_cost"])
        )
        family_row["deterministic_vs_logged"] = pct_change(
            float(family_row["deterministic_cost"]), float(family_row["logged_cost"])
        )

    summary: dict[str, Any] = {
        "scope": {
            "generated_synthetic_requests_excluded": len(synthetic_requests),
            "requests": sum(len(calls) for calls in clean.values()),
            "clean_trajectories": n,
            "single_call_trajectories": sum(len(calls) == 1 for calls in clean.values()),
            "multi_call_trajectories": sum(len(calls) > 1 for calls in clean.values()),
            "excluded_mixed_model_groups": excluded_mixed,
        },
        "assumptions": {
            "model_ladders": MODEL_LADDERS,
            "gpt_tier_warning": "sol/terra/luna order is a test-only assumption; default prices are identical.",
            "claude_tier_warning": "fable/sonnet/opus order is also treated as an assumption.",
            "token_basis": "serialized characters / 4; estimated, not measured",
            "cost_basis": "assumed input prices, item-prefix cache-aware, output cost excluded",
            "routing_unit": "one sticky model per clean trajectory",
            "quality": "not estimated in this comparison; route agreement is not quality",
            "baseline_information_warning": "starter baseline uses total future trajectory input size, which is unavailable at initial routing time",
            "task_text": "live text after final </system>, otherwise final Thread info block, otherwise full user text",
        },
        "cost_usd": {
            "logged": round(totals["logged"], 6),
            "baseline": round(totals["baseline"], 6),
            "deterministic": round(totals["deterministic"], 6),
            "baseline_vs_logged": pct_change(totals["baseline"], totals["logged"]),
            "deterministic_vs_logged": pct_change(totals["deterministic"], totals["logged"]),
            "deterministic_vs_baseline": pct_change(
                totals["deterministic"], totals["baseline"]
            ),
        },
        "routing": {
            "target_tier_counts": dict(sorted(tier_counts.items())),
            "logged_model_counts": dict(sorted(logged_model_counts.items())),
            "deterministic_model_counts": dict(sorted(deterministic_model_counts.items())),
            **dict(agreement),
            "same_full_route_fraction": round(agreement["same_full_route"] / n, 6) if n else None,
        },
        "scores": {
            "min": min(scores) if scores else None,
            "p25": percentile(scores, 0.25),
            "median": statistics.median(scores) if scores else None,
            "p75": percentile(scores, 0.75),
            "max": max(scores) if scores else None,
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "mean_components": {
                key: round(value / n, 3) for key, value in sorted(component_totals.items())
            }
            if n
            else {},
        },
        "by_family": dict(family_totals),
        "route_records": str(route_path),
    }

    summary_path = Path("results/router_comparison.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")
    print(f"wrote {route_path}")


if __name__ == "__main__":
    main()
