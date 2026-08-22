#!/usr/bin/env python3
"""Compare router tiers with logged model tiers as a 3x3 matrix.

Usage:
    python scripts/model_choice_matrix.py export/trajectories_v1_01.jsonl

The logged tier is treated as the reference label only for this diagnostic. The
mapping from anonymized model IDs to tiers remains an explicit assumption.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from deterministic_router import evaluate_route
from extract_router_metrics import metrics_from_request


TIERS = ("economical", "balanced", "strongest")
LOGGED_TIER = {
    "claude-fable-5": "economical",
    "claude-sonnet-4-6": "balanced",
    "claude-sonnet-5": "balanced",
    "claude-opus-4-6": "strongest",
    "claude-opus-4-8": "strongest",
    "claude-opus-5": "strongest",
    "gpt-5.6-sol": "economical",
    "gpt-5.6-terra": "balanced",
    "gpt-5.6-luna": "strongest",
}
TIER_RANK = {tier: index for index, tier in enumerate(TIERS)}


def main() -> None:
    source = Path(
        sys.argv[1] if len(sys.argv) > 1 else "export/trajectories_v1_01.jsonl"
    )
    counts: Counter[tuple[str, str]] = Counter()
    model_counts: Counter[str] = Counter()
    total = 0

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            request = json.loads(line)
            model = request["model"]
            if model not in LOGGED_TIER:
                raise ValueError(f"line {line_number}: no assumed tier for {model!r}")
            decision = evaluate_route({"metrics": asdict(metrics_from_request(request))})
            counts[(LOGGED_TIER[model], decision["target_tier"])] += 1
            model_counts[model] += 1
            total += 1

    count_matrix = {
        reference: {predicted: counts[(reference, predicted)] for predicted in TIERS}
        for reference in TIERS
    }
    overall_percent = {
        reference: {
            predicted: round(100 * counts[(reference, predicted)] / total, 1)
            for predicted in TIERS
        }
        for reference in TIERS
    }
    row_percent = {}
    for reference in TIERS:
        row_total = sum(counts[(reference, predicted)] for predicted in TIERS)
        first = round(100 * counts[(reference, TIERS[0])] / row_total, 1)
        second = round(100 * counts[(reference, TIERS[1])] / row_total, 1)
        row_percent[reference] = {
            TIERS[0]: first,
            TIERS[1]: second,
            TIERS[2]: round(100.0 - first - second, 1),
        }

    agreement = sum(counts[(tier, tier)] for tier in TIERS)
    under_routed = sum(
        count
        for (reference, predicted), count in counts.items()
        if TIER_RANK[predicted] < TIER_RANK[reference]
    )
    over_routed = total - agreement - under_routed

    result = {
        "source": str(source),
        "orientation": "rows=logged reference tier, columns=our router tier",
        "total_requests": total,
        "logged_tier_mapping_assumption": LOGGED_TIER,
        "logged_model_counts": dict(sorted(model_counts.items())),
        "counts": count_matrix,
        "overall_percent_of_all_requests": overall_percent,
        "row_percent_within_logged_tier": row_percent,
        "summary": {
            "exact_agreement_count": agreement,
            "exact_agreement_percent": round(100 * agreement / total, 1),
            "our_tier_lower_count": under_routed,
            "our_tier_lower_percent": round(100 * under_routed / total, 1),
            "our_tier_higher_count": over_routed,
            "our_tier_higher_percent": round(100 * over_routed / total, 1),
        },
        "warning": "Logged choice is assumed ground truth for this requested diagnostic; it is not an observed quality label.",
    }

    Path("results").mkdir(exist_ok=True)
    json_path = Path("results/model_choice_matrix.json")
    csv_path = Path("results/model_choice_matrix_percent.csv")
    overall_csv_path = Path("results/model_choice_matrix_overall_percent.csv")
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["logged_reference / our_router", *TIERS])
        for reference in TIERS:
            writer.writerow(
                [reference, *(row_percent[reference][predicted] for predicted in TIERS)]
            )
    with overall_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["logged_reference / our_router", *TIERS])
        for reference in TIERS:
            writer.writerow(
                [reference, *(overall_percent[reference][predicted] for predicted in TIERS)]
            )

    assert sum(sum(row.values()) for row in count_matrix.values()) == total
    assert agreement + under_routed + over_routed == total
    assert all(abs(sum(row.values()) - 100.0) < 1e-9 for row in row_percent.values())
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path} (each row sums to 100%)")
    print(f"wrote {overall_csv_path} (percent of all requests)")


if __name__ == "__main__":
    main()
