#!/usr/bin/env python3
"""Build a cost-quality frontier from nested out-of-fold router scores.

This sweeps decision thresholds only. The scores remain the held-out predictions
from ``optimize_deterministic.py``. Because this report inspects many thresholds
on the same OOF predictions, it is sensitivity analysis, not a fresh unbiased
estimate for whichever point looks best after inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_twinrouterbench import (  # noqa: E402
    TWIN_TIER_TO_MERGED_ID,
    _load_rows,
    compute_merged_metrics,
)


TIERS = ("economical", "balanced", "strongest")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tier(score: int, economical_threshold: int, strongest_threshold: int) -> str:
    if score >= strongest_threshold:
        return "strongest"
    if score >= economical_threshold:
        return "balanced"
    return "economical"


def _pareto_flags(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        dominated = any(
            other is not row
            and other["normalized_failure_aware_bill_percent"]
            <= row["normalized_failure_aware_bill_percent"]
            and other["row_weighted_trajectory_pass_percent"]
            >= row["row_weighted_trajectory_pass_percent"]
            and (
                other["normalized_failure_aware_bill_percent"]
                < row["normalized_failure_aware_bill_percent"]
                or other["row_weighted_trajectory_pass_percent"]
                > row["row_weighted_trajectory_pass_percent"]
            )
            for other in rows
        )
        row["pareto_efficient"] = not dominated


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    plt.figure(figsize=(7.4, 4.8))
    color = "#6748FD"
    plt.scatter(
        [row["normalized_failure_aware_bill_percent"] for row in rows],
        [row["row_weighted_trajectory_pass_percent"] for row in rows],
        s=14,
        alpha=0.25,
        color=color,
    )
    frontier = sorted(
        (row for row in rows if row["pareto_efficient"]),
        key=lambda row: row["normalized_failure_aware_bill_percent"],
    )
    plt.plot(
        [row["normalized_failure_aware_bill_percent"] for row in frontier],
        [row["row_weighted_trajectory_pass_percent"] for row in frontier],
        "o-",
        markersize=4,
        label="Pareto frontier",
        color=color,
    )
    selected = [row for row in rows if row["runtime_v3_equivalent"]]
    if selected:
        point = selected[0]
        plt.scatter(
            [point["normalized_failure_aware_bill_percent"]],
            [point["row_weighted_trajectory_pass_percent"]],
            marker="*",
            s=170,
            edgecolor="black",
            color=color,
            label="runtime v3 score thresholds + margin",
            zorder=5,
        )
    plt.xlabel("Failure-aware bill (% of always-high; lower is better)")
    plt.ylabel("Row-weighted trajectory pass (%)")
    plt.title("Nested OOF router threshold frontier (tier-price proxy)")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def _failure_aware_proxy(records: list[dict[str, Any]], field: str) -> float:
    """Twin output-tier price proxy used by deterministic weight optimization."""

    tier_cost = {"economical": 0.5, "balanced": 3.5, "strongest": 25.0}
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        trajectories[str(record["instance_id"])].append(record)
    baseline_bill = 25.0 * len(records)
    predicted_bill = 0.0
    for steps in trajectories.values():
        passed = all(
            TIERS.index(str(step[field])) >= int(step["gold_merged_tier_id"])
            for step in steps
        )
        predicted_bill += sum(tier_cost[str(step[field])] for step in steps)
        if not passed:
            predicted_bill += 25.0 * len(steps)
    return 100.0 * (baseline_bill - predicted_bill) / baseline_bill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin-repo", default=".external/TwinRouterBench")
    parser.add_argument(
        "--predictions",
        default="results/optimized_deterministic/nested_oof_predictions.jsonl",
    )
    parser.add_argument("--output-dir", default="results/hybrid_router")
    parser.add_argument(
        "--official-reference",
        default="results/optimized_deterministic/evaluation_summary.json",
    )
    parser.add_argument("--economical-min", type=int, default=20)
    parser.add_argument("--economical-max", type=int, default=55)
    parser.add_argument("--strongest-min", type=int, default=55)
    parser.add_argument("--strongest-max", type=int, default=90)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    twin_repo = Path(args.twin_repo)
    source_rows = _load_rows(twin_repo / "data" / "static" / "question_bank.jsonl")
    source_by_id = {str(row["id"]): row for row in source_rows}
    predictions = _read_jsonl(Path(args.predictions))
    if len(predictions) != len(source_rows):
        raise ValueError(
            f"prediction/source row mismatch: {len(predictions)} != {len(source_rows)}"
        )

    base_records: list[dict[str, Any]] = []
    for prediction in predictions:
        row_id = str(prediction["id"])
        source = source_by_id[row_id]
        base_records.append(
            {
                "id": row_id,
                "benchmark": str(source["benchmark"]),
                "instance_id": str(source.get("instance_id", row_id)),
                "gold_merged_tier_id": TWIN_TIER_TO_MERGED_ID[str(source["target_tier"])],
                "optimized_score": int(prediction["optimized_score"]),
            }
        )

    points: list[dict[str, Any]] = []
    for economical in range(args.economical_min, args.economical_max + 1, args.step):
        for strongest in range(args.strongest_min, args.strongest_max + 1, args.step):
            if economical >= strongest:
                continue
            prediction_field = "frontier_prediction"
            records = [
                {
                    **record,
                    prediction_field: _tier(
                        record["optimized_score"], economical, strongest
                    ),
                }
                for record in base_records
            ]
            merged = compute_merged_metrics(records, prediction_field)
            savings = _failure_aware_proxy(records, prediction_field)
            points.append(
                {
                    "economical_threshold": economical,
                    "strongest_threshold": strongest,
                    "exact_tier_percent": merged["exact_rate_percent"],
                    "safe_step_percent": merged["safe_step_rate_percent"],
                    "under_route_percent": merged["under_routing_rate_percent"],
                    "over_route_percent": merged["over_routing_rate_percent"],
                    "row_weighted_trajectory_pass_percent": merged[
                        "row_weighted_trajectory_pass_rate_percent"
                    ],
                    "failure_aware_cost_savings_proxy_percent": round(savings, 6),
                    "normalized_failure_aware_bill_percent": round(100.0 - savings, 6),
                    "runtime_v3_equivalent": economical == 35 and strongest == 70,
                    "pareto_efficient": False,
                }
            )

    _pareto_flags(points)
    points.sort(
        key=lambda row: (
            row["normalized_failure_aware_bill_percent"],
            -row["row_weighted_trajectory_pass_percent"],
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "frontier_all_points.csv", points)
    pareto = [row for row in points if row["pareto_efficient"]]
    _write_csv(output_dir / "frontier_pareto.csv", pareto)
    plotted = _plot(output_dir / "frontier.png", points)

    selected = [row for row in points if row["runtime_v3_equivalent"]]
    official_reference: dict[str, Any] | None = None
    official_path = Path(args.official_reference)
    if official_path.exists():
        evaluation = json.loads(official_path.read_text(encoding="utf-8"))
        official_reference = evaluation["nested_grouped_oof"]["official_four_tier"]
    summary = {
        "scope": {
            "rows": len(base_records),
            "trajectories": len({row["instance_id"] for row in base_records}),
            "score_source": str(args.predictions),
            "score_evaluation": "nested grouped out-of-fold",
            "threshold_points": len(points),
        },
        "runtime_v3_equivalent": selected,
        "official_selected_policy_reference": official_reference,
        "pareto_points": pareto,
        "interpretation": {
            "quality": "a trajectory passes only when every step is at or above its cheapest-sufficient Twin tier",
            "cost": "fast Twin tier-price proxy (0.5/3.5/25); a failed trajectory adds an always-high retry",
            "official_reference": "the selected 40/75 nested OOF policy's previously computed official two-mapping result is included separately; it is not substituted into every proxy frontier point",
            "selection_warning": "thresholds were swept on the same OOF predictions; this is sensitivity analysis, not a fresh unbiased estimate of a selected threshold",
            "guard_warning": "hard floors, OOD escalation, and learned disagreement are not reconstructed in this score-only frontier",
        },
        "artifacts": {
            "all_points_csv": "frontier_all_points.csv",
            "pareto_csv": "frontier_pareto.csv",
            "png": "frontier.png" if plotted else None,
        },
    }
    (output_dir / "frontier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["scope"], indent=2, sort_keys=True))
    print(json.dumps({"runtime_v3_equivalent": selected}, indent=2, sort_keys=True))
    print(f"wrote {output_dir / 'frontier_summary.json'}")


if __name__ == "__main__":
    main()
