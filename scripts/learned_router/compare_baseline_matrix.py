#!/usr/bin/env python3
"""Compare baseline_router choices with the learned logistic router as a 3x3 matrix.

The comparison runs on reconstructed Viktor trajectories because baseline_router
depends on each trajectory's logged model.  Rows are baseline-router tiers and
columns are learned-router tiers.  Neither axis is a quality label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from baseline_router import CHEAP, SMALL_TRAJECTORY, route_trajectory  # noqa: E402
from extract_router_metrics import metrics_from_request  # noqa: E402
from learned_router.features import reduced_features  # noqa: E402
from learned_router.models import OrdinalLogisticRouter  # noqa: E402
from load_trajectories import (  # noqa: E402
    group_trajectories,
    is_generated_synthetic,
    iter_requests,
)
from model_choice_matrix import LOGGED_TIER  # noqa: E402


TIERS = ("economical", "balanced", "strongest")
TIER_RANK = {tier: index for index, tier in enumerate(TIERS)}


def build_matrix(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    counts: Counter[tuple[str, str]] = Counter(pairs)
    total = sum(counts.values())
    if not total:
        raise ValueError("cannot build a comparison matrix without decisions")
    count_matrix = {
        baseline: {learned: counts[(baseline, learned)] for learned in TIERS}
        for baseline in TIERS
    }
    row_percent: dict[str, dict[str, float]] = {}
    for baseline in TIERS:
        row_total = sum(count_matrix[baseline].values())
        row_percent[baseline] = {
            learned: round(100.0 * count_matrix[baseline][learned] / row_total, 1)
            if row_total
            else 0.0
            for learned in TIERS
        }
    overall_percent = {
        baseline: {
            learned: round(100.0 * counts[(baseline, learned)] / total, 1)
            for learned in TIERS
        }
        for baseline in TIERS
    }
    agreement = sum(counts[(tier, tier)] for tier in TIERS)
    learned_lower = sum(
        count
        for (baseline, learned), count in counts.items()
        if TIER_RANK[learned] < TIER_RANK[baseline]
    )
    learned_higher = total - agreement - learned_lower
    return {
        "total_requests": total,
        "counts": count_matrix,
        "row_percent_within_baseline_tier": row_percent,
        "overall_percent_of_all_requests": overall_percent,
        "summary": {
            "exact_choice_agreement_count": agreement,
            "exact_choice_agreement_percent": round(100.0 * agreement / total, 1),
            "learned_lower_than_baseline_count": learned_lower,
            "learned_lower_than_baseline_percent": round(100.0 * learned_lower / total, 1),
            "learned_higher_than_baseline_count": learned_higher,
            "learned_higher_than_baseline_percent": round(100.0 * learned_higher / total, 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="export")
    parser.add_argument(
        "--model",
        default="results/learned_router/ordinal_logistic_model.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/learned_router",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    model = OrdinalLogisticRouter.from_dict(artifact["model"])
    requests = [
        request
        for _, _, request in iter_requests(args.export)
        if not is_generated_synthetic(request)
    ]
    groups = group_trajectories(requests)
    pairs: list[tuple[str, str]] = []
    baseline_model_counts: Counter[str] = Counter()
    learned_tier_counts: Counter[str] = Counter()
    for calls in groups.values():
        baseline_models = route_trajectory(calls)
        for request, baseline_model in zip(calls, baseline_models):
            if baseline_model not in LOGGED_TIER:
                raise ValueError(f"no assumed tier for baseline model {baseline_model!r}")
            baseline_tier = LOGGED_TIER[baseline_model]
            learned_tier = model.predict_one(
                reduced_features(metrics_from_request(request))
            ).tier
            pairs.append((baseline_tier, learned_tier))
            baseline_model_counts[baseline_model] += 1
            learned_tier_counts[learned_tier] += 1

    result = {
        "source": str(args.export),
        "orientation": "rows=baseline_router tier, columns=learned ordinal-logistic tier",
        "comparison_semantics": "choice agreement only; neither router is ground truth",
        "baseline_policy": {
            "small_trajectory_estimated_input_token_threshold": SMALL_TRAJECTORY,
            "cheap_model_by_family": CHEAP,
            "model_to_tier_assumption": LOGGED_TIER,
            "baseline_model_choice_counts": dict(sorted(baseline_model_counts.items())),
        },
        "learned_model": {
            "artifact": str(model_path),
            "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "model_type": artifact["model"]["model_type"],
            "trainable_parameter_count": artifact["model"]["trainable_parameter_count"],
            "tier_choice_counts": {tier: learned_tier_counts[tier] for tier in TIERS},
        },
        **build_matrix(pairs),
        "warning": (
            "Viktor logs contain no counterfactual quality label. Agreement, a higher learned "
            "tier, or a lower learned tier does not by itself establish better routing."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_vs_logistic_matrix.json"
    row_csv_path = output_dir / "baseline_vs_logistic_matrix_row_percent.csv"
    overall_csv_path = output_dir / "baseline_vs_logistic_matrix_overall_percent.csv"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with row_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline_router / learned_logistic", *TIERS])
        for baseline in TIERS:
            writer.writerow(
                [
                    baseline,
                    *(
                        result["row_percent_within_baseline_tier"][baseline][learned]
                        for learned in TIERS
                    ),
                ]
            )
    with overall_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline_router / learned_logistic", *TIERS])
        for baseline in TIERS:
            writer.writerow(
                [
                    baseline,
                    *(
                        result["overall_percent_of_all_requests"][baseline][learned]
                        for learned in TIERS
                    ),
                ]
            )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {row_csv_path}")
    print(f"wrote {overall_csv_path}")


if __name__ == "__main__":
    main()
