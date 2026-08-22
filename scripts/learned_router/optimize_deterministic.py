#!/usr/bin/env python3
"""Optimize interpretable deterministic-router weights with nested validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backtest_twinrouterbench import (  # noqa: E402
    ROUTER_TIERS,
    _load_rows,
    _merged_id_for_gold,
    _official_policy_summaries,
    _safe_json,
    cluster_bootstrap_cis,
    compute_merged_metrics,
    route_metrics,
    twin_row_to_request,
)
from extract_router_metrics import metrics_from_request  # noqa: E402
from learned_router.deterministic_optimizer import (  # noqa: E402
    DEFAULT_CONFIG,
    SEARCH_SPACE,
    PolicyConfig,
    evaluate_config,
    optimize_policy,
    predict_tier_id,
    score_metrics,
)
from learned_router.train_evaluate import _assign_group_folds  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_safe_json(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _inner_fold_ids(rows: Sequence[Mapping[str, Any]], folds: int, seed: int) -> list[int]:
    assignment = _assign_group_folds(list(rows), folds, seed)
    return [assignment[str(row["instance_id"])] for row in rows]


def _nested_predictions(
    examples: list[dict[str, Any]],
    *,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment = _assign_group_folds(examples, outer_folds, seed)
    predictions: list[dict[str, Any] | None] = [None] * len(examples)
    fold_details: list[dict[str, Any]] = []
    started = time.perf_counter()
    for outer_fold in range(outer_folds):
        train_indices = [
            index
            for index, example in enumerate(examples)
            if assignment[example["instance_id"]] != outer_fold
        ]
        test_indices = [
            index
            for index, example in enumerate(examples)
            if assignment[example["instance_id"]] == outer_fold
        ]
        training_rows = [examples[index] for index in train_indices]
        config, search = optimize_policy(
            training_rows,
            _inner_fold_ids(training_rows, inner_folds, seed + 100 + outer_fold),
        )
        for index in test_indices:
            score = score_metrics(examples[index]["metrics"], config)
            tier_id = predict_tier_id(examples[index]["metrics"], config)
            predictions[index] = {
                "score": score,
                "tier_id": tier_id,
                "tier": ROUTER_TIERS[tier_id],
            }
        fold_details.append(
            {
                "outer_fold": outer_fold,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "train_trajectories": len(
                    {examples[index]["instance_id"] for index in train_indices}
                ),
                "test_trajectories": len(
                    {examples[index]["instance_id"] for index in test_indices}
                ),
                "selected_config": config.to_dict(),
                "search": search,
            }
        )
    if any(prediction is None for prediction in predictions):
        raise AssertionError("nested evaluation left rows without predictions")
    return [prediction for prediction in predictions if prediction is not None], {
        "folds": fold_details,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _leave_one_benchmark_out(
    examples: list[dict[str, Any]], *, inner_folds: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any] | None] = [None] * len(examples)
    details: list[dict[str, Any]] = []
    started = time.perf_counter()
    for split_index, benchmark in enumerate(
        sorted({example["benchmark"] for example in examples})
    ):
        train_indices = [
            index for index, example in enumerate(examples) if example["benchmark"] != benchmark
        ]
        test_indices = [
            index for index, example in enumerate(examples) if example["benchmark"] == benchmark
        ]
        training_rows = [examples[index] for index in train_indices]
        config, search = optimize_policy(
            training_rows,
            _inner_fold_ids(training_rows, inner_folds, seed + 500 + split_index),
        )
        for index in test_indices:
            score = score_metrics(examples[index]["metrics"], config)
            tier_id = predict_tier_id(examples[index]["metrics"], config)
            predictions[index] = {
                "score": score,
                "tier_id": tier_id,
                "tier": ROUTER_TIERS[tier_id],
            }
        details.append(
            {
                "held_out_benchmark": benchmark,
                "selected_config": config.to_dict(),
                "search": search,
            }
        )
    if any(prediction is None for prediction in predictions):
        raise AssertionError("leave-one-benchmark-out evaluation left rows without predictions")
    return [prediction for prediction in predictions if prediction is not None], {
        "splits": details,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _prediction_records(
    examples: list[dict[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    return [
        {
            "id": example["id"],
            "benchmark": example["benchmark"],
            "instance_id": example["instance_id"],
            "step_index": example["step_index"],
            "total_steps": example["total_steps"],
            "gold_merged_tier_id": example["gold_tier_id"],
            field: prediction["tier"],
            f"{field}_score": prediction["score"],
        }
        for example, prediction in zip(examples, predictions)
    ]


def _metrics_with_ci(
    records: list[dict[str, Any]], field: str, bootstrap_samples: int
) -> dict[str, Any]:
    metrics = compute_merged_metrics(records, field)
    metrics["cluster_bootstrap_95_ci"] = cluster_bootstrap_cis(
        records, field, samples=bootstrap_samples
    )
    return metrics


def _official_mean(official: Mapping[str, Any], field: str) -> float:
    return statistics.mean(
        float(result["scores_v2"][field]) for result in official.values()
    )


def _report(path: Path, summary: Mapping[str, Any]) -> None:
    optimized = summary["nested_grouped_oof"]["merged_three_tier"]
    current = summary["frozen_current_router"]["merged_three_tier"]
    lobo = summary["leave_one_benchmark_out"]["merged_three_tier"]
    current_official = summary["frozen_current_router"][
        "mean_official_combined_score_percent"
    ]
    optimized_official = summary["nested_grouped_oof"][
        "mean_official_combined_score_percent"
    ]
    current_official_text = (
        f"{current_official:.1f}%" if current_official is not None else "not run"
    )
    optimized_official_text = (
        f"{optimized_official:.1f}%" if optimized_official is not None else "not run"
    )
    lines = [
        "# Optimized deterministic-router evaluation",
        "",
        "Every optimized headline prediction is from an outer fold whose labels were not used to select its weights. Weights were searched only inside the corresponding training partition.",
        "",
        "| Policy | Exact tier | Safe steps | Under-routed | Over-routed | Row-weighted trajectory pass | Mean official combined |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Frozen current | {current['exact_rate_percent']:.1f}% | "
            f"{current['safe_step_rate_percent']:.1f}% | {current['under_routing_rate_percent']:.1f}% | "
            f"{current['over_routing_rate_percent']:.1f}% | "
            f"{current['row_weighted_trajectory_pass_rate_percent']:.1f}% | "
            f"{current_official_text} |"
        ),
        (
            f"| Nested optimized | {optimized['exact_rate_percent']:.1f}% | "
            f"{optimized['safe_step_rate_percent']:.1f}% | {optimized['under_routing_rate_percent']:.1f}% | "
            f"{optimized['over_routing_rate_percent']:.1f}% | "
            f"{optimized['row_weighted_trajectory_pass_rate_percent']:.1f}% | "
            f"{optimized_official_text} |"
        ),
        "",
        "## Leave-one-benchmark-out",
        "",
        f"Overall safety: {lobo['safe_step_rate_percent']:.1f}%; row-weighted trajectory pass: {lobo['row_weighted_trajectory_pass_rate_percent']:.1f}%.",
        "",
        "| Held-out benchmark | Safe steps | Exact tier | Row-weighted trajectory pass |",
        "|---|---:|---:|---:|",
    ]
    for benchmark, metrics in lobo["by_benchmark"].items():
        lines.append(
            f"| {benchmark} | {metrics['safe_step_rate_percent']:.1f}% | "
            f"{metrics['exact_rate_percent']:.1f}% | "
            f"{metrics['row_weighted_trajectory_pass_rate_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Final configuration",
            "",
            "```json",
            json.dumps(summary["final_fit"]["config"], indent=2, sort_keys=True),
            "```",
            "",
            "The final configuration is fitted on all public rows for deployment experiments. Its in-sample performance is not used as an evaluation result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin-repo", default=".external/TwinRouterBench")
    parser.add_argument("--dependency-path", default=".external/python-packages")
    parser.add_argument("--output-dir", default="results/optimized_deterministic")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-official", action="store_true")
    args = parser.parse_args()

    dependency_path = Path(args.dependency_path)
    if dependency_path.exists() and str(dependency_path.resolve()) not in sys.path:
        sys.path.insert(0, str(dependency_path.resolve()))
    twin_repo = Path(args.twin_repo)
    question_bank = twin_repo / "data" / "static" / "question_bank.jsonl"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _load_rows(question_bank)

    examples: list[dict[str, Any]] = []
    current_records: list[dict[str, Any]] = []
    for row in source_rows:
        metrics = metrics_from_request(twin_row_to_request(row))
        metrics_dict = asdict(metrics)
        example = {
            "id": str(row["id"]),
            "benchmark": str(row["benchmark"]),
            "instance_id": str(row.get("instance_id", row["id"])),
            "step_index": int(row.get("step_index", 1)),
            "total_steps": int(row.get("total_steps", 1)),
            "gold_tier_id": _merged_id_for_gold(row),
            "metrics": metrics_dict,
        }
        examples.append(example)
        current_records.append(
            {
                "id": example["id"],
                "benchmark": example["benchmark"],
                "instance_id": example["instance_id"],
                "step_index": example["step_index"],
                "total_steps": example["total_steps"],
                "gold_merged_tier_id": example["gold_tier_id"],
                "current_prediction": route_metrics(metrics)["target_tier"],
            }
        )

    nested_predictions, nested_details = _nested_predictions(
        examples,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    lobo_predictions, lobo_details = _leave_one_benchmark_out(
        examples, inner_folds=args.inner_folds, seed=args.seed
    )
    nested_field = "optimized_prediction"
    lobo_field = "optimized_lobo_prediction"
    nested_records = _prediction_records(examples, nested_predictions, nested_field)
    lobo_records = _prediction_records(examples, lobo_predictions, lobo_field)

    if args.skip_official:
        current_official: dict[str, Any] = {}
        optimized_official: dict[str, Any] = {}
        current_official_mean = None
        optimized_official_mean = None
    else:
        current_official = _official_policy_summaries(
            source_rows,
            {record["id"]: record for record in current_records},
            "current_prediction",
            twin_repo,
        )
        optimized_official = _official_policy_summaries(
            source_rows,
            {record["id"]: record for record in nested_records},
            nested_field,
            twin_repo,
        )
        current_official_mean = _official_mean(
            current_official, "combined_score_percent"
        )
        optimized_official_mean = _official_mean(
            optimized_official, "combined_score_percent"
        )

    final_config, final_search = optimize_policy(
        examples,
        _inner_fold_ids(examples, args.inner_folds, args.seed + 999),
    )
    final_metrics_proxy = evaluate_config(examples, final_config)
    summary: dict[str, Any] = {
        "dataset": {
            "question_bank": str(question_bank),
            "question_bank_sha256": hashlib.sha256(question_bank.read_bytes()).hexdigest(),
            "rows": len(examples),
            "trajectories": len({example["instance_id"] for example in examples}),
            "benchmarks": dict(sorted(Counter(example["benchmark"] for example in examples).items())),
        },
        "search": {
            "adjustable_weight_families": 15,
            "adjustable_thresholds": 2,
            "search_space": {name: list(values) for name, values in SEARCH_SPACE.items()},
            "step_size": 5,
            "constraints": [
                "non-negative weights",
                "light <= analytic <= deep <= debug intent weights",
                "simple action <= chained action",
                "economical threshold < strongest threshold",
                "risk and capability floors preserved",
            ],
            "objective": (
                "mean of exact tier, safe steps, row-weighted trajectory pass, and "
                "failure-aware tier-price savings proxy; penalized below frozen safety and "
                "regularized toward frozen weights"
            ),
        },
        "frozen_current_router": {
            "merged_three_tier": _metrics_with_ci(
                current_records, "current_prediction", args.bootstrap_samples
            ),
            "official_four_tier": current_official,
            "mean_official_combined_score_percent": current_official_mean,
        },
        "nested_grouped_oof": {
            "merged_three_tier": _metrics_with_ci(
                nested_records, nested_field, args.bootstrap_samples
            ),
            "official_four_tier": optimized_official,
            "mean_official_combined_score_percent": optimized_official_mean,
            "details": nested_details,
        },
        "leave_one_benchmark_out": {
            "merged_three_tier": _metrics_with_ci(
                lobo_records, lobo_field, args.bootstrap_samples
            ),
            "details": lobo_details,
        },
        "final_fit": {
            "config": final_config.to_dict(),
            "search": final_search,
            "training_proxy_metrics": final_metrics_proxy,
            "warning": "In-sample fit on all public rows; not an evaluation estimate.",
        },
    }

    artifact = {
        "policy_type": "optimized_deterministic_v3",
        "policy_version": "complexity-v3-twin-nested-2026-08-22",
        "config": final_config.to_dict(),
        "score_components": [
            "context",
            "intent",
            "reasoning modifier",
            "action and testing",
            "scope",
            "special requirements",
            "software-engineering interaction",
            "specialist interaction",
        ],
        "training_provenance": {
            "question_bank_sha256": summary["dataset"]["question_bank_sha256"],
            "rows": len(examples),
            "trajectories": summary["dataset"]["trajectories"],
        },
        "evaluation_artifact": "evaluation_summary.json",
    }
    _write_json(output_dir / "optimized_policy.json", artifact)
    _write_json(output_dir / "evaluation_summary.json", summary)
    with (output_dir / "nested_oof_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for example, prediction in zip(examples, nested_predictions):
            handle.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "benchmark": example["benchmark"],
                        "instance_id": example["instance_id"],
                        "gold_merged_tier": ROUTER_TIERS[example["gold_tier_id"]],
                        "optimized_score": prediction["score"],
                        "optimized_tier": prediction["tier"],
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
    _report(output_dir / "report.md", summary)
    print(
        json.dumps(
            _safe_json(
                {
                    "frozen": {
                        key: summary["frozen_current_router"]["merged_three_tier"][key]
                        for key in (
                            "exact_rate_percent",
                            "safe_step_rate_percent",
                            "under_routing_rate_percent",
                            "over_routing_rate_percent",
                            "row_weighted_trajectory_pass_rate_percent",
                        )
                    },
                    "nested_optimized": {
                        key: summary["nested_grouped_oof"]["merged_three_tier"][key]
                        for key in (
                            "exact_rate_percent",
                            "safe_step_rate_percent",
                            "under_routing_rate_percent",
                            "over_routing_rate_percent",
                            "row_weighted_trajectory_pass_rate_percent",
                        )
                    },
                    "lobo_safe_percent": summary["leave_one_benchmark_out"][
                        "merged_three_tier"
                    ]["safe_step_rate_percent"],
                    "final_config": final_config.to_dict(),
                    "mean_official_combined": {
                        "frozen": current_official_mean,
                        "nested_optimized": optimized_official_mean,
                    },
                    "output_dir": str(output_dir),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
