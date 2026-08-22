#!/usr/bin/env python3
"""Train and honestly evaluate two small learned routers on TwinRouterBench.

Evaluation predictions are strictly out-of-fold by complete trajectory.  A
second leave-one-benchmark-out pass measures domain transfer.  Final models are
then fitted on all public rows for deployment; their in-sample predictions are
never reported as evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backtest_twinrouterbench import (  # noqa: E402
    OFFICIAL_BALANCED_MAPS,
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
from learned_router.features import (  # noqa: E402
    EXCLUDED_SOURCE_FIELDS,
    RAW_FEATURES,
    reduced_features,
)
from learned_router.models import (  # noqa: E402
    OrdinalBoostedRouter,
    OrdinalLogisticRouter,
)


FitFunction = Callable[
    [Sequence[Mapping[str, float | str]], Sequence[int]],
    OrdinalLogisticRouter | OrdinalBoostedRouter,
]


def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _assign_group_folds(examples: list[dict[str, Any]], folds: int, seed: int) -> dict[str, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[example["instance_id"]].append(example)
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for instance_id, group in groups.items():
        strata[(group[0]["benchmark"], max(item["gold_tier_id"] for item in group))].append(
            instance_id
        )
    assignment: dict[str, int] = {}
    fold_rows = [0] * folds
    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=lambda value: _stable_hash(value, seed))
        for instance_id in ordered:
            candidate_folds = sorted(
                range(folds),
                key=lambda fold: (fold_rows[fold], _stable_hash(f"{stratum}:{fold}", seed)),
            )
            chosen = candidate_folds[0]
            assignment[instance_id] = chosen
            fold_rows[chosen] += len(groups[instance_id])
    return assignment


def _cross_validated_predictions(
    examples: list[dict[str, Any]],
    fit: FitFunction,
    *,
    folds: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment = _assign_group_folds(examples, folds, seed)
    predictions: list[dict[str, Any] | None] = [None] * len(examples)
    fold_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(folds):
        train_indices = [
            index
            for index, example in enumerate(examples)
            if assignment[example["instance_id"]] != fold
        ]
        test_indices = [
            index
            for index, example in enumerate(examples)
            if assignment[example["instance_id"]] == fold
        ]
        model = fit(
            [examples[index]["features"] for index in train_indices],
            [examples[index]["gold_tier_id"] for index in train_indices],
        )
        for index in test_indices:
            prediction = model.predict_one(examples[index]["features"])
            predictions[index] = asdict(prediction)
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "train_trajectories": len(
                    {examples[index]["instance_id"] for index in train_indices}
                ),
                "test_trajectories": len(
                    {examples[index]["instance_id"] for index in test_indices}
                ),
                "test_gold_counts": dict(
                    sorted(Counter(examples[index]["gold_tier_id"] for index in test_indices).items())
                ),
            }
        )
    if any(prediction is None for prediction in predictions):
        raise AssertionError("some grouped out-of-fold rows were not predicted")
    return [prediction for prediction in predictions if prediction is not None], {
        "folds": fold_summaries,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _leave_one_benchmark_out_predictions(
    examples: list[dict[str, Any]], fit: FitFunction
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any] | None] = [None] * len(examples)
    split_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for benchmark in sorted({example["benchmark"] for example in examples}):
        train_indices = [
            index for index, example in enumerate(examples) if example["benchmark"] != benchmark
        ]
        test_indices = [
            index for index, example in enumerate(examples) if example["benchmark"] == benchmark
        ]
        model = fit(
            [examples[index]["features"] for index in train_indices],
            [examples[index]["gold_tier_id"] for index in train_indices],
        )
        for index in test_indices:
            predictions[index] = asdict(model.predict_one(examples[index]["features"]))
        split_summaries.append(
            {
                "held_out_benchmark": benchmark,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
            }
        )
    if any(prediction is None for prediction in predictions):
        raise AssertionError("some leave-one-benchmark-out rows were not predicted")
    return [prediction for prediction in predictions if prediction is not None], {
        "splits": split_summaries,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _records_with_predictions(
    examples: list[dict[str, Any]], predictions: Sequence[Mapping[str, Any]], field: str
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
            f"{field}_score": round(float(prediction["score"]), 6),
            f"{field}_p_balanced": round(float(prediction["probability_balanced"]), 8),
            f"{field}_p_strongest": round(float(prediction["probability_strongest"]), 8),
        }
        for example, prediction in zip(examples, predictions)
    ]


def _metrics_with_ci(
    records: list[dict[str, Any]], prediction_field: str, bootstrap_samples: int
) -> dict[str, Any]:
    metrics = compute_merged_metrics(records, prediction_field)
    metrics["cluster_bootstrap_95_ci"] = cluster_bootstrap_cis(
        records,
        prediction_field,
        samples=bootstrap_samples,
    )
    return metrics


def _constant_baseline_records(
    examples: list[dict[str, Any]], tier: str, field: str
) -> list[dict[str, Any]]:
    return [
        {
            "id": example["id"],
            "benchmark": example["benchmark"],
            "instance_id": example["instance_id"],
            "gold_merged_tier_id": example["gold_tier_id"],
            field: tier,
        }
        for example in examples
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_safe_json(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _mean_official_combined(official: Mapping[str, Any]) -> float | None:
    if not official:
        return None
    return statistics.mean(
        float(result["scores_v2"]["combined_score_percent"])
        for result in official.values()
    )


def _report(path: Path, summary: dict[str, Any]) -> None:
    model_names = ("ordinal_logistic", "ordinal_boosted_trees")
    lines = [
        "# Learned-router evaluation",
        "",
        "All headline predictions are out-of-fold by complete trajectory. The final saved models were trained only after evaluation predictions were complete.",
        "",
        "## Grouped five-fold results",
        "",
        "| Model | Exact tier | Safe steps | Under-routed | Over-routed | Row-weighted trajectory pass | Failure-aware cost saving | Official combined |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in model_names:
        model = summary["models"][name]
        metrics = model["grouped_oof"]["merged_three_tier"]
        official = model["grouped_oof"]["official_four_tier"]
        savings = [
            result["scores_v2"]["cost_savings_score_percent"] for result in official.values()
        ]
        saving_text = f"{min(savings):.1f}-{max(savings):.1f}%" if savings else "not run"
        combined = _mean_official_combined(official)
        combined_text = f"{combined:.1f}%" if combined is not None else "not run"
        lines.append(
            f"| {name} | {metrics['exact_rate_percent']:.1f}% | "
            f"{metrics['safe_step_rate_percent']:.1f}% | "
            f"{metrics['under_routing_rate_percent']:.1f}% | "
            f"{metrics['over_routing_rate_percent']:.1f}% | "
            f"{metrics['row_weighted_trajectory_pass_rate_percent']:.1f}% | "
            f"{saving_text} | {combined_text} |"
        )
    deterministic = summary["baselines"]["deterministic_router"]
    lines.extend(
        [
            (
                f"| frozen deterministic | {deterministic['exact_rate_percent']:.1f}% | "
                f"{deterministic['safe_step_rate_percent']:.1f}% | "
                f"{deterministic['under_routing_rate_percent']:.1f}% | "
                f"{deterministic['over_routing_rate_percent']:.1f}% | "
                f"{deterministic['row_weighted_trajectory_pass_rate_percent']:.1f}% | n/a | n/a |"
            ),
            "",
            "## Leave-one-benchmark-out stress test",
            "",
            "| Model | Exact tier | Safe steps | Under-routed | Row-weighted trajectory pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in model_names:
        metrics = summary["models"][name]["leave_one_benchmark_out"]["merged_three_tier"]
        lines.append(
            f"| {name} | {metrics['exact_rate_percent']:.1f}% | "
            f"{metrics['safe_step_rate_percent']:.1f}% | "
            f"{metrics['under_routing_rate_percent']:.1f}% | "
            f"{metrics['row_weighted_trajectory_pass_rate_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "### Safety when each benchmark is completely unseen",
            "",
            "| Held-out benchmark | Logistic safe steps | Boosted-tree safe steps |",
            "|---|---:|---:|",
        ]
    )
    logistic_lobo = summary["models"]["ordinal_logistic"]["leave_one_benchmark_out"][
        "merged_three_tier"
    ]["by_benchmark"]
    boosted_lobo = summary["models"]["ordinal_boosted_trees"]["leave_one_benchmark_out"][
        "merged_three_tier"
    ]["by_benchmark"]
    for benchmark in sorted(logistic_lobo):
        lines.append(
            f"| {benchmark} | {logistic_lobo[benchmark]['safe_step_rate_percent']:.1f}% | "
            f"{boosted_lobo[benchmark]['safe_step_rate_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected model: **{summary['selection']['winner']}**.",
            f"Selection criterion: {summary['selection']['criterion']}.",
            "This is the experiment winner, not a deployment approval: both learned models fall below the deterministic router's overall safety when the benchmark family is unseen.",
            "",
            "## Important limitations",
            "",
            "- Twin target tiers are pool- and protocol-specific cheapest-sufficient estimates.",
            "- Prefixes from one trajectory are correlated; grouped evaluation prevents them crossing train/test boundaries.",
            "- Leave-one-benchmark-out performance is the more pessimistic estimate of transfer to unseen task families.",
            "- The 35/75 thresholds and score formula were frozen before this evaluation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin-repo", default=".external/TwinRouterBench")
    parser.add_argument("--dependency-path", default=".external/python-packages")
    parser.add_argument("--output-dir", default="results/learned_router")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--boosted-trees", type=int, default=60)
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
    deterministic_records: list[dict[str, Any]] = []
    for row in source_rows:
        metrics = metrics_from_request(twin_row_to_request(row))
        gold = _merged_id_for_gold(row)
        example = {
            "id": str(row["id"]),
            "benchmark": str(row["benchmark"]),
            "instance_id": str(row.get("instance_id", row["id"])),
            "step_index": int(row.get("step_index", 1)),
            "total_steps": int(row.get("total_steps", 1)),
            "gold_tier_id": gold,
            "features": reduced_features(metrics),
        }
        examples.append(example)
        deterministic_records.append(
            {
                "id": example["id"],
                "benchmark": example["benchmark"],
                "instance_id": example["instance_id"],
                "gold_merged_tier_id": gold,
                "prediction": route_metrics(metrics)["target_tier"],
            }
        )

    fitters: dict[str, FitFunction] = {
        "ordinal_logistic": lambda features, labels: OrdinalLogisticRouter.fit(
            features, labels
        ),
        "ordinal_boosted_trees": lambda features, labels: OrdinalBoostedRouter.fit(
            features,
            labels,
            tree_count=args.boosted_trees,
        ),
    }

    model_results: dict[str, Any] = {}
    all_oof_records: dict[str, list[dict[str, Any]]] = {}
    final_models: dict[str, OrdinalLogisticRouter | OrdinalBoostedRouter] = {}
    source_by_id = {str(row["id"]): row for row in source_rows}
    for model_name, fitter in fitters.items():
        oof_predictions, grouped_details = _cross_validated_predictions(
            examples, fitter, folds=args.folds, seed=args.seed
        )
        lobo_predictions, lobo_details = _leave_one_benchmark_out_predictions(
            examples, fitter
        )
        prediction_field = f"{model_name}_predicted_tier"
        oof_records = _records_with_predictions(examples, oof_predictions, prediction_field)
        lobo_records = _records_with_predictions(
            examples, lobo_predictions, f"{model_name}_lobo_predicted_tier"
        )
        records_by_id = {record["id"]: record for record in oof_records}
        official = (
            {}
            if args.skip_official
            else _official_policy_summaries(
                source_rows,
                records_by_id,
                prediction_field,
                twin_repo,
            )
        )
        final_model = fitter(
            [example["features"] for example in examples],
            [example["gold_tier_id"] for example in examples],
        )
        final_models[model_name] = final_model
        all_oof_records[model_name] = oof_records
        model_results[model_name] = {
            "grouped_oof": {
                "merged_three_tier": _metrics_with_ci(
                    oof_records, prediction_field, args.bootstrap_samples
                ),
                "official_four_tier": official,
                "split_details": grouped_details,
            },
            "leave_one_benchmark_out": {
                "merged_three_tier": _metrics_with_ci(
                    lobo_records,
                    f"{model_name}_lobo_predicted_tier",
                    args.bootstrap_samples,
                ),
                "split_details": lobo_details,
            },
            "final_model": {
                "artifact": f"{model_name}_model.json",
                "trainable_parameter_count": getattr(
                    final_model, "trainable_parameter_count", None
                ),
                "learned_leaf_count": getattr(final_model, "learned_leaf_count", None),
            },
        }

    combined_scores = {
        name: _mean_official_combined(result["grouped_oof"]["official_four_tier"])
        for name, result in model_results.items()
    }
    if all(value is not None for value in combined_scores.values()):
        winner = max(combined_scores, key=lambda name: float(combined_scores[name]))
        criterion = "mean TwinRouterBench official combined score across both middle-tier mappings"
    else:
        winner = max(
            model_results,
            key=lambda name: model_results[name]["grouped_oof"]["merged_three_tier"][
                "row_weighted_trajectory_pass_rate_percent"
            ],
        )
        criterion = "grouped row-weighted trajectory pass (official scorer skipped)"

    question_bank_hash = hashlib.sha256(question_bank.read_bytes()).hexdigest()
    baselines = {
        "deterministic_router": compute_merged_metrics(deterministic_records, "prediction")
    }
    for tier in ROUTER_TIERS:
        field = "prediction"
        baselines[f"always_{tier}"] = compute_merged_metrics(
            _constant_baseline_records(examples, tier, field), field
        )

    summary = {
        "dataset": {
            "question_bank": str(question_bank),
            "question_bank_sha256": question_bank_hash,
            "rows": len(examples),
            "trajectories": len({example["instance_id"] for example in examples}),
            "benchmarks": dict(sorted(Counter(example["benchmark"] for example in examples).items())),
            "gold_merged_tier_counts": {
                ROUTER_TIERS[tier]: count
                for tier, count in sorted(Counter(example["gold_tier_id"] for example in examples).items())
            },
        },
        "feature_contract": {
            "raw_feature_count": len(RAW_FEATURES),
            "raw_features": list(RAW_FEATURES),
            "encoded_column_count": final_models["ordinal_logistic"].encoder.input_dimension,
            "excluded_source_fields": EXCLUDED_SOURCE_FIELDS,
        },
        "evaluation_protocol": {
            "grouped_folds": args.folds,
            "grouping_key": "instance_id / complete trajectory",
            "leave_one_benchmark_out": True,
            "seed": args.seed,
            "thresholds": {"economical_max": 34.999999, "strongest_min": 75.0},
            "thresholds_tuned_on_test": False,
            "bootstrap_samples": args.bootstrap_samples,
            "official_balanced_sensitivity_maps": OFFICIAL_BALANCED_MAPS,
        },
        "models": model_results,
        "baselines": baselines,
        "selection": {
            "winner": winner,
            "criterion": criterion,
            "mean_official_combined_scores": combined_scores,
        },
    }

    for model_name, model in final_models.items():
        artifact = {
            "training_provenance": {
                "question_bank_sha256": question_bank_hash,
                "rows": len(examples),
                "trajectories": summary["dataset"]["trajectories"],
                "note": "Final deployment fit on all rows; evaluation metrics use separate out-of-fold predictions.",
            },
            "routing_thresholds": {"economical_below": 35.0, "strongest_at_or_above": 75.0},
            "model": model.to_dict(),
        }
        _write_json(output_dir / f"{model_name}_model.json", artifact)

    merged_predictions: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        row = {
            "id": example["id"],
            "benchmark": example["benchmark"],
            "instance_id": example["instance_id"],
            "step_index": example["step_index"],
            "gold_merged_tier": ROUTER_TIERS[example["gold_tier_id"]],
        }
        for model_name, records in all_oof_records.items():
            field = f"{model_name}_predicted_tier"
            row[field] = records[index][field]
            row[f"{model_name}_score"] = records[index][f"{field}_score"]
            row[f"{model_name}_p_balanced"] = records[index][f"{field}_p_balanced"]
            row[f"{model_name}_p_strongest"] = records[index][f"{field}_p_strongest"]
        merged_predictions.append(row)
    with (output_dir / "oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged_predictions:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    _write_json(output_dir / "evaluation_summary.json", summary)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "exact_rate_percent",
            "safe_step_rate_percent",
            "under_routing_rate_percent",
            "over_routing_rate_percent",
            "row_weighted_trajectory_pass_rate_percent",
            "mean_official_combined_score_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in fitters:
            metrics = model_results[model_name]["grouped_oof"]["merged_three_tier"]
            writer.writerow(
                {
                    "model": model_name,
                    **{name: metrics[name] for name in fieldnames[1:-1]},
                    "mean_official_combined_score_percent": combined_scores[model_name],
                }
            )
    _report(output_dir / "report.md", summary)
    print(
        json.dumps(
            _safe_json(
                {
                    "winner": winner,
                    "selection_criterion": criterion,
                    "models": {
                        name: {
                            "grouped_oof": {
                                key: result["grouped_oof"]["merged_three_tier"][key]
                                for key in (
                                    "exact_rate_percent",
                                    "safe_step_rate_percent",
                                    "under_routing_rate_percent",
                                    "over_routing_rate_percent",
                                    "row_weighted_trajectory_pass_rate_percent",
                                )
                            },
                            "leave_one_benchmark_out_safe_percent": result[
                                "leave_one_benchmark_out"
                            ]["merged_three_tier"]["safe_step_rate_percent"],
                            "mean_official_combined_score_percent": combined_scores[name],
                        }
                        for name, result in model_results.items()
                    },
                    "output_dir": str(output_dir),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
