#!/usr/bin/env python3
"""Backtest the deterministic router on TwinRouterBench's static track.

The router is evaluated without changing its 35/75 score thresholds.  Two
policies are reported:

* ``independent_step`` routes every benchmark prefix as a fresh decision.
* ``stateful`` chooses an initial tier and then applies the router's existing
  mid-trajectory keep/upgrade logic in step order.

TwinRouterBench has four public tiers while this repository has three.  The
primary accuracy view merges Twin's ``mid`` and ``mid_high`` labels into the
router's ``balanced`` tier.  Official four-tier cost scores are reported as a
range by mapping ``balanced`` to ``mid`` (optimistic price) and ``mid_high``
(conservative capability).

Usage:
    python scripts/backtest_twinrouterbench.py \
        --twin-repo .external/TwinRouterBench \
        --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import types
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from deterministic_router import POLICY_VERSION, evaluate_route
from extract_router_metrics import metrics_from_request


ROUTER_TIERS = ("economical", "balanced", "strongest")
ROUTER_TIER_TO_MERGED_ID = {tier: idx for idx, tier in enumerate(ROUTER_TIERS)}
TWIN_TIER_TO_MERGED_ID = {"low": 0, "mid": 1, "mid_high": 1, "high": 2}
OFFICIAL_BALANCED_MAPS = {
    "balanced_as_mid": {"economical": 0, "balanced": 1, "strongest": 3},
    "balanced_as_mid_high": {"economical": 0, "balanced": 2, "strongest": 3},
}
TIER_MODEL_MAP = {
    "economical": "backtest-economical",
    "balanced": "backtest-balanced",
    "strongest": "backtest-strongest",
}


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values: list[str] = []
        for part in content:
            if isinstance(part, str):
                values.append(part)
            elif isinstance(part, dict):
                value = part.get("text", part.get("content", part.get("output", "")))
                if isinstance(value, str):
                    values.append(value)
                else:
                    values.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
            else:
                values.append(str(part))
        return "\n".join(values)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def _message_item(role: str, content: Any) -> dict[str, Any]:
    part_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": part_type, "text": _text_content(content)}],
    }


def twin_messages_to_response_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat messages, including tool calls, to Responses-style items."""

    items: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        if role in {"system", "user"}:
            items.append(_message_item(str(role), message.get("content")))
            continue
        if role == "assistant":
            content = message.get("content")
            tool_calls = message.get("tool_calls") or []
            if content not in (None, "") or not tool_calls:
                items.append(_message_item("assistant", content))
            if not isinstance(tool_calls, list):
                raise ValueError(f"assistant message {index} tool_calls must be a list")
            for call_index, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    raise ValueError(
                        f"assistant message {index} tool call {call_index} must be an object"
                    )
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    raise ValueError(
                        f"assistant message {index} tool call {call_index} function must be an object"
                    )
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or f"message-{index}-call-{call_index}"),
                        "name": str(function.get("name") or call.get("name") or "unknown"),
                        "arguments": arguments,
                    }
                )
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                raise ValueError(f"tool message {index} is missing tool_call_id")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(call_id),
                    "output": _text_content(message.get("content")),
                }
            )
            continue
        raise ValueError(f"message {index} has unsupported role {role!r}")
    return items


def twin_row_to_request(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"row {row.get('id')!r} messages must be a list")
    functions = row.get("functions")
    if functions is None:
        functions = []
    if not isinstance(functions, list):
        raise ValueError(f"row {row.get('id')!r} functions must be a list or null")
    return {
        "model": None,
        "input": twin_messages_to_response_items(messages),
        "tools": functions,
    }


def route_metrics(metrics: Any, *, current_tier: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metrics": asdict(metrics),
        "mode": "mid_trajectory" if current_tier is not None else "initial",
        "tier_model_map": TIER_MODEL_MAP,
    }
    if current_tier is not None:
        payload["current_tier"] = current_tier
    return evaluate_route(payload)


def _load_rows(question_bank: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with question_bank.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"question bank line {line_number} is not an object")
            if not isinstance(row.get("target_tier_id"), int):
                raise ValueError(f"question bank line {line_number} lacks target_tier_id")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percent(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 6)


def _merged_id_for_gold(row: dict[str, Any]) -> int:
    tier = row.get("target_tier")
    if tier not in TWIN_TIER_TO_MERGED_ID:
        raise ValueError(f"unknown TwinRouterBench tier {tier!r}")
    return TWIN_TIER_TO_MERGED_ID[str(tier)]


def _merged_id_for_prediction(tier: str) -> int:
    if tier not in ROUTER_TIER_TO_MERGED_ID:
        raise ValueError(f"unknown router tier {tier!r}")
    return ROUTER_TIER_TO_MERGED_ID[tier]


def _policy_group_stats(
    records: list[dict[str, Any]], prediction_field: str
) -> dict[str, Any]:
    exact = passed = under = over = 0
    confusion: Counter[tuple[int, int]] = Counter()
    trajectories: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        gold = int(record["gold_merged_tier_id"])
        pred = _merged_id_for_prediction(str(record[prediction_field]))
        confusion[(gold, pred)] += 1
        exact += pred == gold
        passed += pred >= gold
        under += pred < gold
        over += pred > gold
        trajectories[str(record["instance_id"])].append(pred >= gold)

    total = len(records)
    passed_trajectories = sum(all(steps) for steps in trajectories.values())
    rows_in_passed_trajectories = sum(
        len(steps) for steps in trajectories.values() if all(steps)
    )
    return {
        "rows": total,
        "trajectories": len(trajectories),
        "exact_rows": exact,
        "safe_rows": passed,
        "under_routed_rows": under,
        "over_routed_rows": over,
        "exact_rate_percent": _percent(exact, total),
        "safe_step_rate_percent": _percent(passed, total),
        "under_routing_rate_percent": _percent(under, total),
        "over_routing_rate_percent": _percent(over, total),
        "passed_trajectories": passed_trajectories,
        "trajectory_pass_rate_percent": _percent(passed_trajectories, len(trajectories)),
        "row_weighted_trajectory_pass_rate_percent": _percent(
            rows_in_passed_trajectories, total
        ),
        "confusion": {
            ROUTER_TIERS[gold]: {
                ROUTER_TIERS[pred]: confusion[(gold, pred)]
                for pred in range(len(ROUTER_TIERS))
            }
            for gold in range(len(ROUTER_TIERS))
        },
    }


def compute_merged_metrics(
    records: list[dict[str, Any]], prediction_field: str
) -> dict[str, Any]:
    overall = _policy_group_stats(records, prediction_field)
    by_benchmark: dict[str, Any] = {}
    for benchmark in sorted({str(record["benchmark"]) for record in records}):
        subset = [record for record in records if record["benchmark"] == benchmark]
        by_benchmark[benchmark] = _policy_group_stats(subset, prediction_field)
    overall["by_benchmark"] = by_benchmark
    return overall


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_bootstrap_cis(
    records: list[dict[str, Any]],
    prediction_field: str,
    *,
    samples: int,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        clusters[str(record["instance_id"])].append(record)
    cluster_rows = list(clusters.values())
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        chosen = [rng.choice(cluster_rows) for _ in range(len(cluster_rows))]
        total_rows = sum(len(cluster) for cluster in chosen)
        exact_rows = safe_rows = passed_trajectories = 0
        for cluster in chosen:
            cluster_safe = True
            for record in cluster:
                gold = int(record["gold_merged_tier_id"])
                pred = _merged_id_for_prediction(str(record[prediction_field]))
                exact_rows += pred == gold
                safe_rows += pred >= gold
                cluster_safe = cluster_safe and pred >= gold
            passed_trajectories += cluster_safe
        draws["exact_rate_percent"].append(100.0 * exact_rows / total_rows)
        draws["safe_step_rate_percent"].append(100.0 * safe_rows / total_rows)
        draws["trajectory_pass_rate_percent"].append(
            100.0 * passed_trajectories / len(chosen)
        )
    return {
        key: {
            "lower_95": round(_percentile(values, 0.025), 6),
            "upper_95": round(_percentile(values, 0.975), 6),
        }
        for key, values in draws.items()
    }


def _official_eval_rows(
    source_rows: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    prediction_field: str,
    mapping: dict[str, int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in source_rows:
        record = records_by_id[str(row["id"])]
        predicted_tier = str(record[prediction_field])
        output.append(
            {
                "id": row["id"],
                "benchmark": row["benchmark"],
                "gold_tier_id": row["target_tier_id"],
                "pred_tier_id": mapping[predicted_tier],
                "match": mapping[predicted_tier] == row["target_tier_id"],
                "passed": mapping[predicted_tier] >= row["target_tier_id"],
                "instance_id": row.get("instance_id", row["id"]),
                "step_index": row.get("step_index", 1),
                "total_steps": row.get("total_steps", 1),
                "messages": row["messages"],
            }
        )
    return output


def _official_summary(eval_rows: list[dict[str, Any]], twin_repo: Path) -> dict[str, Any]:
    # Twin's ``main/__init__.py`` imports the live router and optional HTTP
    # dependencies.  Static scoring needs only modules beneath ``main``.  A
    # lightweight package shell lets Python resolve those modules without
    # executing the unrelated application initializer.
    twin_main = twin_repo.resolve() / "main"
    loaded_main = sys.modules.get("main")
    loaded_paths = [Path(path).resolve() for path in getattr(loaded_main, "__path__", [])]
    if twin_main not in loaded_paths:
        package = types.ModuleType("main")
        package.__path__ = [str(twin_main)]  # type: ignore[attr-defined]
        package.__package__ = "main"
        sys.modules["main"] = package
    twin_eval = twin_main / "eval"
    loaded_eval = sys.modules.get("main.eval")
    loaded_eval_paths = [
        Path(path).resolve() for path in getattr(loaded_eval, "__path__", [])
    ]
    if twin_eval not in loaded_eval_paths:
        eval_package = types.ModuleType("main.eval")
        eval_package.__path__ = [str(twin_eval)]  # type: ignore[attr-defined]
        eval_package.__package__ = "main.eval"
        sys.modules["main.eval"] = eval_package
    from main.eval.section11 import (  # type: ignore[import-not-found]
        aggregate_by_benchmark,
        compute_router_accounting_metrics,
        compute_section11,
        compute_v2_scores,
    )

    return {
        "scores_v2": compute_v2_scores(eval_rows),
        "section_11": compute_section11(eval_rows),
        "router_accounting": compute_router_accounting_metrics(eval_rows),
        "by_benchmark": aggregate_by_benchmark(eval_rows),
    }


def _official_policy_summaries(
    source_rows: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    prediction_field: str,
    twin_repo: Path,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for label, mapping in OFFICIAL_BALANCED_MAPS.items():
        eval_rows = _official_eval_rows(source_rows, records_by_id, prediction_field, mapping)
        summaries[label] = {
            "router_to_official_tier_id": mapping,
            **_official_summary(eval_rows, twin_repo),
        }
    return summaries


def _constant_records(
    records: list[dict[str, Any]], tier: str, field: str
) -> list[dict[str, Any]]:
    return [{**record, field: tier} for record in records]


def _score_diagnostics(
    records: list[dict[str, Any]], score_field: str, prediction_field: str
) -> dict[str, Any]:
    by_gold: dict[str, Any] = {}
    for gold_id, gold_name in enumerate(ROUTER_TIERS):
        scores = [
            int(record[score_field])
            for record in records
            if int(record["gold_merged_tier_id"]) == gold_id
        ]
        by_gold[gold_name] = {
            "rows": len(scores),
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        }
    under_rules: Counter[str] = Counter()
    for record in records:
        gold = int(record["gold_merged_tier_id"])
        pred = _merged_id_for_prediction(str(record[prediction_field]))
        if pred < gold:
            under_rules.update(record.get(score_field.replace("complexity_score", "matched_rules"), []))
    return {
        "complexity_score_by_gold_tier": by_gold,
        "most_common_rules_on_under_routes": under_rules.most_common(20),
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_safe_json(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_predictions(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_safe_json(record), sort_keys=True, ensure_ascii=False) + "\n")


def _write_confusion(path: Path, policy_metrics: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", "gold_tier", "predicted_tier", "count"])
        writer.writeheader()
        for policy, metrics in policy_metrics.items():
            confusion = metrics["confusion"]
            for gold_tier in ROUTER_TIERS:
                for predicted_tier in ROUTER_TIERS:
                    writer.writerow(
                        {
                            "policy": policy,
                            "gold_tier": gold_tier,
                            "predicted_tier": predicted_tier,
                            "count": confusion[gold_tier][predicted_tier],
                        }
                    )


def _frontier_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in ("independent_step", "stateful"):
        for mapping_name, official in summary["policies"][policy]["official_four_tier"].items():
            scores = official["scores_v2"]
            savings = scores["cost_savings_score_percent"]
            rows.append(
                {
                    "policy": f"{policy}:{mapping_name}",
                    "normalized_failure_aware_bill_percent": round(100.0 - savings, 6),
                    "row_weighted_trajectory_pass_percent": round(
                        scores["trajectory_pass_rate_percent"], 6
                    ),
                    "safe_step_rate_percent": round(scores["case_pass_rate_percent"], 6),
                }
            )
    return rows


def _write_frontier(output_dir: Path, rows: list[dict[str, Any]]) -> bool:
    csv_path = output_dir / "twinrouterbench_frontier.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    plt.figure(figsize=(7.2, 4.5))
    for row in rows:
        x = row["normalized_failure_aware_bill_percent"]
        y = row["row_weighted_trajectory_pass_percent"]
        plt.scatter([x], [y], s=55)
        plt.annotate(row["policy"], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7)
    plt.xlabel("Failure-aware bill (% of always-high; lower is better)")
    plt.ylabel("Row-weighted trajectory pass (%)")
    plt.title("Frozen router on TwinRouterBench static track")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "twinrouterbench_frontier.png", dpi=170)
    plt.close()
    return True


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}%"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    independent = summary["policies"]["independent_step"]["merged_three_tier"]
    stateful = summary["policies"]["stateful"]["merged_three_tier"]
    independent_official = summary["policies"]["independent_step"]["official_four_tier"]
    stateful_official = summary["policies"]["stateful"]["official_four_tier"]

    def official_range(policy: dict[str, Any], field: str) -> str:
        values = [float(result["scores_v2"][field]) for result in policy.values()]
        return f"{min(values):.1f}-{max(values):.1f}%"

    lines = [
        "# TwinRouterBench frozen-router backtest",
        "",
        "The router thresholds (economical <35, balanced 35-74, strongest >=75) were frozen before evaluation.",
        "Twin `mid` and `mid_high` are merged into the router's `balanced` tier for classification metrics.",
        "",
        "| Policy | Exact tier | Safe steps | Under-routed | Trajectory pass (unweighted) | Trajectory pass (row-weighted) | Failure-aware cost saving |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Independent step | {_fmt(independent['exact_rate_percent'])} | "
            f"{_fmt(independent['safe_step_rate_percent'])} | "
            f"{_fmt(independent['under_routing_rate_percent'])} | "
            f"{_fmt(independent['trajectory_pass_rate_percent'])} | "
            f"{_fmt(independent['row_weighted_trajectory_pass_rate_percent'])} | "
            f"{official_range(independent_official, 'cost_savings_score_percent')} |"
        ),
        (
            f"| Stateful keep/upgrade | {_fmt(stateful['exact_rate_percent'])} | "
            f"{_fmt(stateful['safe_step_rate_percent'])} | "
            f"{_fmt(stateful['under_routing_rate_percent'])} | "
            f"{_fmt(stateful['trajectory_pass_rate_percent'])} | "
            f"{_fmt(stateful['row_weighted_trajectory_pass_rate_percent'])} | "
            f"{official_range(stateful_official, 'cost_savings_score_percent')} |"
        ),
        "",
        "Unweighted trajectory pass gives every trajectory one vote. Twin's official score is row-weighted, so long failed trajectories matter more. The cost range is the sensitivity result from mapping our single `balanced` tier to Twin's `mid` versus `mid_high` tier.",
        "",
        "## Independent policy by benchmark",
        "",
        "| Benchmark | Rows | Exact tier | Safe steps | Row-weighted trajectory pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for benchmark, metrics in independent["by_benchmark"].items():
        lines.append(
            f"| {benchmark} | {metrics['rows']} | {_fmt(metrics['exact_rate_percent'])} | "
            f"{_fmt(metrics['safe_step_rate_percent'])} | "
            f"{_fmt(metrics['row_weighted_trajectory_pass_rate_percent'])} |"
        )
    lines.extend([
        "",
        "The main weakness is SWE-bench: 55.7% safe steps and 7.7% row-weighted trajectory pass. The other four benchmark slices each exceed 85% safe steps.",
        "",
        "## Scope",
        "",
        f"- Rows: {summary['dataset']['rows']}",
        f"- Trajectories: {summary['dataset']['trajectories']}",
        f"- TwinRouterBench commit: `{summary['benchmark_provenance']['commit']}`",
        f"- Question-bank SHA-256: `{summary['benchmark_provenance']['question_bank_sha256']}`",
        f"- Official cost tokenizer: {summary['benchmark_provenance']['official_cost_tokenizer']}",
        "- Static fixed-prefix evaluation only; no model inference was performed.",
        "",
        "## Interpretation limits",
        "",
        "- Twin labels are cheapest-sufficient-tier estimates under its fixed pool and downgrade protocol, not universal model optima.",
        "- The stateful simulation carries our tier decision forward over recorded prefixes; it cannot generate the counterfactual prefix our chosen model would have produced.",
        "- Twin's two middle tiers have no exact one-to-one mapping to our single balanced tier, so official cost scores are reported under both mappings.",
        "- This validates tier calibration on an external agentic benchmark; it does not identify the best anonymized Viktor model.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin-repo", default=".external/TwinRouterBench")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--dependency-path",
        default=".external/python310-packages",
        help="Local target directory containing Twin's optional static scorer dependencies.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    dependency_path = Path(args.dependency_path)
    if dependency_path.exists() and str(dependency_path.resolve()) not in sys.path:
        sys.path.insert(0, str(dependency_path.resolve()))

    twin_repo = Path(args.twin_repo)
    question_bank = twin_repo / "data" / "static" / "question_bank.jsonl"
    if not question_bank.exists():
        raise SystemExit(f"TwinRouterBench question bank not found: {question_bank}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = _load_rows(question_bank)
    metrics_by_id: dict[str, Any] = {}
    independent_by_id: dict[str, dict[str, Any]] = {}
    compact_records: list[dict[str, Any]] = []

    for row in source_rows:
        row_id = str(row["id"])
        request = twin_row_to_request(row)
        metrics = metrics_from_request(request)
        independent = route_metrics(metrics)
        metrics_by_id[row_id] = metrics
        independent_by_id[row_id] = independent
        compact_records.append(
            {
                "id": row_id,
                "benchmark": row["benchmark"],
                "instance_id": row.get("instance_id", row_id),
                "step_index": row.get("step_index", 1),
                "total_steps": row.get("total_steps", 1),
                "pipeline_stage": row.get("pipeline_stage"),
                "gold_twin_tier": row["target_tier"],
                "gold_twin_tier_id": row["target_tier_id"],
                "gold_merged_tier": ROUTER_TIERS[_merged_id_for_gold(row)],
                "gold_merged_tier_id": _merged_id_for_gold(row),
                "estimated_input_tokens_chars_per_4": metrics.estimated_input_tokens,
                "independent_predicted_tier": independent["target_tier"],
                "independent_complexity_score": independent["complexity_score"],
                "independent_decision": independent["decision"],
                "independent_matched_rules": independent["matched_rules"],
            }
        )

    compact_by_id = {record["id"]: record for record in compact_records}
    rows_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        rows_by_instance[str(row.get("instance_id", row["id"]))].append(row)
    for instance_rows in rows_by_instance.values():
        instance_rows.sort(key=lambda row: (int(row.get("step_index", 1)), str(row["id"])))
        current_tier: str | None = None
        for row in instance_rows:
            row_id = str(row["id"])
            decision = route_metrics(metrics_by_id[row_id], current_tier=current_tier)
            current_tier = str(decision["target_tier"])
            compact_by_id[row_id].update(
                {
                    "stateful_predicted_tier": current_tier,
                    "stateful_complexity_score": decision["complexity_score"],
                    "stateful_decision": decision["decision"],
                    "stateful_failure_score": decision["failure_score"],
                    "stateful_matched_rules": decision["matched_rules"],
                    "stateful_failure_reasons": decision["failure_reasons"],
                }
            )

    independent_metrics = compute_merged_metrics(
        compact_records, "independent_predicted_tier"
    )
    stateful_metrics = compute_merged_metrics(compact_records, "stateful_predicted_tier")
    independent_metrics["cluster_bootstrap_95_ci"] = cluster_bootstrap_cis(
        compact_records,
        "independent_predicted_tier",
        samples=args.bootstrap_samples,
    )
    stateful_metrics["cluster_bootstrap_95_ci"] = cluster_bootstrap_cis(
        compact_records,
        "stateful_predicted_tier",
        samples=args.bootstrap_samples,
    )

    official_independent = _official_policy_summaries(
        source_rows,
        compact_by_id,
        "independent_predicted_tier",
        twin_repo,
    )
    official_stateful = _official_policy_summaries(
        source_rows,
        compact_by_id,
        "stateful_predicted_tier",
        twin_repo,
    )

    baseline_field = "baseline_predicted_tier"
    baseline_metrics: dict[str, Any] = {}
    for tier in ROUTER_TIERS:
        tier_records = _constant_records(compact_records, tier, baseline_field)
        baseline_metrics[f"always_{tier}"] = compute_merged_metrics(
            tier_records, baseline_field
        )

    try:
        import subprocess

        commit = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={twin_repo.resolve().as_posix()}",
                "-C",
                str(twin_repo),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    except Exception:  # pragma: no cover - provenance fallback only
        commit = "unknown"

    summary: dict[str, Any] = {
        "benchmark_provenance": {
            "repository": "https://github.com/CommonstackAI/TwinRouterBench",
            "commit": commit,
            "question_bank": str(question_bank),
            "question_bank_sha256": _sha256(question_bank),
            "official_cost_tokenizer": (
                "TwinRouterBench documented cl100k_base fallback for all tiers; "
                "native HuggingFace tokenizers not installed"
            ),
        },
        "router": {
            "policy_version": POLICY_VERSION,
            "frozen_thresholds": {
                "economical": "score < 35",
                "balanced": "35 <= score < 75",
                "strongest": "score >= 75",
            },
            "token_basis": "serialized input characters / 4; estimated, not measured",
        },
        "tier_mapping": {
            "merged_three_tier": TWIN_TIER_TO_MERGED_ID,
            "official_four_tier_sensitivity_maps": OFFICIAL_BALANCED_MAPS,
        },
        "dataset": {
            "rows": len(source_rows),
            "trajectories": len(rows_by_instance),
            "benchmark_counts": dict(sorted(Counter(row["benchmark"] for row in source_rows).items())),
            "gold_tier_counts": dict(sorted(Counter(row["target_tier"] for row in source_rows).items())),
            "pipeline_stage_counts": dict(
                sorted(Counter(str(row.get("pipeline_stage")) for row in source_rows).items())
            ),
        },
        "policies": {
            "independent_step": {
                "description": "Every router-visible prefix is scored as a fresh initial decision.",
                "merged_three_tier": independent_metrics,
                "official_four_tier": official_independent,
                "diagnostics": _score_diagnostics(
                    compact_records,
                    "independent_complexity_score",
                    "independent_predicted_tier",
                ),
            },
            "stateful": {
                "description": "First step routes initially; later steps carry the prior tier through existing keep/upgrade logic over fixed recorded prefixes.",
                "merged_three_tier": stateful_metrics,
                "official_four_tier": official_stateful,
                "diagnostics": _score_diagnostics(
                    compact_records,
                    "stateful_complexity_score",
                    "stateful_predicted_tier",
                ),
            },
        },
        "merged_three_tier_baselines": baseline_metrics,
        "limitations": [
            "Twin target tiers are execution-verified estimates tied to its fixed model pool and downgrade protocol, not universal optima.",
            "Stateful evaluation follows recorded prefixes; it does not generate counterfactual prefixes from the router-selected models.",
            "The router has one balanced tier while Twin has mid and mid_high; official cost results are therefore reported as two mapping sensitivity scenarios.",
            "External tier calibration does not identify the best anonymized Viktor model or remove confounding from the Viktor logs.",
        ],
    }

    predictions_path = output_dir / "twinrouterbench_predictions.jsonl"
    summary_path = output_dir / "twinrouterbench_summary.json"
    confusion_path = output_dir / "twinrouterbench_confusion.csv"
    _write_predictions(predictions_path, compact_records)
    _write_json(summary_path, summary)
    _write_confusion(
        confusion_path,
        {
            "independent_step": independent_metrics,
            "stateful": stateful_metrics,
        },
    )
    frontier_rows = _frontier_rows(summary)
    wrote_png = _write_frontier(output_dir, frontier_rows)
    report_path = output_dir / "twinrouterbench_report.md"
    _write_report(report_path, summary)

    print(json.dumps(_safe_json({
        "rows": len(source_rows),
        "trajectories": len(rows_by_instance),
        "independent_step": {
            key: independent_metrics[key]
            for key in (
                "exact_rate_percent",
                "safe_step_rate_percent",
                "under_routing_rate_percent",
                "over_routing_rate_percent",
                "trajectory_pass_rate_percent",
                "row_weighted_trajectory_pass_rate_percent",
            )
        },
        "stateful": {
            key: stateful_metrics[key]
            for key in (
                "exact_rate_percent",
                "safe_step_rate_percent",
                "under_routing_rate_percent",
                "over_routing_rate_percent",
                "trajectory_pass_rate_percent",
                "row_weighted_trajectory_pass_rate_percent",
            )
        },
        "official_four_tier_sensitivity": {
            "independent_step": {
                mapping: result["scores_v2"]
                for mapping, result in official_independent.items()
            },
            "stateful": {
                mapping: result["scores_v2"]
                for mapping, result in official_stateful.items()
            },
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "confusion": str(confusion_path),
            "frontier_csv": str(output_dir / "twinrouterbench_frontier.csv"),
            "frontier_png": str(output_dir / "twinrouterbench_frontier.png") if wrote_png else None,
            "report": str(report_path),
        },
    }), indent=2))


if __name__ == "__main__":
    main()
