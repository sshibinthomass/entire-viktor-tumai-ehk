#!/usr/bin/env python3
"""Audit deterministic router signals without exposing raw challenge data.

Usage: python scripts/analyze_router_signals.py export/
Writes: results/router_signal_analysis.json
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from deterministic_router import evaluate_route
from extract_router_metrics import THREAD_INFO_MARKER, actionable_text, metrics_from_request
from load_trajectories import (
    first_user_text,
    group_trajectories,
    is_generated_synthetic,
    iter_requests,
)


LOGGED_TIER = {
    "claude-fable-5": 0,
    "claude-sonnet-4-6": 1,
    "claude-sonnet-5": 1,
    "claude-opus-4-6": 2,
    "claude-opus-4-8": 2,
    "claude-opus-5": 2,
    "gpt-5.6-sol": 0,
    "gpt-5.6-terra": 1,
    "gpt-5.6-luna": 2,
}


def _round(value: float) -> float:
    return round(value, 4)


def _mean(values: list[float]) -> float | None:
    return _round(statistics.mean(values)) if values else None


def _phi(left: list[bool], right: list[bool]) -> float | None:
    n11 = sum(a and b for a, b in zip(left, right))
    n10 = sum(a and not b for a, b in zip(left, right))
    n01 = sum(not a and b for a, b in zip(left, right))
    n00 = len(left) - n11 - n10 - n01
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return _round((n11 * n00 - n10 * n01) / denominator) if denominator else None


def main() -> None:
    export_dir = sys.argv[1] if len(sys.argv) > 1 else "export"
    all_requests = [request for _, _, request in iter_requests(export_dir)]
    synthetic = [request for request in all_requests if is_generated_synthetic(request)]
    requests = [request for request in all_requests if not is_generated_synthetic(request)]
    groups = group_trajectories(requests)

    metrics = [metrics_from_request(request) for request in requests]
    decisions = [evaluate_route({"metrics": asdict(item)}) for item in metrics]

    feature_functions: dict[str, Callable[[Any], bool]] = {
        "context_ge_24000_est_tokens": lambda m: m.estimated_input_tokens >= 24_000,
        "history_gt_30_items": lambda m: m.input_item_count > 30,
        "high_reasoning_intent": lambda m: m.primary_intent
        in {"research_or_synthesize", "plan_or_design", "debug_prove_or_diagnose"},
        "reasoning_modifier": lambda m: any(
            (
                m.has_tradeoffs_or_competing_objectives,
                m.has_counterfactual_uncertainty_or_scenarios,
                m.has_contradiction_or_ambiguity_to_resolve,
            )
        ),
        "multi_write_or_chained_action": lambda m: m.expected_action
        == "multiple_writes_or_chained_tools",
        "testing_or_verification": lambda m: m.requires_testing_or_verification,
        "multi_file_or_stateful": lambda m: m.is_multi_file_modification
        or m.has_stateful_coordination,
        "cross_system_dependency": lambda m: m.has_cross_file_or_system_dependency,
        "any_special_requirement": lambda m: any(
            (
                m.requires_strict_schema_or_machine_output,
                m.requires_citations_or_source_traceability,
                m.requires_compatibility_style_template_or_behavior_preservation,
                m.requires_mutually_consistent_outputs,
                m.has_exact_numerical_or_factual_acceptance_criteria,
                m.requires_ocr_spatial_or_chart_reasoning,
                m.requires_layout_sensitive_artifact,
            )
        ),
        "risk_or_security_floor": lambda m: (
            m.action_risk == "destructive_public_deployment_or_permission_change"
            or m.is_high_stakes_domain
            or m.involves_security_secrets_authentication_or_permissions
        ),
        "has_image": lambda m: m.input_image_count > 0,
    }

    feature_values = {
        name: [bool(function(item)) for item in metrics]
        for name, function in feature_functions.items()
    }
    n = len(requests)
    signal_rows: dict[str, Any] = {}
    logged_ranks = [LOGGED_TIER.get(request["model"]) for request in requests]
    scores = [int(decision["complexity_score"]) for decision in decisions]
    for name, values in feature_values.items():
        present = [index for index, value in enumerate(values) if value]
        absent = [index for index, value in enumerate(values) if not value]
        signal_rows[name] = {
            "count": len(present),
            "prevalence": _round(len(present) / n),
            "mean_router_score_present": _mean([scores[index] for index in present]),
            "mean_router_score_absent": _mean([scores[index] for index in absent]),
            "mean_logged_assumed_tier_present": _mean(
                [logged_ranks[index] for index in present if logged_ranks[index] is not None]
            ),
            "mean_logged_assumed_tier_absent": _mean(
                [logged_ranks[index] for index in absent if logged_ranks[index] is not None]
            ),
        }

    correlations = []
    names = list(feature_values)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            value = _phi(feature_values[left], feature_values[right])
            if value is not None:
                correlations.append({"left": left, "right": right, "phi": value})
    correlations.sort(key=lambda row: abs(row["phi"]), reverse=True)

    extraction_modes = Counter()
    full_lengths: list[int] = []
    active_lengths: list[int] = []
    for request in requests:
        text = first_user_text(request)
        active = actionable_text(text)
        full_lengths.append(len(text))
        active_lengths.append(len(active))
        if "</system>" in text and text.rsplit("</system>", 1)[1].strip():
            extraction_modes["tail_after_final_system"] += 1
        elif THREAD_INFO_MARKER in text:
            extraction_modes["final_thread_info_block"] += 1
        else:
            extraction_modes["full_text_fallback"] += 1

    result = {
        "scope": {
            "all_requests_seen": len(all_requests),
            "generated_synthetic_requests_excluded": len(synthetic),
            "real_requests": n,
            "reconstructed_groups": len(groups),
            "singleton_groups": sum(len(calls) == 1 for calls in groups.values()),
            "multi_request_groups": sum(len(calls) > 1 for calls in groups.values()),
            "mixed_model_groups": sum(
                len({call["model"] for call in calls}) > 1 for calls in groups.values()
            ),
        },
        "task_text_extraction": {
            "modes": dict(extraction_modes),
            "median_full_user_characters": statistics.median(full_lengths),
            "median_active_characters": statistics.median(active_lengths),
            "median_active_share": _round(
                statistics.median(
                    active / full if full else 0
                    for active, full in zip(active_lengths, full_lengths)
                )
            ),
        },
        "distributions": {
            "primary_intent": dict(Counter(item.primary_intent for item in metrics)),
            "expected_action": dict(Counter(item.expected_action for item in metrics)),
            "target_tier": dict(Counter(item["target_tier"] for item in decisions)),
        },
        "signals": signal_rows,
        "largest_absolute_phi_correlations": correlations[:10],
        "embedded_history": {
            "requests_with_tool_outputs": sum(item.tool_output_count > 0 for item in metrics),
            "requests_with_assistant_messages": sum(item.assistant_message_count > 0 for item in metrics),
            "note": "These are prior items embedded in a request, not linked next-request outcomes.",
        },
        "interpretation_warnings": [
            "Logged model tier is a treatment/selection decision, not a quality label.",
            "A feature/model association does not show that the feature needs that tier.",
            "The export has no output, usage, acceptance, success, or counterfactual labels.",
            "All token and input-cost figures are estimates; output cost is absent.",
            "Tier ordering and anonymized-model prices are assumptions.",
        ],
    }

    Path("results").mkdir(exist_ok=True)
    output = Path("results/router_signal_analysis.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
