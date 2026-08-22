"""Leakage-resistant 15-feature contract for learned routing models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


INTENT_CATEGORIES = (
    "extract_or_classify",
    "rewrite_or_format",
    "summarize",
    "compare",
    "analyze",
    "research_or_synthesize",
    "plan_or_design",
    "debug_prove_or_diagnose",
)

ACTION_CATEGORIES = (
    "no_tool",
    "one_read_only",
    "search_or_multiple_reads",
    "one_local_write",
    "multiple_writes_or_chained_tools",
)

NUMERIC_FEATURES = (
    "log_input_tokens",
    "log_history_items",
    "log_total_tool_calls",
    "log_unique_tools",
    "log_tool_schema_chars",
    "log_dependent_steps",
    "scope_level",
    "log_artifact_count",
    "log_deliverable_count",
    "log_hard_constraint_count",
    "reasoning_complexity_count",
    "specialist_work_count",
    "special_requirement_count",
)

RAW_FEATURES = NUMERIC_FEATURES + ("primary_intent", "expected_action")

EXCLUDED_SOURCE_FIELDS = {
    "request_model": "historical-policy leakage",
    "serialized_character_count": "duplicate of estimated_input_tokens",
    "estimated_uncached_tokens_after_switch": "initial-route duplicate of input size",
    "explicit_requirement_term_count": "duplicate of hard_constraint_count",
    "distinct_referenced_artifact_count": "duplicate of artifact_object_count",
    "tool_output_count": "redundant with completed tool-call history",
    "tool_definition_count": "near-duplicate of tool_schema_character_count",
    "tool_call_sequence": "high-cardinality domain identifier",
    "tool_argument_fingerprints": "high-cardinality identifier",
    "tool_output_statuses": "runtime outcome, excluded from initial difficulty",
    "matched_requirement_terms": "represented by hard_constraint_count",
    "shared_prefix_tokens": "cost rather than task difficulty",
    "switch_count": "cost rather than task difficulty",
    "reasoning_item_count": "can reveal the historical provider family",
}


def _value(metrics: Mapping[str, Any], name: str, default: Any = 0) -> Any:
    value = metrics.get(name, default)
    return default if value is None else value


def _log_count(value: Any) -> float:
    return math.log1p(max(0.0, float(value or 0)))


def _scope_level(metrics: Mapping[str, Any]) -> float:
    if bool(_value(metrics, "is_multi_file_modification")) or bool(
        _value(metrics, "has_stateful_coordination")
    ):
        return 3.0
    if (
        bool(_value(metrics, "has_cross_file_or_system_dependency"))
        or int(_value(metrics, "artifact_object_count", 1)) >= 4
        or int(_value(metrics, "deliverable_count", 1)) >= 4
    ):
        return 2.0
    if (
        int(_value(metrics, "artifact_object_count", 1)) >= 2
        or int(_value(metrics, "deliverable_count", 1)) >= 2
    ):
        return 1.0
    return 0.0


def reduced_features(metrics: Mapping[str, Any] | Any) -> dict[str, float | str]:
    """Collapse RouteMetrics into the fixed 15-feature learning contract."""

    if not isinstance(metrics, Mapping):
        metrics = asdict(metrics)
    reasoning_count = sum(
        bool(_value(metrics, name))
        for name in (
            "has_cross_reference_or_synthesis",
            "has_tradeoffs_or_competing_objectives",
            "has_counterfactual_uncertainty_or_scenarios",
            "has_contradiction_or_ambiguity_to_resolve",
        )
    )
    specialist_count = sum(
        bool(_value(metrics, name))
        for name in (
            "has_code_sql_formula_or_data_transformation",
            "has_debug_error_failing_test_or_stacktrace",
            "has_architecture_migration_concurrency_or_integration",
            "has_formal_math_algorithms_security_or_specialist_domain",
            "has_cross_domain_reasoning",
        )
    )
    special_count = sum(
        bool(_value(metrics, name))
        for name in (
            "requires_testing_or_verification",
            "requires_tool_chaining",
            "requires_strict_schema_or_machine_output",
            "requires_citations_or_source_traceability",
            "requires_compatibility_style_template_or_behavior_preservation",
            "requires_mutually_consistent_outputs",
            "has_exact_numerical_or_factual_acceptance_criteria",
        )
    )
    return {
        "log_input_tokens": _log_count(_value(metrics, "estimated_input_tokens")),
        "log_history_items": _log_count(_value(metrics, "input_item_count")),
        "log_total_tool_calls": _log_count(
            int(_value(metrics, "function_tool_call_count"))
            + int(_value(metrics, "custom_tool_call_count"))
        ),
        "log_unique_tools": _log_count(_value(metrics, "unique_tool_count")),
        "log_tool_schema_chars": _log_count(
            _value(metrics, "tool_schema_character_count")
        ),
        "log_dependent_steps": _log_count(_value(metrics, "dependent_step_count")),
        "scope_level": _scope_level(metrics),
        "log_artifact_count": _log_count(_value(metrics, "artifact_object_count", 1)),
        "log_deliverable_count": _log_count(_value(metrics, "deliverable_count", 1)),
        "log_hard_constraint_count": _log_count(
            _value(metrics, "hard_constraint_count")
        ),
        "reasoning_complexity_count": float(reasoning_count),
        "specialist_work_count": float(specialist_count),
        "special_requirement_count": float(special_count),
        "primary_intent": str(_value(metrics, "primary_intent", INTENT_CATEGORIES[0])),
        "expected_action": str(_value(metrics, "expected_action", ACTION_CATEGORIES[0])),
    }


@dataclass
class FeatureEncoder:
    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, rows: Iterable[Mapping[str, float | str]]) -> "FeatureEncoder":
        materialized = list(rows)
        if not materialized:
            raise ValueError("cannot fit a feature encoder without rows")
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for name in NUMERIC_FEATURES:
            values = [float(row[name]) for row in materialized]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            means[name] = mean
            scales[name] = max(math.sqrt(variance), 1e-8)
        return cls(means=means, scales=scales)

    @property
    def feature_names(self) -> tuple[str, ...]:
        intent_columns = tuple(f"intent={value}" for value in INTENT_CATEGORIES[1:])
        action_columns = tuple(f"action={value}" for value in ACTION_CATEGORIES[1:])
        return NUMERIC_FEATURES + intent_columns + action_columns

    @property
    def input_dimension(self) -> int:
        return len(self.feature_names)

    def transform_one(self, row: Mapping[str, float | str]) -> list[float]:
        values = [
            (float(row[name]) - self.means[name]) / self.scales[name]
            for name in NUMERIC_FEATURES
        ]
        intent = str(row["primary_intent"])
        action = str(row["expected_action"])
        values.extend(float(intent == category) for category in INTENT_CATEGORIES[1:])
        values.extend(float(action == category) for category in ACTION_CATEGORIES[1:])
        return values

    def transform(self, rows: Iterable[Mapping[str, float | str]]) -> list[list[float]]:
        return [self.transform_one(row) for row in rows]

    def to_dict(self) -> dict[str, Any]:
        return {
            "means": self.means,
            "scales": self.scales,
            "feature_names": list(self.feature_names),
            "raw_feature_count": len(RAW_FEATURES),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureEncoder":
        return cls(
            means={name: float(value) for name, value in payload["means"].items()},
            scales={name: float(value) for name, value in payload["scales"].items()},
        )

