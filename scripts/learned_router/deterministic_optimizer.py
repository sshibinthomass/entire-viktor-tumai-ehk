"""Constrained, explainable weight optimization for a deterministic router."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence


TIERS = ("economical", "balanced", "strongest")


@dataclass(frozen=True)
class PolicyConfig:
    context_unit: int = 5
    history_weight: int = 5
    light_intent_weight: int = 5
    compare_intent_weight: int = 10
    analytic_intent_weight: int = 15
    deep_intent_weight: int = 20
    debug_intent_weight: int = 30
    reasoning_modifier_weight: int = 5
    simple_action_weight: int = 5
    chained_action_weight: int = 10
    testing_weight: int = 5
    scope_unit: int = 5
    special_requirement_unit: int = 5
    software_engineering_interaction: int = 0
    specialist_interaction: int = 0
    economical_threshold: int = 35
    strongest_threshold: int = 75

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyConfig":
        return cls(**{name: int(payload[name]) for name in cls.__dataclass_fields__})


DEFAULT_CONFIG = PolicyConfig()

SEARCH_SPACE: dict[str, tuple[int, ...]] = {
    "context_unit": (0, 5, 10),
    "history_weight": (0, 5, 10),
    "light_intent_weight": (0, 5, 10),
    "compare_intent_weight": (5, 10, 15, 20),
    "analytic_intent_weight": (5, 10, 15, 20, 25),
    "deep_intent_weight": (10, 15, 20, 25, 30),
    "debug_intent_weight": (20, 25, 30, 35, 40),
    "reasoning_modifier_weight": (0, 5, 10, 15),
    "simple_action_weight": (0, 5, 10),
    "chained_action_weight": (5, 10, 15, 20),
    "testing_weight": (0, 5, 10, 15),
    "scope_unit": (0, 5, 10),
    "special_requirement_unit": (0, 5, 10),
    "software_engineering_interaction": (0, 5, 10, 15, 20, 25, 30),
    "specialist_interaction": (0, 5, 10, 15, 20),
    "economical_threshold": (30, 35, 40),
    "strongest_threshold": (65, 70, 75, 80),
}


def is_valid_config(config: PolicyConfig) -> bool:
    return (
        config.light_intent_weight <= config.compare_intent_weight
        <= config.analytic_intent_weight
        <= config.deep_intent_weight
        <= config.debug_intent_weight
        and config.simple_action_weight <= config.chained_action_weight
        and config.economical_threshold < config.strongest_threshold
        and all(value >= 0 and value % 5 == 0 for value in config.to_dict().values())
    )


def _metric(metrics: Mapping[str, Any], name: str, default: Any = 0) -> Any:
    value = metrics.get(name, default)
    return default if value is None else value


def score_metrics_detailed(
    metrics: Mapping[str, Any] | Any, config: PolicyConfig
) -> dict[str, Any]:
    """Return the optimized score with an additive, human-readable breakdown."""

    if not isinstance(metrics, Mapping):
        metrics = asdict(metrics)
    tokens = int(_metric(metrics, "estimated_input_tokens"))
    components = {
        "context": 0,
        "intent": 0,
        "reasoning_modifier": 0,
        "action_and_testing": 0,
        "scope": 0,
        "special_requirements": 0,
        "software_interaction": 0,
        "specialist_interaction": 0,
    }
    matched: list[str] = []

    def add(component: str, rule: str, points: int) -> None:
        components[component] += points
        if points:
            matched.append(f"{component}.{rule}:+{points}")

    if tokens >= 64_000:
        add("context", "tokens_ge_64000", 3 * config.context_unit)
    elif tokens >= 24_000:
        add("context", "tokens_24000_63999", 2 * config.context_unit)
    elif tokens >= 8_000:
        add("context", "tokens_8000_23999", config.context_unit)
    if int(_metric(metrics, "input_item_count")) > 30:
        add("context", "history_items_gt_30", config.history_weight)

    intent = str(_metric(metrics, "primary_intent", "extract_or_classify"))
    if intent in {"rewrite_or_format", "summarize"}:
        add("intent", f"intent_{intent}", config.light_intent_weight)
    elif intent == "compare":
        add("intent", "intent_compare", config.compare_intent_weight)
    elif intent == "analyze":
        add("intent", "intent_analyze", config.analytic_intent_weight)
    elif intent in {"research_or_synthesize", "plan_or_design"}:
        add("intent", f"intent_{intent}", config.deep_intent_weight)
    elif intent == "debug_prove_or_diagnose":
        add("intent", "intent_debug_prove_or_diagnose", config.debug_intent_weight)

    if any(
        bool(_metric(metrics, name))
        for name in (
            "has_tradeoffs_or_competing_objectives",
            "has_counterfactual_uncertainty_or_scenarios",
            "has_contradiction_or_ambiguity_to_resolve",
        )
    ):
        add(
            "reasoning_modifier",
            "tradeoff_counterfactual_or_ambiguity",
            config.reasoning_modifier_weight,
        )

    action = str(_metric(metrics, "expected_action", "no_tool"))
    if action in {"search_or_multiple_reads", "one_local_write"}:
        add("action_and_testing", f"action_{action}", config.simple_action_weight)
    elif action == "multiple_writes_or_chained_tools":
        add("action_and_testing", f"action_{action}", config.chained_action_weight)
    if bool(_metric(metrics, "requires_testing_or_verification")):
        add("action_and_testing", "testing_or_verification", config.testing_weight)

    multi_or_stateful = bool(_metric(metrics, "is_multi_file_modification")) or bool(
        _metric(metrics, "has_stateful_coordination")
    )
    cross_dependency = bool(_metric(metrics, "has_cross_file_or_system_dependency"))
    artifacts = int(_metric(metrics, "artifact_object_count", 1))
    deliverables = int(_metric(metrics, "deliverable_count", 1))
    if multi_or_stateful:
        add("scope", "multi_file_or_stateful", 4 * config.scope_unit)
    elif cross_dependency or artifacts >= 4 or deliverables >= 4:
        add("scope", "cross_dependency_or_many_objects", 2 * config.scope_unit)
    elif artifacts >= 2 or deliverables >= 2:
        add("scope", "multiple_objects_or_deliverables", config.scope_unit)

    special_count = sum(
        bool(_metric(metrics, name))
        for name in (
            "requires_strict_schema_or_machine_output",
            "requires_citations_or_source_traceability",
            "requires_compatibility_style_template_or_behavior_preservation",
            "requires_mutually_consistent_outputs",
            "has_exact_numerical_or_factual_acceptance_criteria",
            "requires_ocr_spatial_or_chart_reasoning",
            "requires_layout_sensitive_artifact",
        )
    )
    if special_count >= 2:
        add(
            "special_requirements",
            "two_or_more",
            2 * config.special_requirement_unit,
        )
    elif special_count == 1:
        add("special_requirements", "one", config.special_requirement_unit)

    software_bundle = (
        bool(_metric(metrics, "has_code_sql_formula_or_data_transformation"))
        and bool(_metric(metrics, "requires_testing_or_verification"))
        and (multi_or_stateful or cross_dependency)
    )
    if software_bundle:
        add(
            "software_interaction",
            "code_testing_and_cross_scope",
            config.software_engineering_interaction,
        )
    if any(
        bool(_metric(metrics, name))
        for name in (
            "has_architecture_migration_concurrency_or_integration",
            "has_formal_math_algorithms_security_or_specialist_domain",
            "has_cross_domain_reasoning",
        )
    ):
        add(
            "specialist_interaction",
            "architecture_formal_or_cross_domain",
            config.specialist_interaction,
        )

    raw_score = sum(components.values())
    requires_balanced_floor = (
        str(_metric(metrics, "action_risk", "read_only_or_none"))
        == "destructive_public_deployment_or_permission_change"
        or bool(_metric(metrics, "is_high_stakes_domain"))
        or bool(_metric(metrics, "involves_security_secrets_authentication_or_permissions"))
        or (
            bool(_metric(metrics, "is_multi_file_modification"))
            and bool(_metric(metrics, "requires_testing_or_verification"))
        )
    )
    floor_reasons: list[str] = []
    score = raw_score
    if requires_balanced_floor:
        score = max(score, config.economical_threshold)
        floor_reasons.append("optimized_balanced_safety_floor")
    score = min(100, max(0, int(score)))
    return {
        "score": score,
        "raw_score": raw_score,
        "components": components,
        "matched_rules": matched,
        "score_floor": config.economical_threshold if requires_balanced_floor else None,
        "score_floor_reasons": floor_reasons,
    }


def score_metrics(metrics: Mapping[str, Any] | Any, config: PolicyConfig) -> int:
    return int(score_metrics_detailed(metrics, config)["score"])


def tier_id_for_score(score: int, config: PolicyConfig) -> int:
    if score >= config.strongest_threshold:
        return 2
    if score >= config.economical_threshold:
        return 1
    return 0


def predict_tier_id(metrics: Mapping[str, Any] | Any, config: PolicyConfig) -> int:
    return tier_id_for_score(score_metrics(metrics, config), config)


def evaluate_config(rows: Sequence[Mapping[str, Any]], config: PolicyConfig) -> dict[str, float]:
    predictions = [predict_tier_id(row["metrics"], config) for row in rows]
    total = len(rows)
    exact = sum(prediction == int(row["gold_tier_id"]) for prediction, row in zip(predictions, rows))
    safe = sum(prediction >= int(row["gold_tier_id"]) for prediction, row in zip(predictions, rows))
    trajectories: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for prediction, row in zip(predictions, rows):
        trajectories[str(row["instance_id"])].append((prediction, int(row["gold_tier_id"])))
    passed_instances = {
        instance_id
        for instance_id, steps in trajectories.items()
        if all(prediction >= gold for prediction, gold in steps)
    }
    rows_in_passed = sum(
        1 for row in rows if str(row["instance_id"]) in passed_instances
    )

    # Fast proxy for Twin's failure-aware bill. These are its output-token tier
    # prices with the two middle tiers averaged into this router's balanced tier.
    tier_cost = (0.5, 3.5, 25.0)
    baseline_bill = 25.0 * total
    predicted_bill = 0.0
    for instance_id, steps in trajectories.items():
        trajectory_bill = sum(tier_cost[prediction] for prediction, _ in steps)
        if instance_id not in passed_instances:
            trajectory_bill += 25.0 * len(steps)
        predicted_bill += trajectory_bill
    savings = 100.0 * (baseline_bill - predicted_bill) / baseline_bill
    return {
        "exact_rate_percent": 100.0 * exact / total,
        "safe_step_rate_percent": 100.0 * safe / total,
        "row_weighted_trajectory_pass_rate_percent": 100.0 * rows_in_passed / total,
        "failure_aware_savings_proxy_percent": savings,
    }


def _config_distance(config: PolicyConfig, reference: PolicyConfig) -> float:
    return sum(
        abs(config.to_dict()[name] - reference.to_dict()[name]) / 5.0
        for name in SEARCH_SPACE
    )


def _objective(
    fold_metrics: Sequence[Mapping[str, float]],
    baseline_metrics: Sequence[Mapping[str, float]],
    config: PolicyConfig,
    *,
    regularization: float,
) -> float:
    values: list[float] = []
    for metrics, baseline in zip(fold_metrics, baseline_metrics):
        combined = statistics_mean(
            (
                metrics["exact_rate_percent"],
                metrics["safe_step_rate_percent"],
                metrics["row_weighted_trajectory_pass_rate_percent"],
                metrics["failure_aware_savings_proxy_percent"],
            )
        )
        safety_penalty = 3.0 * max(
            0.0,
            baseline["safe_step_rate_percent"] - metrics["safe_step_rate_percent"],
        )
        trajectory_penalty = 2.0 * max(
            0.0,
            baseline["row_weighted_trajectory_pass_rate_percent"]
            - metrics["row_weighted_trajectory_pass_rate_percent"],
        )
        values.append(combined - safety_penalty - trajectory_penalty)
    return statistics_mean(values) - regularization * _config_distance(config, DEFAULT_CONFIG)


def statistics_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def optimize_policy(
    rows: Sequence[Mapping[str, Any]],
    fold_ids: Sequence[int],
    *,
    max_passes: int = 5,
    regularization: float = 0.05,
) -> tuple[PolicyConfig, dict[str, Any]]:
    unique_folds = sorted(set(fold_ids))
    fold_rows = [
        [row for row, fold_id in zip(rows, fold_ids) if fold_id == fold]
        for fold in unique_folds
    ]
    baseline_metrics = [evaluate_config(subset, DEFAULT_CONFIG) for subset in fold_rows]
    cache: dict[PolicyConfig, float] = {}

    def score(config: PolicyConfig) -> float:
        if config not in cache:
            metrics = [evaluate_config(subset, config) for subset in fold_rows]
            cache[config] = _objective(
                metrics,
                baseline_metrics,
                config,
                regularization=regularization,
            )
        return cache[config]

    current = DEFAULT_CONFIG
    history: list[dict[str, Any]] = [
        {"pass": 0, "parameter": "initial", "value": None, "objective": score(current)}
    ]
    for pass_index in range(1, max_passes + 1):
        changed = False
        for parameter, candidates in SEARCH_SPACE.items():
            valid: list[PolicyConfig] = []
            for value in candidates:
                candidate = replace(current, **{parameter: value})
                if is_valid_config(candidate):
                    valid.append(candidate)
            best = max(
                valid,
                key=lambda candidate: (
                    score(candidate),
                    -_config_distance(candidate, DEFAULT_CONFIG),
                    -candidate.to_dict()[parameter],
                ),
            )
            if best != current:
                current = best
                changed = True
                history.append(
                    {
                        "pass": pass_index,
                        "parameter": parameter,
                        "value": current.to_dict()[parameter],
                        "objective": score(current),
                    }
                )
        if not changed:
            break
    return current, {
        "objective": score(current),
        "baseline_objective": score(DEFAULT_CONFIG),
        "evaluated_configurations": len(cache),
        "passes_completed": pass_index,
        "history": history,
        "regularization": regularization,
    }
