#!/usr/bin/env python3
"""Deterministic 0-100 pre-router described by ROUTER_COMPLEXITY_SPEC.md.

The module accepts explicit metrics rather than calling another LLM. It is both
importable and callable as a JSON CLI:

    python scripts/deterministic_router.py --example
    python scripts/deterministic_router.py request_metrics.json --pretty
    Get-Content request_metrics.json | python scripts/deterministic_router.py - --pretty

Importable API:

    from deterministic_router import evaluate_route
    decision = evaluate_route(payload)

Token counts and costs are estimates. The export contains no measured usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping


POLICY_VERSION = "complexity-v2-simple"
TIERS = ("economical", "balanced", "strongest")
TIER_RANK = {tier: index for index, tier in enumerate(TIERS)}

INTENT_POINTS = {
    "extract_or_classify": 0,
    "rewrite_or_format": 5,
    "summarize": 5,
    "compare": 10,
    "analyze": 15,
    "research_or_synthesize": 20,
    "plan_or_design": 20,
    "debug_prove_or_diagnose": 30,
}

ACTION_POINTS = {
    "no_tool": 0,
    "one_read_only": 0,
    "search_or_multiple_reads": 5,
    "one_local_write": 5,
    "multiple_writes_or_chained_tools": 10,
}

ACTION_RISK_POINTS = {
    "read_only_or_none": 0,
    "reversible_local_write": 1,
    "persistent_or_external": 3,
    "destructive_public_deployment_or_permission_change": 5,
}


@dataclass
class RouteMetrics:
    """All raw, structural, inferred, runtime, and cache metrics in the spec.

    The structural fields are retained even when they do not directly contribute
    points so callers can log and later recalibrate the policy without reparsing.
    Semantic boolean/enum values must be produced by deterministic versioned rules.
    """

    # Directly exported or structurally derived request/history metrics.
    request_model: str | None = None
    serialized_character_count: int = 0
    estimated_input_tokens: int | None = None
    input_item_count: int = 0
    system_message_count: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    text_part_count: int = 0
    reasoning_item_count: int = 0
    input_image_count: int = 0

    function_tool_call_count: int = 0
    custom_tool_call_count: int = 0
    tool_output_count: int = 0
    unique_tool_count: int = 0
    tool_call_sequence: list[str] = field(default_factory=list)
    tool_argument_fingerprints: list[str] = field(default_factory=list)
    tool_output_statuses: list[str] = field(default_factory=list)

    tool_definition_count: int = 0
    tool_schema_character_count: int = 0
    required_tool_argument_count: int = 0

    code_fence_count: int = 0
    url_count: int = 0
    file_path_count: int = 0
    placeholder_entity_count: int = 0
    heading_count: int = 0
    bullet_count: int = 0
    numbered_step_count: int = 0
    table_count: int = 0
    json_like_block_count: int = 0
    error_or_stacktrace_marker_count: int = 0
    explicit_requirement_term_count: int = 0
    matched_requirement_terms: list[str] = field(default_factory=list)

    distinct_referenced_artifact_count: int = 0
    distinct_input_source_count: int = 0

    # Deterministically inferred complexity metrics.
    primary_intent: str = "extract_or_classify"
    dependent_step_count: int = 0
    has_cross_reference_or_synthesis: bool = False
    has_tradeoffs_or_competing_objectives: bool = False
    has_counterfactual_uncertainty_or_scenarios: bool = False
    has_contradiction_or_ambiguity_to_resolve: bool = False

    artifact_object_count: int = 1
    deliverable_count: int = 1
    has_cross_file_or_system_dependency: bool = False
    has_ordered_dependent_workflow: bool = False
    has_stateful_coordination: bool = False
    is_multi_file_modification: bool = False

    expected_action: str = "no_tool"
    expected_tool_stage_count: int = 0
    requires_testing_or_verification: bool = False
    requires_tool_chaining: bool = False

    has_code_sql_formula_or_data_transformation: bool = False
    has_debug_error_failing_test_or_stacktrace: bool = False
    has_architecture_migration_concurrency_or_integration: bool = False
    has_formal_math_algorithms_security_or_specialist_domain: bool = False
    has_cross_domain_reasoning: bool = False

    hard_constraint_count: int = 0
    requires_strict_schema_or_machine_output: bool = False
    requires_citations_or_source_traceability: bool = False
    requires_compatibility_style_template_or_behavior_preservation: bool = False
    requires_mutually_consistent_outputs: bool = False
    has_exact_numerical_or_factual_acceptance_criteria: bool = False

    action_risk: str = "read_only_or_none"
    is_high_stakes_domain: bool = False
    involves_security_secrets_authentication_or_permissions: bool = False
    has_irreversibility_or_broad_blast_radius: bool = False

    requires_ocr_spatial_or_chart_reasoning: bool = False
    requires_layout_sensitive_artifact: bool = False

    # Runtime failure metrics. These never contribute to initial complexity.
    malformed_tool_or_structured_response_count: int = 0
    repeated_failed_tool_and_arguments_count: int = 0
    explicit_user_correction_count: int = 0
    consecutive_model_attributable_failure_count: int = 0
    failed_verification_after_claimed_completion_count: int = 0
    no_progress_two_call_window_count: int = 0
    environmental_error_count: int = 0

    # Cache/cost telemetry.
    shared_prefix_tokens: int = 0
    estimated_uncached_tokens_after_switch: int = 0
    switch_count: int = 0


@dataclass
class CapabilityRequirements:
    """Requirements can be supplied explicitly; None means derive from metrics."""

    min_context_window_tokens: int | None = None
    requires_images: bool | None = None
    requires_tools: bool | None = None
    requires_structured_output: bool | None = None
    requires_layout_tasks: bool | None = None
    required_language: str | None = None
    required_data_tags: list[str] = field(default_factory=list)


@dataclass
class ModelSpec:
    """Configurable model ladder and capabilities; never infer tier from its ID."""

    id: str
    tier: str
    context_window_tokens: int
    supports_images: bool = False
    supports_tools: bool = True
    supports_structured_output: bool = True
    supports_layout_tasks: bool = True
    supported_languages: list[str] = field(default_factory=lambda: ["*"])
    allowed_data_tags: list[str] = field(default_factory=lambda: ["*"])
    uncached_input_usd_per_million: float | None = None
    cached_input_usd_per_million: float | None = None


@dataclass
class RouterInput:
    metrics: RouteMetrics
    mode: str = "initial"
    current_model_id: str | None = None
    current_tier: str | None = None
    requirements: CapabilityRequirements = field(default_factory=CapabilityRequirements)
    models: list[ModelSpec] = field(default_factory=list)
    tier_model_map: dict[str, str] = field(default_factory=dict)
    policy_version: str = POLICY_VERSION


@dataclass
class PolicyScore:
    """Policy-specific score supplied to the shared routing engine.

    The v2 router constructs this internally. Newer policies can supply one to
    reuse validation, capability filtering, model selection, cost estimates,
    and sticky mid-trajectory behavior without changing the frozen v2 policy.
    """

    score: int
    component_scores: dict[str, int]
    component_raw_scores: dict[str, int]
    base_target_tier: str
    matched_rules: list[str] = field(default_factory=list)
    score_floor: int | None = None
    score_floor_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    policy_version: str
    mode: str
    decision: str
    complexity_score: int
    component_scores: dict[str, int]
    component_raw_scores: dict[str, int]
    base_target_tier: str
    target_tier: str
    current_tier: str | None
    current_model_id: str | None
    selected_model_id: str | None
    score_floor: int | None
    score_floor_reasons: list[str]
    failure_score: int
    failure_reasons: list[str]
    matched_rules: list[str]
    capability_requirements: dict[str, Any]
    eligible_models: list[str]
    capability_filters: dict[str, list[str]]
    estimated_input_tokens: int
    estimated_cache_reset_tokens: int
    estimated_uncached_tokens_after_switch: int
    cost_estimates_by_model: dict[str, dict[str, float | None]]
    selected_input_cost_usd: float | None
    estimated_cost_change_vs_keep_usd: float | None
    warnings: list[str]
    telemetry_summary: dict[str, Any]
    policy_metadata: dict[str, Any]


def _construct(cls: type[Any], value: Mapping[str, Any] | None) -> Any:
    data = dict(value or {})
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {', '.join(unknown)}")
    return cls(**data)


def parse_router_input(payload: Mapping[str, Any]) -> RouterInput:
    allowed = {item.name for item in fields(RouterInput)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown RouterInput fields: {', '.join(unknown)}")

    metrics = _construct(RouteMetrics, payload.get("metrics"))
    requirements = _construct(CapabilityRequirements, payload.get("requirements"))
    models = [_construct(ModelSpec, model) for model in payload.get("models", [])]
    result = RouterInput(
        metrics=metrics,
        mode=payload.get("mode", "initial"),
        current_model_id=payload.get("current_model_id"),
        current_tier=payload.get("current_tier"),
        requirements=requirements,
        models=models,
        tier_model_map=dict(payload.get("tier_model_map", {})),
        policy_version=payload.get("policy_version", POLICY_VERSION),
    )
    validate_router_input(result)
    return result


def validate_router_input(router_input: RouterInput) -> None:
    if router_input.mode not in {"initial", "mid_trajectory"}:
        raise ValueError("mode must be 'initial' or 'mid_trajectory'")
    if router_input.current_tier is not None and router_input.current_tier not in TIER_RANK:
        raise ValueError(f"current_tier must be one of {TIERS}")
    if router_input.mode == "mid_trajectory" and not (
        router_input.current_tier or router_input.current_model_id
    ):
        raise ValueError("mid_trajectory mode requires current_tier or current_model_id")

    metrics = router_input.metrics
    if metrics.primary_intent not in INTENT_POINTS:
        raise ValueError(f"primary_intent must be one of {tuple(INTENT_POINTS)}")
    if metrics.expected_action not in ACTION_POINTS:
        raise ValueError(f"expected_action must be one of {tuple(ACTION_POINTS)}")
    if metrics.action_risk not in ACTION_RISK_POINTS:
        raise ValueError(f"action_risk must be one of {tuple(ACTION_RISK_POINTS)}")

    for item in fields(RouteMetrics):
        value = getattr(metrics, item.name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            raise ValueError(f"metrics.{item.name} cannot be negative")

    if router_input.requirements.min_context_window_tokens is not None:
        if router_input.requirements.min_context_window_tokens < 0:
            raise ValueError("requirements.min_context_window_tokens cannot be negative")

    seen_model_ids: set[str] = set()
    for model in router_input.models:
        if model.id in seen_model_ids:
            raise ValueError(f"duplicate model id: {model.id}")
        seen_model_ids.add(model.id)
        if model.tier not in TIER_RANK:
            raise ValueError(f"model {model.id!r} has invalid tier {model.tier!r}")
        if model.context_window_tokens < 0:
            raise ValueError(f"model {model.id!r} has a negative context window")
        for price in (
            model.uncached_input_usd_per_million,
            model.cached_input_usd_per_million,
        ):
            if price is not None and price < 0:
                raise ValueError(f"model {model.id!r} has a negative price")

    for tier in router_input.tier_model_map:
        if tier not in TIER_RANK:
            raise ValueError(f"tier_model_map contains invalid tier {tier!r}")


def estimated_tokens(metrics: RouteMetrics) -> int:
    if metrics.estimated_input_tokens is not None:
        return metrics.estimated_input_tokens
    return metrics.serialized_character_count // 4


def _add(
    category: str,
    rule: str,
    points: int,
    raw_scores: dict[str, int],
    matched_rules: list[str],
) -> None:
    raw_scores[category] += points
    if points:
        matched_rules.append(f"{category}.{rule}:+{points}")


def calculate_complexity(metrics: RouteMetrics) -> tuple[
    int, dict[str, int], dict[str, int], list[str], int | None, list[str]
]:
    """Calculate the five-component v2 score.

    Each underlying fact is used once.  Related booleans are collapsed into a
    single band, avoiding the v1 behaviour where one phrase could accumulate
    points in several overlapping categories.
    """

    raw = {
        "context": 0,
        "reasoning": 0,
        "execution": 0,
        "scope": 0,
        "special_requirements": 0,
    }
    matched: list[str] = []
    tokens = estimated_tokens(metrics)

    # 1. Context load (0/5/10/15) plus one history increment (+5).
    if tokens >= 64_000:
        _add("context", "tokens_ge_64000", 15, raw, matched)
    elif tokens >= 24_000:
        _add("context", "tokens_24000_63999", 10, raw, matched)
    elif tokens >= 8_000:
        _add("context", "tokens_8000_23999", 5, raw, matched)
    if metrics.input_item_count > 30:
        _add("context", "history_items_gt_30", 5, raw, matched)

    # 2. Reasoning depth: one intent band and at most one +5 modifier.
    _add(
        "reasoning",
        f"intent_{metrics.primary_intent}",
        INTENT_POINTS[metrics.primary_intent],
        raw,
        matched,
    )
    reasoning_modifiers = (
        metrics.has_tradeoffs_or_competing_objectives,
        metrics.has_counterfactual_uncertainty_or_scenarios,
        metrics.has_contradiction_or_ambiguity_to_resolve,
    )
    if any(reasoning_modifiers):
        _add("reasoning", "complex_reasoning_modifier", 5, raw, matched)

    # 3. Execution: one action band and one verification increment.
    _add(
        "execution",
        f"action_{metrics.expected_action}",
        ACTION_POINTS[metrics.expected_action],
        raw,
        matched,
    )
    if metrics.requires_testing_or_verification:
        _add("execution", "testing_or_verification", 5, raw, matched)

    # 4. Scope: exactly one mutually exclusive band.
    if metrics.is_multi_file_modification or metrics.has_stateful_coordination:
        _add("scope", "multi_file_or_stateful", 20, raw, matched)
    elif (
        metrics.has_cross_file_or_system_dependency
        or metrics.artifact_object_count >= 4
        or metrics.deliverable_count >= 4
    ):
        _add("scope", "cross_dependency_or_many_objects", 10, raw, matched)
    elif metrics.artifact_object_count >= 2 or metrics.deliverable_count >= 2:
        _add("scope", "multiple_objects_or_deliverables", 5, raw, matched)

    # 5. Special requirements: count independent requirements, then band once.
    special_count = sum(
        (
            metrics.requires_strict_schema_or_machine_output,
            metrics.requires_citations_or_source_traceability,
            metrics.requires_compatibility_style_template_or_behavior_preservation,
            metrics.requires_mutually_consistent_outputs,
            metrics.has_exact_numerical_or_factual_acceptance_criteria,
            metrics.requires_ocr_spatial_or_chart_reasoning,
            metrics.requires_layout_sensitive_artifact,
        )
    )
    if special_count >= 2:
        _add("special_requirements", "two_or_more", 10, raw, matched)
    elif special_count == 1:
        _add("special_requirements", "one", 5, raw, matched)

    caps = {
        "context": 20,
        "reasoning": 35,
        "execution": 15,
        "scope": 20,
        "special_requirements": 10,
    }
    scores = {category: min(value, caps[category]) for category, value in raw.items()}
    score = sum(scores.values())

    floor_reasons: list[str] = []
    score_floor: int | None = None
    destructive = (
        metrics.action_risk == "destructive_public_deployment_or_permission_change"
    )
    if destructive or metrics.is_high_stakes_domain:
        score_floor = 35
        floor_reasons.append("destructive_public_deployment_permission_or_high_stakes")
    if metrics.involves_security_secrets_authentication_or_permissions:
        score_floor = max(score_floor or 0, 35)
        floor_reasons.append("security_secrets_authentication_or_permissions")
    if metrics.is_multi_file_modification and metrics.requires_testing_or_verification:
        score_floor = max(score_floor or 0, 35)
        floor_reasons.append("multi_file_modification_with_verification")
    if score_floor is not None:
        score = max(score, score_floor)

    return score, scores, raw, matched, score_floor, floor_reasons


def calculate_failure_score(metrics: RouteMetrics) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    def present(name: str, count: int, points: int) -> None:
        nonlocal score
        if count > 0:
            score += points
            reasons.append(f"{name}:+{points}")

    present("malformed_tool_or_structured_response", metrics.malformed_tool_or_structured_response_count, 8)
    present("repeated_failed_tool_and_arguments", metrics.repeated_failed_tool_and_arguments_count, 8)
    present("explicit_user_correction", metrics.explicit_user_correction_count, 10)
    if metrics.consecutive_model_attributable_failure_count >= 2:
        score += 8
        reasons.append("two_consecutive_model_attributable_failures:+8")
    present(
        "failed_verification_after_claimed_completion",
        metrics.failed_verification_after_claimed_completion_count,
        6,
    )
    present("no_progress_across_two_calls", metrics.no_progress_two_call_window_count, 6)
    return score, reasons


def tier_for_score(score: int) -> str:
    if score >= 75:
        return "strongest"
    if score >= 35:
        return "balanced"
    return "economical"


def derive_requirements(
    metrics: RouteMetrics, supplied: CapabilityRequirements
) -> CapabilityRequirements:
    tokens = estimated_tokens(metrics)
    return CapabilityRequirements(
        min_context_window_tokens=(
            supplied.min_context_window_tokens
            if supplied.min_context_window_tokens is not None
            else tokens
        ),
        requires_images=(
            supplied.requires_images
            if supplied.requires_images is not None
            else metrics.input_image_count > 0
        ),
        requires_tools=(
            supplied.requires_tools
            if supplied.requires_tools is not None
            else metrics.expected_action != "no_tool"
        ),
        requires_structured_output=(
            supplied.requires_structured_output
            if supplied.requires_structured_output is not None
            else metrics.requires_strict_schema_or_machine_output
        ),
        requires_layout_tasks=(
            supplied.requires_layout_tasks
            if supplied.requires_layout_tasks is not None
            else metrics.requires_layout_sensitive_artifact
        ),
        required_language=supplied.required_language,
        required_data_tags=list(supplied.required_data_tags),
    )


def capability_failures(model: ModelSpec, req: CapabilityRequirements) -> list[str]:
    failures: list[str] = []
    if req.min_context_window_tokens is not None:
        if model.context_window_tokens < req.min_context_window_tokens:
            failures.append("context_window")
    if req.requires_images and not model.supports_images:
        failures.append("images")
    if req.requires_tools and not model.supports_tools:
        failures.append("tools")
    if req.requires_structured_output and not model.supports_structured_output:
        failures.append("structured_output")
    if req.requires_layout_tasks and not model.supports_layout_tasks:
        failures.append("layout_tasks")
    if req.required_language:
        supported = set(model.supported_languages)
        if "*" not in supported and req.required_language not in supported:
            failures.append(f"language:{req.required_language}")
    if req.required_data_tags:
        allowed = set(model.allowed_data_tags)
        if "*" not in allowed:
            missing = sorted(set(req.required_data_tags) - allowed)
            failures.extend(f"data_tag:{tag}" for tag in missing)
    return failures


def _model_by_id(models: list[ModelSpec], model_id: str | None) -> ModelSpec | None:
    if model_id is None:
        return None
    return next((model for model in models if model.id == model_id), None)


def _select_model(
    models: list[ModelSpec], eligible_ids: set[str], target_tier: str
) -> ModelSpec | None:
    target_rank = TIER_RANK[target_tier]
    candidates = [
        model
        for model in models
        if model.id in eligible_ids and TIER_RANK[model.tier] >= target_rank
    ]
    if not candidates:
        return None

    def sort_key(model: ModelSpec) -> tuple[int, float, str]:
        price = (
            model.uncached_input_usd_per_million
            if model.uncached_input_usd_per_million is not None
            else float("inf")
        )
        return TIER_RANK[model.tier], price, model.id

    return min(candidates, key=sort_key)


def calculate_costs(
    router_input: RouterInput,
    selected_model_id: str | None,
    tokens: int,
) -> tuple[
    dict[str, dict[str, float | None]], float | None, float | None, int, int
]:
    metrics = router_input.metrics
    shared = min(metrics.shared_prefix_tokens, tokens)
    uncached_after_switch = metrics.estimated_uncached_tokens_after_switch or tokens
    estimates: dict[str, dict[str, float | None]] = {}

    def priced(model: ModelSpec, uncached: int, cached: int) -> float | None:
        if model.uncached_input_usd_per_million is None:
            return None
        cached_price = model.cached_input_usd_per_million
        if cached and cached_price is None:
            return None
        return round(
            (
                uncached * model.uncached_input_usd_per_million
                + cached * (cached_price or 0.0)
            )
            / 1_000_000,
            9,
        )

    for model in router_input.models:
        fresh = priced(model, tokens, 0)
        hypothetical_continue = priced(model, tokens - shared, shared)
        if router_input.mode == "mid_trajectory" and model.id == router_input.current_model_id:
            selected_scenario = hypothetical_continue
        else:
            selected_scenario = fresh
        estimates[model.id] = {
            "fresh_or_switch_input_cost_usd": fresh,
            "hypothetical_same_model_continuation_cost_usd": hypothetical_continue,
            "cost_if_selected_now_usd": selected_scenario,
        }

    selected_cost = (
        estimates.get(selected_model_id, {}).get("cost_if_selected_now_usd")
        if selected_model_id
        else None
    )
    keep_cost = (
        estimates.get(router_input.current_model_id, {}).get(
            "hypothetical_same_model_continuation_cost_usd"
        )
        if router_input.current_model_id
        else None
    )
    cost_change = None
    if selected_cost is not None and keep_cost is not None:
        cost_change = round(selected_cost - keep_cost, 9)
    return estimates, selected_cost, cost_change, shared, uncached_after_switch


def evaluate_router_input(
    router_input: RouterInput, policy_score: PolicyScore | None = None
) -> RoutingDecision:
    metrics = router_input.metrics
    warnings = [
        "Input tokens use an estimate unless measured usage is supplied externally.",
        "Input cost excludes output tokens and depends on configured assumed prices.",
    ]
    if metrics.estimated_input_tokens is None:
        warnings.append("estimated_input_tokens was derived as serialized characters / 4.")
    elif metrics.serialized_character_count:
        chars_estimate = metrics.serialized_character_count // 4
        if chars_estimate and abs(metrics.estimated_input_tokens - chars_estimate) / chars_estimate > 0.25:
            warnings.append("supplied token estimate differs from characters / 4 by more than 25%.")

    if policy_score is None:
        score, components, raw_components, matched, score_floor, floor_reasons = (
            calculate_complexity(metrics)
        )
        base_target_tier = tier_for_score(score)
        policy_metadata: dict[str, Any] = {}
    else:
        score = policy_score.score
        components = policy_score.component_scores
        raw_components = policy_score.component_raw_scores
        matched = policy_score.matched_rules
        score_floor = policy_score.score_floor
        floor_reasons = policy_score.score_floor_reasons
        base_target_tier = policy_score.base_target_tier
        policy_metadata = policy_score.metadata
    failure_score, failure_reasons = calculate_failure_score(metrics)
    requirements = derive_requirements(metrics, router_input.requirements)

    model_by_id = {model.id: model for model in router_input.models}
    capability_filters = {
        model.id: capability_failures(model, requirements) for model in router_input.models
    }
    eligible_ids = {
        model_id for model_id, failures in capability_filters.items() if not failures
    }
    eligible_models = sorted(eligible_ids)

    current_model = _model_by_id(router_input.models, router_input.current_model_id)
    current_tier = router_input.current_tier
    if current_tier is None and current_model is not None:
        current_tier = current_model.tier

    if router_input.models and router_input.current_model_id and current_model is None:
        warnings.append("current_model_id is not present in models; its capabilities are unknown.")
    if not router_input.models:
        warnings.append("No model specifications supplied; capability filtering and priced selection are unavailable.")

    target_tier = base_target_tier
    decision = "ROUTE"
    current_capability_failures = (
        capability_filters.get(router_input.current_model_id, [])
        if router_input.current_model_id
        else []
    )

    if router_input.mode == "mid_trajectory":
        if current_tier is None:
            raise ValueError("could not derive current_tier for mid_trajectory mode")
        if current_capability_failures:
            decision = "CAPABILITY_CHANGE_REQUIRED"
            target_tier = TIERS[max(TIER_RANK[current_tier], TIER_RANK[base_target_tier])]
        elif failure_score >= 16:
            if current_tier == "strongest":
                decision = "REVIEW_REQUIRED"
                target_tier = current_tier
            else:
                decision = "UPGRADE_RECOMMENDED"
                next_tier = TIERS[TIER_RANK[current_tier] + 1]
                target_tier = TIERS[max(TIER_RANK[next_tier], TIER_RANK[base_target_tier])]
        else:
            decision = "KEEP"
            target_tier = current_tier
    elif current_tier is not None:
        if current_capability_failures:
            decision = "CAPABILITY_CHANGE_REQUIRED"
        elif TIER_RANK[target_tier] > TIER_RANK[current_tier]:
            decision = "UPGRADE_RECOMMENDED"
        elif TIER_RANK[target_tier] < TIER_RANK[current_tier]:
            decision = "DOWNGRADE_RECOMMENDED"
        else:
            decision = "KEEP"

    selected_model: ModelSpec | None = None
    selected_model_id: str | None = None
    if decision in {"KEEP", "REVIEW_REQUIRED"} and router_input.current_model_id:
        selected_model_id = router_input.current_model_id
    elif router_input.models:
        selected_model = _select_model(router_input.models, eligible_ids, target_tier)
        selected_model_id = selected_model.id if selected_model else None
        if selected_model is None:
            warnings.append(f"No eligible model is available at or above tier {target_tier!r}.")
            if decision != "REVIEW_REQUIRED":
                decision = "NO_ELIGIBLE_MODEL"
    else:
        selected_model_id = router_input.tier_model_map.get(target_tier)
        if selected_model_id is None:
            warnings.append(f"tier_model_map has no model for tier {target_tier!r}.")

    tokens = estimated_tokens(metrics)
    costs, selected_cost, cost_change, cache_reset_tokens, uncached_after_switch = (
        calculate_costs(router_input, selected_model_id, tokens)
    )

    telemetry = {
        "input_item_count": metrics.input_item_count,
        "message_counts": {
            "system": metrics.system_message_count,
            "user": metrics.user_message_count,
            "assistant": metrics.assistant_message_count,
        },
        "tool_calls": metrics.function_tool_call_count + metrics.custom_tool_call_count,
        "tool_outputs": metrics.tool_output_count,
        "unique_tools": metrics.unique_tool_count,
        "environmental_errors_not_scored": metrics.environmental_error_count,
        "switch_count": metrics.switch_count,
    }

    return RoutingDecision(
        policy_version=router_input.policy_version,
        mode=router_input.mode,
        decision=decision,
        complexity_score=score,
        component_scores=components,
        component_raw_scores=raw_components,
        base_target_tier=base_target_tier,
        target_tier=target_tier,
        current_tier=current_tier,
        current_model_id=router_input.current_model_id,
        selected_model_id=selected_model_id,
        score_floor=score_floor,
        score_floor_reasons=floor_reasons,
        failure_score=failure_score,
        failure_reasons=failure_reasons,
        matched_rules=matched,
        capability_requirements=asdict(requirements),
        eligible_models=eligible_models,
        capability_filters=capability_filters,
        estimated_input_tokens=tokens,
        estimated_cache_reset_tokens=cache_reset_tokens,
        estimated_uncached_tokens_after_switch=uncached_after_switch,
        cost_estimates_by_model=costs,
        selected_input_cost_usd=selected_cost,
        estimated_cost_change_vs_keep_usd=cost_change,
        warnings=warnings,
        telemetry_summary=telemetry,
        policy_metadata=policy_metadata,
    )


def evaluate_route(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a JSON-compatible payload and return a JSON-compatible decision."""

    return asdict(evaluate_router_input(parse_router_input(payload)))


def example_payload() -> dict[str, Any]:
    """Return a complete example containing every accepted metric field."""

    metrics = asdict(RouteMetrics())
    metrics.update(
        {
            "serialized_character_count": 48_000,
            "estimated_input_tokens": 12_000,
            "input_item_count": 14,
            "system_message_count": 1,
            "user_message_count": 2,
            "assistant_message_count": 1,
            "text_part_count": 4,
            "reasoning_item_count": 0,
            "function_tool_call_count": 2,
            "tool_output_count": 2,
            "unique_tool_count": 2,
            "tool_call_sequence": ["read_file", "apply_patch"],
            "tool_argument_fingerprints": ["sha256:example-a", "sha256:example-b"],
            "tool_output_statuses": ["success", "success"],
            "tool_definition_count": 8,
            "tool_schema_character_count": 10_000,
            "required_tool_argument_count": 12,
            "code_fence_count": 1,
            "file_path_count": 4,
            "heading_count": 2,
            "bullet_count": 5,
            "numbered_step_count": 3,
            "error_or_stacktrace_marker_count": 1,
            "explicit_requirement_term_count": 3,
            "matched_requirement_terms": ["must", "preserve", "exactly"],
            "distinct_referenced_artifact_count": 4,
            "distinct_input_source_count": 2,
            "primary_intent": "debug_prove_or_diagnose",
            "dependent_step_count": 4,
            "has_cross_reference_or_synthesis": True,
            "has_tradeoffs_or_competing_objectives": True,
            "artifact_object_count": 4,
            "deliverable_count": 2,
            "has_cross_file_or_system_dependency": True,
            "has_ordered_dependent_workflow": True,
            "is_multi_file_modification": True,
            "expected_action": "multiple_writes_or_chained_tools",
            "expected_tool_stage_count": 4,
            "requires_testing_or_verification": True,
            "requires_tool_chaining": True,
            "has_code_sql_formula_or_data_transformation": True,
            "has_debug_error_failing_test_or_stacktrace": True,
            "has_architecture_migration_concurrency_or_integration": True,
            "hard_constraint_count": 3,
            "requires_compatibility_style_template_or_behavior_preservation": True,
            "has_exact_numerical_or_factual_acceptance_criteria": True,
            "action_risk": "reversible_local_write",
            "shared_prefix_tokens": 10_500,
            "estimated_uncached_tokens_after_switch": 12_000,
        }
    )
    return {
        "policy_version": POLICY_VERSION,
        "mode": "initial",
        "current_model_id": None,
        "current_tier": None,
        "metrics": metrics,
        "requirements": asdict(CapabilityRequirements(required_language="en")),
        "models": [
            asdict(
                ModelSpec(
                    id="model-economical",
                    tier="economical",
                    context_window_tokens=32_000,
                    supports_images=False,
                    uncached_input_usd_per_million=0.8,
                    cached_input_usd_per_million=0.08,
                )
            ),
            asdict(
                ModelSpec(
                    id="model-balanced",
                    tier="balanced",
                    context_window_tokens=128_000,
                    supports_images=True,
                    uncached_input_usd_per_million=3.0,
                    cached_input_usd_per_million=0.3,
                )
            ),
            asdict(
                ModelSpec(
                    id="model-strongest",
                    tier="strongest",
                    context_window_tokens=128_000,
                    supports_images=True,
                    uncached_input_usd_per_million=15.0,
                    cached_input_usd_per_million=1.5,
                )
            ),
        ],
        "tier_model_map": {},
    }


def run_self_tests() -> dict[str, Any]:
    simple = evaluate_route({"metrics": {}})
    assert simple["complexity_score"] == 0
    assert simple["target_tier"] == "economical"

    maximal_metrics = asdict(RouteMetrics())
    maximal_metrics.update(
        {
            "estimated_input_tokens": 64_000,
            "input_item_count": 31,
            "distinct_referenced_artifact_count": 11,
            "has_cross_reference_or_synthesis": True,
            "distinct_input_source_count": 2,
            "primary_intent": "debug_prove_or_diagnose",
            "dependent_step_count": 7,
            "has_tradeoffs_or_competing_objectives": True,
            "has_counterfactual_uncertainty_or_scenarios": True,
            "has_contradiction_or_ambiguity_to_resolve": True,
            "artifact_object_count": 11,
            "deliverable_count": 4,
            "has_cross_file_or_system_dependency": True,
            "has_ordered_dependent_workflow": True,
            "has_stateful_coordination": True,
            "expected_action": "multiple_writes_or_chained_tools",
            "expected_tool_stage_count": 4,
            "requires_testing_or_verification": True,
            "requires_tool_chaining": True,
            "has_code_sql_formula_or_data_transformation": True,
            "has_debug_error_failing_test_or_stacktrace": True,
            "has_architecture_migration_concurrency_or_integration": True,
            "has_formal_math_algorithms_security_or_specialist_domain": True,
            "has_cross_domain_reasoning": True,
            "hard_constraint_count": 4,
            "requires_strict_schema_or_machine_output": True,
            "requires_citations_or_source_traceability": True,
            "requires_compatibility_style_template_or_behavior_preservation": True,
            "requires_mutually_consistent_outputs": True,
            "has_exact_numerical_or_factual_acceptance_criteria": True,
            "action_risk": "destructive_public_deployment_or_permission_change",
            "is_high_stakes_domain": True,
            "involves_security_secrets_authentication_or_permissions": True,
            "has_irreversibility_or_broad_blast_radius": True,
            "input_image_count": 2,
            "requires_ocr_spatial_or_chart_reasoning": True,
            "requires_layout_sensitive_artifact": True,
        }
    )
    maximal = evaluate_route({"metrics": maximal_metrics})
    assert maximal["complexity_score"] == 100
    assert maximal["component_scores"] == {
        "context": 20,
        "reasoning": 35,
        "execution": 15,
        "scope": 20,
        "special_requirements": 10,
    }

    high_risk = evaluate_route(
        {"metrics": {"is_high_stakes_domain": True, "primary_intent": "summarize"}}
    )
    assert high_risk["complexity_score"] == 35
    assert high_risk["score_floor"] == 35

    mid = evaluate_route(
        {
            "mode": "mid_trajectory",
            "current_tier": "economical",
            "metrics": {
                "malformed_tool_or_structured_response_count": 1,
                "explicit_user_correction_count": 1,
                "shared_prefix_tokens": 500,
                "estimated_input_tokens": 1_000,
            },
            "tier_model_map": {"balanced": "balanced-example"},
        }
    )
    assert mid["failure_score"] == 18
    assert mid["decision"] == "UPGRADE_RECOMMENDED"
    assert mid["target_tier"] == "balanced"

    capability = evaluate_route(
        {
            "metrics": {"input_image_count": 1},
            "models": [
                {
                    "id": "text-only",
                    "tier": "economical",
                    "context_window_tokens": 8_000,
                    "supports_images": False,
                },
                {
                    "id": "vision-balanced",
                    "tier": "balanced",
                    "context_window_tokens": 8_000,
                    "supports_images": True,
                },
            ],
        }
    )
    assert capability["capability_filters"]["text-only"] == ["images"]
    assert capability["selected_model_id"] == "vision-balanced"

    return {
        "status": "ok",
        "checks": [
            "zero-score economical route",
            "all category caps sum to 100",
            "high-stakes score floor",
            "mid-trajectory failure escalation",
            "capability filtering and fallback",
        ],
    }


def _read_payload(path: str) -> Mapping[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="JSON input path, or '-' for stdin")
    parser.add_argument("--example", action="store_true", help="print a complete example input")
    parser.add_argument("--self-test", action="store_true", help="run built-in policy checks")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    args = parser.parse_args()

    try:
        if args.example:
            output: Any = example_payload()
        elif args.self_test:
            output = run_self_tests()
        else:
            if not args.input:
                parser.error("provide a JSON input path, '-' for stdin, --example, or --self-test")
            output = evaluate_route(_read_payload(args.input))
        print(json.dumps(output, indent=2 if args.pretty or args.example else None, sort_keys=True))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
