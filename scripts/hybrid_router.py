#!/usr/bin/env python3
"""Versioned optimized router with conservative, explainable escalation guards.

The frozen ``deterministic_router.py`` remains the v2 baseline. This module
supplies a v3 policy score to the same validation, capability, cost, and sticky
trajectory engine. The learned model is only a one-way uncertainty signal: it
can trigger escalation but can never downgrade the deterministic decision.

Examples:
    python scripts/hybrid_router.py --example
    python scripts/hybrid_router.py request_metrics.json --pretty
    Get-Content request_metrics.json | python scripts/hybrid_router.py - --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from deterministic_router import (
    PolicyScore,
    RouteMetrics,
    TIERS,
    TIER_RANK,
    evaluate_router_input,
    example_payload,
    parse_router_input,
)
from learned_router.deterministic_optimizer import (
    PolicyConfig,
    score_metrics_detailed,
)
from learned_router.features import reduced_features
from learned_router.models import OrdinalLogisticRouter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = Path(__file__).with_name("router_policies") / "hybrid_v3.json"


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        policy = json.load(handle)
    required = {"policy_version", "optimized_config", "guards", "learned_second_opinion"}
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"hybrid policy is missing: {', '.join(missing)}")
    config = PolicyConfig.from_dict(policy["optimized_config"])
    if not 0 <= int(policy["guards"]["threshold_margin"]) < config.economical_threshold:
        raise ValueError("threshold_margin must be non-negative and below the economical threshold")
    return policy


def _tier_for_score(score: int, config: PolicyConfig) -> str:
    if score >= config.strongest_threshold:
        return "strongest"
    if score >= config.economical_threshold:
        return "balanced"
    return "economical"


def _promote(tier: str) -> str:
    return TIERS[min(len(TIERS) - 1, TIER_RANK[tier] + 1)]


def _hard_floor(metrics: RouteMetrics, guards: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    strongest_risk = (
        metrics.action_risk == "destructive_public_deployment_or_permission_change"
        or (
            metrics.has_irreversibility_or_broad_blast_radius
            and (
                metrics.is_high_stakes_domain
                or metrics.involves_security_secrets_authentication_or_permissions
            )
        )
    )
    balanced_risk = (
        metrics.is_high_stakes_domain
        or metrics.involves_security_secrets_authentication_or_permissions
        or metrics.has_irreversibility_or_broad_blast_radius
    )
    cross_scope = (
        metrics.is_multi_file_modification
        or metrics.has_stateful_coordination
        or metrics.has_cross_file_or_system_dependency
    )
    complex_software = metrics.has_code_sql_formula_or_data_transformation and (
        metrics.has_debug_error_failing_test_or_stacktrace
        or metrics.requires_testing_or_verification
        or metrics.has_architecture_migration_concurrency_or_integration
        or cross_scope
    )
    long_cross_scope_work = (
        metrics.dependent_step_count >= int(guards["strong_floor_dependent_steps"])
        and metrics.is_multi_file_modification
        and metrics.requires_testing_or_verification
        and metrics.requires_tool_chaining
        and (
            metrics.has_debug_error_failing_test_or_stacktrace
            or metrics.has_architecture_migration_concurrency_or_integration
        )
    )
    if strongest_risk:
        reasons.append("compound_high_risk_or_destructive_work")
    if long_cross_scope_work:
        reasons.append("long_dependent_cross_scope_work")
    if reasons:
        return "strongest", reasons
    if balanced_risk:
        return "balanced", ["high_stakes_security_or_irreversible_work"]
    if complex_software:
        return "balanced", ["complex_software_debugging_or_testing"]
    return None, []


def _ood_reasons(metrics: RouteMetrics, guards: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if metrics.has_formal_math_algorithms_security_or_specialist_domain:
        reasons.append("formal_security_or_specialist_domain")
    if metrics.has_cross_domain_reasoning:
        reasons.append("cross_domain_reasoning")
    if metrics.requires_ocr_spatial_or_chart_reasoning or metrics.requires_layout_sensitive_artifact:
        reasons.append("spatial_or_layout_sensitive_work")
    if (
        (metrics.estimated_input_tokens or metrics.serialized_character_count // 4)
        >= int(guards["large_shape_input_tokens"])
        or metrics.input_item_count >= int(guards["large_shape_history_items"])
        or metrics.dependent_step_count >= int(guards["large_shape_dependent_steps"])
        or metrics.artifact_object_count >= int(guards["large_shape_artifacts"])
        or metrics.deliverable_count >= int(guards["large_shape_deliverables"])
    ):
        reasons.append("outside_calibrated_request_shape")
    return reasons


def load_learned_model(path: str | Path) -> OrdinalLogisticRouter:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    model_payload = payload.get("model", payload)
    if model_payload.get("model_type") != "two_head_ordinal_logistic":
        raise ValueError("hybrid v3 expects a two_head_ordinal_logistic artifact")
    return OrdinalLogisticRouter.from_dict(model_payload)


def _learned_signal(
    metrics: RouteMetrics,
    deterministic_tier: str,
    model: OrdinalLogisticRouter | None,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    settings = policy["learned_second_opinion"]
    if not settings.get("enabled") or model is None:
        return {"available": False, "used_for_downgrade": False}, False
    prediction = model.predict_one(reduced_features(metrics))
    learned_rank = TIER_RANK[prediction.tier]
    deterministic_rank = TIER_RANK[deterministic_tier]
    rank_gap = abs(learned_rank - deterministic_rank)
    higher_confidence = (
        prediction.probability_strongest
        if learned_rank == 2
        else prediction.probability_balanced
    )
    strong_disagreement = rank_gap >= int(settings["strong_rank_gap"])
    confident_higher = (
        learned_rank > deterministic_rank
        and higher_confidence >= float(settings["higher_tier_probability"])
    )
    return {
        "available": True,
        "score": round(prediction.score, 6),
        "tier": prediction.tier,
        "probability_balanced": round(prediction.probability_balanced, 6),
        "probability_strongest": round(prediction.probability_strongest, 6),
        "rank_gap": rank_gap,
        "strong_disagreement": strong_disagreement,
        "confident_higher_prediction": confident_higher,
        "used_for_downgrade": False,
    }, strong_disagreement or confident_higher


def build_policy_score(
    metrics: RouteMetrics,
    policy: Mapping[str, Any],
    learned_model: OrdinalLogisticRouter | None = None,
) -> PolicyScore:
    config = PolicyConfig.from_dict(policy["optimized_config"])
    guards = policy["guards"]
    details = score_metrics_detailed(metrics, config)
    score = int(details["score"])
    score_only_tier = _tier_for_score(score, config)
    proposals: list[tuple[str, str]] = []

    margin = int(guards["threshold_margin"])
    if config.economical_threshold - margin <= score < config.economical_threshold:
        proposals.append(("balanced", "within_economical_threshold_margin"))
    elif config.strongest_threshold - margin <= score < config.strongest_threshold:
        proposals.append(("strongest", "within_strongest_threshold_margin"))

    hard_floor, hard_floor_reasons = _hard_floor(metrics, guards)
    if hard_floor is not None:
        proposals.extend((hard_floor, reason) for reason in hard_floor_reasons)

    ood = _ood_reasons(metrics, guards)
    if ood:
        # One unfamiliar signal is enough to protect an economical route, but
        # not enough to send every broad specialist/layout flag to strongest.
        # A balanced route needs two independent OOD signals for that step.
        if score_only_tier == "economical" or (
            score_only_tier == "balanced" and len(ood) >= 2
        ):
            proposals.extend((_promote(score_only_tier), f"ood:{reason}") for reason in ood)

    learned, learned_escalation = _learned_signal(
        metrics, score_only_tier, learned_model, policy
    )
    if learned_escalation:
        proposals.append((_promote(score_only_tier), "learned_strong_disagreement"))

    guarded_tier = score_only_tier
    for proposed_tier, _ in proposals:
        if TIER_RANK[proposed_tier] > TIER_RANK[guarded_tier]:
            guarded_tier = proposed_tier
    escalation_reasons = [
        reason for tier, reason in proposals if TIER_RANK[tier] > TIER_RANK[score_only_tier]
    ]
    matched_rules = list(details["matched_rules"])
    matched_rules.extend(f"guard.{reason}" for reason in escalation_reasons)
    return PolicyScore(
        score=score,
        component_scores=dict(details["components"]),
        component_raw_scores=dict(details["components"]),
        base_target_tier=guarded_tier,
        matched_rules=matched_rules,
        score_floor=details["score_floor"],
        score_floor_reasons=list(details["score_floor_reasons"]),
        metadata={
            "policy_type": "optimized_deterministic_with_one_way_uncertainty_guards",
            "score_only_tier": score_only_tier,
            "guarded_target_tier": guarded_tier,
            "thresholds": {
                "economical_below": config.economical_threshold,
                "strongest_at_or_above": config.strongest_threshold,
                "escalation_margin": margin,
            },
            "hard_floor_reasons": hard_floor_reasons,
            "ood_reasons": ood,
            "escalation_reasons": escalation_reasons,
            "learned_second_opinion": learned,
            "sticky_trajectory_policy": "initial choice is kept unless capability failure or repeated model-attributable failure requires escalation",
        },
    )


def evaluate_route(
    payload: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
    learned_model: OrdinalLogisticRouter | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_policy())
    router_input = parse_router_input(
        {**payload, "policy_version": str(policy["policy_version"])}
    )
    policy_score = build_policy_score(router_input.metrics, policy, learned_model)
    return asdict(evaluate_router_input(router_input, policy_score))


def _read_payload(path: str) -> Mapping[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_learned_model(
    policy: Mapping[str, Any], override: str | None, disabled: bool
) -> OrdinalLogisticRouter | None:
    if disabled or not policy["learned_second_opinion"].get("enabled"):
        return None
    configured = override or policy["learned_second_opinion"].get("artifact")
    if not configured:
        return None
    path = Path(str(configured))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    return load_learned_model(path)


def run_self_tests(policy: Mapping[str, Any]) -> dict[str, Any]:
    simple = evaluate_route({"metrics": {}}, policy=policy)
    assert simple["target_tier"] == "economical"

    margin_case = evaluate_route(
        {
            "metrics": {
                "primary_intent": "compare",
                "is_multi_file_modification": True,
            }
        },
        policy=policy,
    )
    assert margin_case["complexity_score"] == 35
    assert margin_case["target_tier"] == "balanced"

    risky = evaluate_route(
        {
            "metrics": {
                "action_risk": "destructive_public_deployment_or_permission_change"
            }
        },
        policy=policy,
    )
    assert risky["target_tier"] == "strongest"

    sticky = evaluate_route(
        {
            "mode": "mid_trajectory",
            "current_tier": "economical",
            "metrics": {"is_high_stakes_domain": True},
        },
        policy=policy,
    )
    assert sticky["decision"] == "KEEP"
    assert sticky["target_tier"] == "economical"
    return {
        "status": "ok",
        "checks": [
            "simple request remains economical",
            "threshold-margin escalation",
            "destructive-work strongest capability floor",
            "trajectory stickiness without observed failure",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="JSON input path, or '-' for stdin")
    parser.add_argument("--policy-config", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--learned-model")
    parser.add_argument("--no-learned", action="store_true")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        policy = load_policy(args.policy_config)
        learned_model = _resolve_learned_model(policy, args.learned_model, args.no_learned)
        if args.example:
            output: Any = example_payload()
        elif args.self_test:
            output = run_self_tests(policy)
        else:
            if not args.input:
                parser.error("provide an input path, '-', --example, or --self-test")
            output = evaluate_route(
                _read_payload(args.input), policy=policy, learned_model=learned_model
            )
        print(json.dumps(output, indent=2 if args.pretty or args.example else None, sort_keys=True))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
