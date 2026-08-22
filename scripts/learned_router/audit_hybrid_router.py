#!/usr/bin/env python3
"""Aggregate-only audit of the deployable hybrid-v3 policy.

No prompts or trajectory contents are written. Twin results use the full-data
deployment fit and are therefore a diagnostic, not an out-of-sample estimate.
The nested OOF frontier remains the honest headline evaluation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_twinrouterbench import (  # noqa: E402
    TIER_MODEL_MAP,
    TWIN_TIER_TO_MERGED_ID,
    _load_rows,
    compute_merged_metrics,
    twin_row_to_request,
)
from deterministic_router import evaluate_route as evaluate_v2  # noqa: E402
from extract_router_metrics import metrics_from_request  # noqa: E402
from hybrid_router import (  # noqa: E402
    ROOT,
    evaluate_route as evaluate_v3,
    load_learned_model,
    load_policy,
)
from learned_router.hybrid_frontier import _failure_aware_proxy  # noqa: E402
from load_trajectories import (  # noqa: E402
    group_trajectories,
    is_generated_synthetic,
    iter_requests,
)


def _learned(policy: dict[str, Any]):
    settings = policy["learned_second_opinion"]
    if not settings.get("enabled") or not settings.get("artifact"):
        return None
    path = Path(str(settings["artifact"]))
    if not path.is_absolute():
        path = ROOT / path
    return load_learned_model(path) if path.exists() else None


def _twin_audit(twin_repo: Path, policy: dict[str, Any], model: Any) -> dict[str, Any]:
    rows = _load_rows(twin_repo / "data" / "static" / "question_bank.jsonl")
    records: list[dict[str, Any]] = []
    escalation_reasons: Counter[str] = Counter()
    for row in rows:
        metrics = metrics_from_request(twin_row_to_request(row))
        decision = evaluate_v3(
            {"metrics": asdict(metrics), "tier_model_map": TIER_MODEL_MAP},
            policy=policy,
            learned_model=model,
        )
        escalation_reasons.update(decision["policy_metadata"]["escalation_reasons"])
        records.append(
            {
                "id": str(row["id"]),
                "benchmark": str(row["benchmark"]),
                "instance_id": str(row.get("instance_id", row["id"])),
                "gold_merged_tier_id": TWIN_TIER_TO_MERGED_ID[str(row["target_tier"])],
                "prediction": decision["target_tier"],
            }
        )
    metrics = compute_merged_metrics(records, "prediction")
    metrics["failure_aware_savings_proxy_percent"] = round(
        _failure_aware_proxy(records, "prediction"), 6
    )
    return {
        "warning": "full-data deployment fit on its source benchmark; diagnostic only",
        "metrics": metrics,
        "escalation_reasons": dict(escalation_reasons.most_common()),
    }


def _viktor_audit(export_dir: Path, policy: dict[str, Any], model: Any) -> dict[str, Any]:
    requests = [
        request
        for _, _, request in iter_requests(export_dir)
        if not is_generated_synthetic(request)
    ]
    groups = group_trajectories(requests)
    clean = [
        calls for calls in groups.values() if len({call["model"] for call in calls}) == 1
    ]
    v2_counts: Counter[str] = Counter()
    v3_counts: Counter[str] = Counter()
    matrix: Counter[tuple[str, str]] = Counter()
    escalation_reasons: Counter[str] = Counter()
    ood_reasons: Counter[str] = Counter()
    learned_tiers: Counter[str] = Counter()
    scores: list[int] = []
    for calls in clean:
        metrics = metrics_from_request(calls[0])
        payload = {"metrics": asdict(metrics)}
        v2 = evaluate_v2(payload)
        v3 = evaluate_v3(payload, policy=policy, learned_model=model)
        v2_tier = str(v2["target_tier"])
        v3_tier = str(v3["target_tier"])
        v2_counts[v2_tier] += 1
        v3_counts[v3_tier] += 1
        matrix[(v2_tier, v3_tier)] += 1
        scores.append(int(v3["complexity_score"]))
        metadata = v3["policy_metadata"]
        escalation_reasons.update(metadata["escalation_reasons"])
        ood_reasons.update(metadata["ood_reasons"])
        learned = metadata["learned_second_opinion"]
        if learned.get("available"):
            learned_tiers[str(learned["tier"])] += 1
    return {
        "scope": {
            "real_requests": len(requests),
            "clean_trajectories": len(clean),
            "mixed_model_groups_excluded": len(groups) - len(clean),
            "routing_unit": "first request; one sticky choice for the full trajectory",
        },
        "tier_counts": {
            "frozen_v2": dict(v2_counts),
            "hybrid_v3": dict(v3_counts),
            "learned_second_opinion": dict(learned_tiers),
        },
        "v2_rows_v3_columns": {
            source: {target: matrix[(source, target)] for target in ("economical", "balanced", "strongest")}
            for source in ("economical", "balanced", "strongest")
        },
        "score": {
            "min": min(scores),
            "median": statistics.median(scores),
            "mean": round(statistics.mean(scores), 3),
            "max": max(scores),
        },
        "escalation_reasons": dict(escalation_reasons.most_common()),
        "ood_reasons": dict(ood_reasons.most_common()),
        "quality_warning": "Viktor has no counterfactual quality labels; these are routing-distribution diagnostics only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", default="export")
    parser.add_argument("--twin-repo", default=".external/TwinRouterBench")
    parser.add_argument("--policy-config", default="scripts/router_policies/hybrid_v3.json")
    parser.add_argument("--output", default="results/hybrid_router/deployment_audit.json")
    args = parser.parse_args()
    policy = load_policy(args.policy_config)
    model = _learned(policy)
    result = {
        "policy_version": policy["policy_version"],
        "learned_second_opinion_loaded": model is not None,
        "twin": _twin_audit(Path(args.twin_repo), policy, model),
        "viktor": _viktor_audit(Path(args.export_dir), policy, model),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
