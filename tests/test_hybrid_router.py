from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from deterministic_router import evaluate_route as evaluate_v2  # noqa: E402
from hybrid_router import evaluate_route, load_policy  # noqa: E402
from learned_router.deterministic_optimizer import (  # noqa: E402
    PolicyConfig,
    score_metrics,
    score_metrics_detailed,
)
from learned_router.hybrid_frontier import _failure_aware_proxy  # noqa: E402


class OptimizedScoreBreakdownTests(unittest.TestCase):
    def test_breakdown_sums_to_raw_score_and_matches_public_scorer(self) -> None:
        metrics = {
            "estimated_input_tokens": 30_000,
            "input_item_count": 40,
            "primary_intent": "plan_or_design",
            "expected_action": "multiple_writes_or_chained_tools",
            "requires_testing_or_verification": True,
            "is_multi_file_modification": True,
            "has_code_sql_formula_or_data_transformation": True,
        }
        config = PolicyConfig()
        details = score_metrics_detailed(metrics, config)
        self.assertEqual(sum(details["components"].values()), details["raw_score"])
        self.assertEqual(score_metrics(metrics, config), details["score"])


class HybridPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy(ROOT / "scripts" / "router_policies" / "hybrid_v3.json")

    def test_frozen_v2_remains_the_default_router(self) -> None:
        decision = evaluate_v2({"metrics": {"primary_intent": "compare"}})
        self.assertEqual(decision["policy_version"], "complexity-v2-simple")
        self.assertEqual(decision["policy_metadata"], {})

    def test_simple_request_stays_economical(self) -> None:
        decision = evaluate_route({"metrics": {}}, policy=self.policy)
        self.assertEqual(decision["target_tier"], "economical")
        self.assertEqual(decision["complexity_score"], 0)

    def test_margin_escalation_is_visible_without_changing_score(self) -> None:
        decision = evaluate_route(
            {
                "metrics": {
                    "primary_intent": "compare",
                    "is_multi_file_modification": True,
                }
            },
            policy=self.policy,
        )
        self.assertEqual(decision["complexity_score"], 35)
        self.assertEqual(decision["policy_metadata"]["score_only_tier"], "economical")
        self.assertEqual(decision["target_tier"], "balanced")

    def test_destructive_work_gets_strongest_floor(self) -> None:
        decision = evaluate_route(
            {
                "metrics": {
                    "action_risk": "destructive_public_deployment_or_permission_change"
                }
            },
            policy=self.policy,
        )
        self.assertEqual(decision["target_tier"], "strongest")
        self.assertIn(
            "compound_high_risk_or_destructive_work",
            decision["policy_metadata"]["hard_floor_reasons"],
        )

    def test_mid_trajectory_policy_is_sticky_without_failure(self) -> None:
        decision = evaluate_route(
            {
                "mode": "mid_trajectory",
                "current_tier": "economical",
                "metrics": {
                    "action_risk": "destructive_public_deployment_or_permission_change"
                },
            },
            policy=self.policy,
        )
        self.assertEqual(decision["decision"], "KEEP")
        self.assertEqual(decision["target_tier"], "economical")


class FrontierProxyTests(unittest.TestCase):
    def test_failed_trajectory_includes_always_high_retry(self) -> None:
        records = [
            {"instance_id": "a", "gold_merged_tier_id": 2, "prediction": "economical"},
            {"instance_id": "a", "gold_merged_tier_id": 0, "prediction": "economical"},
        ]
        self.assertLess(_failure_aware_proxy(records, "prediction"), 0.0)


if __name__ == "__main__":
    unittest.main()
