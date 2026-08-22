from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from learned_router.features import (  # noqa: E402
    EXCLUDED_SOURCE_FIELDS,
    RAW_FEATURES,
    FeatureEncoder,
    reduced_features,
)
from learned_router.models import (  # noqa: E402
    OrdinalBoostedRouter,
    OrdinalLogisticRouter,
)
from learned_router.compare_baseline_matrix import build_matrix  # noqa: E402
from learned_router.deterministic_optimizer import (  # noqa: E402
    DEFAULT_CONFIG,
    PolicyConfig,
    is_valid_config,
    optimize_policy,
    score_metrics,
)
from deterministic_router import RouteMetrics, calculate_complexity  # noqa: E402


def _feature_row(tokens: int, action: str, dependent_steps: int) -> dict[str, float | str]:
    return reduced_features(
        {
            "estimated_input_tokens": tokens,
            "input_item_count": dependent_steps + 2,
            "primary_intent": (
                "extract_or_classify"
                if tokens < 1_000
                else "analyze"
                if tokens < 10_000
                else "debug_prove_or_diagnose"
            ),
            "expected_action": action,
            "dependent_step_count": dependent_steps,
            "artifact_object_count": max(1, dependent_steps),
            "deliverable_count": 1 if tokens < 1_000 else 2,
            "hard_constraint_count": dependent_steps // 2,
            "is_multi_file_modification": tokens >= 10_000,
            "requires_testing_or_verification": tokens >= 10_000,
            "has_code_sql_formula_or_data_transformation": tokens >= 1_000,
            "has_debug_error_failing_test_or_stacktrace": tokens >= 10_000,
        }
    )


class FeatureContractTests(unittest.TestCase):
    def test_contract_has_15_raw_and_24_encoded_features(self) -> None:
        rows = [_feature_row(100, "no_tool", 0), _feature_row(10_000, "one_local_write", 3)]
        encoder = FeatureEncoder.fit(rows)
        self.assertEqual(len(RAW_FEATURES), 15)
        self.assertEqual(encoder.input_dimension, 24)

    def test_leakage_fields_are_explicitly_excluded(self) -> None:
        for field in (
            "request_model",
            "reasoning_item_count",
            "tool_argument_fingerprints",
            "shared_prefix_tokens",
        ):
            self.assertIn(field, EXCLUDED_SOURCE_FIELDS)


class LearnedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows: list[dict[str, float | str]] = []
        cls.labels: list[int] = []
        for offset in range(15):
            cls.rows.append(_feature_row(100 + offset * 10, "no_tool", 0))
            cls.labels.append(0)
            cls.rows.append(
                _feature_row(2_000 + offset * 100, "search_or_multiple_reads", 2)
            )
            cls.labels.append(1)
            cls.rows.append(
                _feature_row(20_000 + offset * 1_000, "multiple_writes_or_chained_tools", 5)
            )
            cls.labels.append(2)

    def test_logistic_model_is_ordered_and_round_trips(self) -> None:
        model = OrdinalLogisticRouter.fit(self.rows, self.labels)
        scores = [model.predict_one(self.rows[index]).score for index in (0, 1, 2)]
        restored = OrdinalLogisticRouter.from_dict(model.to_dict())

        self.assertEqual(model.trainable_parameter_count, 50)
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
        self.assertAlmostEqual(
            model.predict_one(self.rows[2]).score,
            restored.predict_one(self.rows[2]).score,
            places=10,
        )

    def test_boosted_model_is_ordered_and_round_trips(self) -> None:
        model = OrdinalBoostedRouter.fit(
            self.rows,
            self.labels,
            tree_count=12,
            min_samples_leaf=3,
            max_bins=8,
        )
        scores = [model.predict_one(self.rows[index]).score for index in (0, 1, 2)]
        restored = OrdinalBoostedRouter.from_dict(model.to_dict())

        self.assertGreater(model.learned_leaf_count, 0)
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
        self.assertAlmostEqual(
            model.predict_one(self.rows[2]).score,
            restored.predict_one(self.rows[2]).score,
            places=10,
        )


class MatrixComparisonTests(unittest.TestCase):
    def test_matrix_orientation_and_summary(self) -> None:
        result = build_matrix(
            [
                ("economical", "economical"),
                ("balanced", "strongest"),
                ("strongest", "balanced"),
                ("strongest", "strongest"),
            ]
        )
        self.assertEqual(result["counts"]["balanced"]["strongest"], 1)
        self.assertEqual(result["summary"]["exact_choice_agreement_count"], 2)
        self.assertEqual(result["summary"]["learned_lower_than_baseline_count"], 1)
        self.assertEqual(result["summary"]["learned_higher_than_baseline_count"], 1)


class DeterministicOptimizerTests(unittest.TestCase):
    def test_default_parameterization_matches_current_complexity_score(self) -> None:
        cases = [
            RouteMetrics(),
            RouteMetrics(primary_intent="compare", expected_action="one_local_write"),
            RouteMetrics(
                estimated_input_tokens=30_000,
                input_item_count=40,
                primary_intent="plan_or_design",
                expected_action="multiple_writes_or_chained_tools",
                requires_testing_or_verification=True,
                is_multi_file_modification=True,
                requires_strict_schema_or_machine_output=True,
            ),
        ]
        for metrics in cases:
            self.assertEqual(
                score_metrics(metrics, DEFAULT_CONFIG),
                calculate_complexity(metrics)[0],
            )

    def test_constraints_reject_reversed_severity(self) -> None:
        self.assertTrue(is_valid_config(DEFAULT_CONFIG))
        self.assertFalse(
            is_valid_config(PolicyConfig(light_intent_weight=20))
        )

    def test_optimizer_adds_capacity_for_software_tasks(self) -> None:
        rows = []
        for index in range(24):
            hard = index >= 12
            metrics = RouteMetrics(
                primary_intent="plan_or_design" if hard else "extract_or_classify",
                expected_action=(
                    "multiple_writes_or_chained_tools" if hard else "no_tool"
                ),
                requires_testing_or_verification=hard,
                is_multi_file_modification=hard,
                has_code_sql_formula_or_data_transformation=hard,
            )
            rows.append(
                {
                    "instance_id": f"instance-{index}",
                    "metrics": asdict(metrics),
                    "gold_tier_id": 2 if hard else 0,
                }
            )
        config, _ = optimize_policy(rows, [index % 4 for index in range(len(rows))])
        self.assertNotEqual(config, DEFAULT_CONFIG)
        self.assertGreaterEqual(score_metrics(rows[-1]["metrics"], config), config.strongest_threshold)

if __name__ == "__main__":
    unittest.main()
