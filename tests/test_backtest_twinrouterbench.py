from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_twinrouterbench import (  # noqa: E402
    OFFICIAL_BALANCED_MAPS,
    ROUTER_TIERS,
    TWIN_TIER_TO_MERGED_ID,
    _official_eval_rows,
    compute_merged_metrics,
    twin_messages_to_response_items,
)


class MessageConversionTests(unittest.TestCase):
    def test_converts_assistant_tool_call_and_result(self) -> None:
        messages = [
            {"role": "system", "content": "Use the tool."},
            {"role": "user", "content": "Look it up."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": {"q": "x"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "found"},
        ]

        items = twin_messages_to_response_items(messages)

        self.assertEqual(
            [item["type"] for item in items],
            ["message", "message", "function_call", "function_call_output"],
        )
        self.assertEqual(items[2]["name"], "search")
        self.assertEqual(items[2]["arguments"], '{"q": "x"}')
        self.assertEqual(items[3]["call_id"], "call-1")

    def test_keeps_textual_assistant_message(self) -> None:
        items = twin_messages_to_response_items(
            [{"role": "assistant", "content": "working", "tool_calls": []}]
        )
        self.assertEqual(items[0]["role"], "assistant")
        self.assertEqual(items[0]["content"][0]["type"], "output_text")


class TierEvaluationTests(unittest.TestCase):
    def _record(
        self,
        row_id: str,
        instance: str,
        gold: int,
        prediction: str,
        benchmark: str = "toy",
    ) -> dict[str, object]:
        return {
            "id": row_id,
            "instance_id": instance,
            "benchmark": benchmark,
            "gold_merged_tier_id": gold,
            "prediction": prediction,
        }

    def test_middle_twin_tiers_merge_to_balanced(self) -> None:
        self.assertEqual(TWIN_TIER_TO_MERGED_ID["low"], 0)
        self.assertEqual(TWIN_TIER_TO_MERGED_ID["mid"], 1)
        self.assertEqual(TWIN_TIER_TO_MERGED_ID["mid_high"], 1)
        self.assertEqual(TWIN_TIER_TO_MERGED_ID["high"], 2)

    def test_merged_metrics_use_all_steps_for_trajectory_pass(self) -> None:
        records = [
            self._record("a-1", "a", 0, "economical"),
            self._record("a-2", "a", 2, "balanced"),
            self._record("b-1", "b", 1, "strongest"),
        ]

        result = compute_merged_metrics(records, "prediction")

        self.assertEqual(result["exact_rows"], 1)
        self.assertEqual(result["safe_rows"], 2)
        self.assertEqual(result["passed_trajectories"], 1)
        self.assertAlmostEqual(result["trajectory_pass_rate_percent"], 50.0)
        self.assertEqual(result["confusion"]["strongest"]["balanced"], 1)

    def test_official_middle_mapping_is_reported_as_two_bounds(self) -> None:
        source = [
            {
                "id": "row",
                "benchmark": "toy",
                "target_tier_id": 2,
                "instance_id": "instance",
                "messages": [],
            }
        ]
        records = {"row": {"prediction": ROUTER_TIERS[1]}}

        optimistic = _official_eval_rows(
            source,
            records,
            "prediction",
            OFFICIAL_BALANCED_MAPS["balanced_as_mid"],
        )
        conservative = _official_eval_rows(
            source,
            records,
            "prediction",
            OFFICIAL_BALANCED_MAPS["balanced_as_mid_high"],
        )

        self.assertEqual(optimistic[0]["pred_tier_id"], 1)
        self.assertFalse(optimistic[0]["passed"])
        self.assertEqual(conservative[0]["pred_tier_id"], 2)
        self.assertTrue(conservative[0]["passed"])


if __name__ == "__main__":
    unittest.main()
