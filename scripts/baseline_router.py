#!/usr/bin/env python3
"""Baseline heuristic router (starter idea 01) + cache-aware cost report.

Policy: short-prompt calls early in a trajectory go to a cheaper sibling model;
everything else stays on the logged model. Deliberately simple — an honest floor.

Usage: python scripts/baseline_router.py export/   -> writes results/routes.jsonl
"""
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from abstract_router import AbstractRouter, Request
from load_trajectories import iter_requests, group_trajectories, est_tokens
from cost_model import trajectory_cost, logged_route, load_pricing

# Cheaper sibling per family. Anonymized ids: within claude, sonnet is assumed the
# mid tier and fable the small tier; the gpt-5.6 variants (sol/terra/luna) have no
# published tier order. ASSUMPTION — check the reconstructed traces (e.g. which model
# gets long-context work) before trusting it.
CHEAP = {"claude": "claude-fable-5", "gpt": "gpt-5.6-sol"}

def cheap_for(model: str) -> str:
    return CHEAP["claude"] if model.startswith("claude") else CHEAP["gpt"]

SMALL_TRAJECTORY = 15_000  # est. input tokens; ~40% of real trajectories fall under this

class BaselineRouter(AbstractRouter):
    """Route small trajectories to a cheaper sibling model."""

    def route_trajectory(self, calls: Sequence[Request]) -> list[str]:
        """Send whole small trajectories to the cheap sibling.

        Whole-trajectory routing respects the one-model-per-trajectory premise
        and never pays the cache-reset penalty for a mid-task switch.
        """
        total = sum(est_tokens(call["input"]) for call in calls)
        if total < SMALL_TRAJECTORY:
            return [cheap_for(call["model"]) for call in calls]
        return [call["model"] for call in calls]

    def run(self, export: str | Path = "export") -> None:
        """Route all trajectories in ``export`` and write the cost report."""
        pricing = load_pricing()
        groups = group_trajectories(
            request for _, _, request in iter_requests(export)
        )
        output_path = Path("results/routes.jsonl")
        output_path.parent.mkdir(exist_ok=True)
        total_logged = total_routed = 0.0

        with output_path.open("w") as output:
            for key, calls in groups.items():
                logged = logged_route(calls)
                routed = self.route_trajectory(calls)
                cost_logged, _ = trajectory_cost(calls, logged, pricing)
                cost_routed, _ = trajectory_cost(calls, routed, pricing)
                total_logged += cost_logged
                total_routed += cost_routed
                output.write(
                    json.dumps(
                        {
                            "trajectory": key,
                            "n_calls": len(calls),
                            "logged_model": logged[0],
                            "route": routed,
                            "cost_logged_usd": round(cost_logged, 6),
                            "cost_routed_usd": round(cost_routed, 6),
                            "switches": sum(
                                1
                                for index in range(1, len(routed))
                                if routed[index] != routed[index - 1]
                            ),
                        }
                    )
                    + "\n"
                )

        print(
            "logged cost (est. input tokens, assumed prices):  "
            f"${total_logged:,.4f}"
        )
        print(
            f"routed cost:  ${total_routed:,.4f}  "
            f"({(total_routed / total_logged - 1):+.1%}, cache-aware)"
        )
        print(
            "NOTE: no outputs/usage in the export — token counts are estimates, "
            "output cost excluded,"
        )
        print(
            "and this baseline has NO outcome estimate. Constructing one is the "
            "challenge."
        )
        print(f"wrote {output_path}")


def route_trajectory(calls: Sequence[Request]) -> list[str]:
    """Compatibility wrapper for callers of the original module function."""
    return BaselineRouter().route_trajectory(calls)


def main() -> None:
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    BaselineRouter().run(export)


if __name__ == "__main__":
    main()
