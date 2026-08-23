"""Rule-based difficulty (1/2/3) from trajectory effort metrics.

Recipe (every number below is a tunable hyperparameter — the dashboard exposes
all of them):
  1. percentile-rank each metric across the dataset (robust to heavy tails)
  2. difficulty score = weighted mean of the ranks
  3. cut at dataset percentiles -> difficulty 1/2/3
  4. hard overrides: pathological trajectories (error storms, tool-call
     marathons) are promoted to difficulty 3 regardless of their score

Policy-dependence correction (optional, `residual_tiers`): effort metrics are
shaped by WHICH model served the trajectory — weaker models retry more — so
tasks historically served by weak models would be labeled harder. Passing the
inferred tier of each trajectory's logged model residualizes the two
policy-sensitive metrics (`n_tool_errors`, `max_repeat_streak`) by subtracting
the per-tier median rank before weighting.

Caveats to keep in the writeup:
  - effort is a PROXY for difficulty: long-and-easy vs short-and-hard exists
  - reasoning_tokens only exist for the gpt family, so its default weight is
    kept small to limit provider bias
  - the deepest LOGGED call may still be mid-task; all counts are lower bounds
"""
import numpy as np

from rank_utils import self_ranks

DEFAULT_WEIGHTS = {
    "n_tool_calls": 0.22,
    "n_llm_calls": 0.18,
    "gen_tokens": 0.16,
    "context_tokens": 0.10,
    "n_tool_errors": 0.10,
    "n_distinct_tools": 0.08,
    "tool_output_tokens": 0.06,
    "max_repeat_streak": 0.04,
    "n_user_turns": 0.03,
    "reasoning_tokens": 0.03,
}

DEFAULT_CUTS = (0.55, 0.85)          # score percentiles for the 1|2 and 2|3 borders
DEFAULT_OVERRIDES = {"errors_t3": 5, "tool_calls_t3": 40}  # promote-to-3 thresholds

POLICY_SENSITIVE = ("n_tool_errors", "max_repeat_streak")


def percentile_ranks(values):
    """Canonical right-ECDF ranks (see rank_utils); constant columns -> 0.5."""
    return self_ranks(values)


def difficulty_scores(metric_rows, weights=None, residual_tiers=None):
    """Weighted mean of metric percentile ranks -> score in [0, 1].

    With `residual_tiers` (inferred tier of each trajectory's logged model),
    the policy-sensitive metrics are residualized per tier: rank minus the
    tier's median rank, recentred at 0.5."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(w.values())
    tiers = None if residual_tiers is None else np.asarray(residual_tiers)
    score = np.zeros(len(metric_rows))
    for metric, weight in w.items():
        r = percentile_ranks([row[metric] for row in metric_rows])
        if tiers is not None and metric in POLICY_SENSITIVE:
            for t in np.unique(tiers):
                m = tiers == t
                r[m] = np.clip(r[m] - np.median(r[m]) + 0.5, 0, 1)
        score += weight * r
    return score / total


def grade(metric_rows, weights=None, cuts=DEFAULT_CUTS, overrides=DEFAULT_OVERRIDES,
          residual_tiers=None):
    """-> (difficulty 1/2/3 per trajectory, raw scores)."""
    scores = difficulty_scores(metric_rows, weights, residual_tiers)
    c1, c2 = np.quantile(scores, cuts[0]), np.quantile(scores, cuts[1])
    diff = np.where(scores <= c1, 1, np.where(scores <= c2, 2, 3)).astype(int)
    if overrides:
        for i, r in enumerate(metric_rows):
            if (r["n_tool_errors"] >= overrides.get("errors_t3", 10**9)
                    or r["n_tool_calls"] >= overrides.get("tool_calls_t3", 10**9)):
                diff[i] = 3
    return diff, scores
