"""The ONE rank transform used everywhere (router, evaluator, benchmarks, JS).

Right-ECDF fit on a train slice and applied to any values:

    rank(x) = |{t in train : t <= x}| / |train|

Properties the rest of the repo relies on:
  - fit/apply split: ranks can be fit per CV fold (train rows only) and applied
    to test rows — no test information enters the transform
  - constant train columns rank to a NEUTRAL 0.5 (not 0, not 1), so a
    degenerate metric contributes nothing instead of deflating scores
  - vectorized (sort + searchsorted): O(n log n), not O(n * unique)

dashboard/index.html ports exactly these semantics to JS so the lab and the
Python benchmarks compute identical numbers.
"""
import numpy as np


def ecdf_ranks(train_vals, all_vals):
    """Right-ECDF of `all_vals` under the empirical distribution of `train_vals`."""
    s = np.sort(np.asarray(train_vals, dtype=float))
    n = len(s)
    if n == 0:
        return np.full(len(np.atleast_1d(all_vals)), 0.5)
    if s[0] == s[-1]:  # constant train column -> neutral
        return np.full(len(np.atleast_1d(all_vals)), 0.5)
    return np.searchsorted(s, np.asarray(all_vals, dtype=float), side="right") / n


def self_ranks(values):
    """ecdf_ranks of values against themselves (the full-dataset transform)."""
    return ecdf_ranks(values, values)
