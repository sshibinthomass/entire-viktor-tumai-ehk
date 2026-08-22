# Learned router experiment

This package trains two small ordinal routers from the 15-feature projection of
`RouteMetrics`:

- two-head L2-regularized logistic regression;
- two-head depth-2 gradient-boosted trees.

Both models emit `P(needs at least balanced)` and `P(needs strongest)`. The
single routing score is frozen as:

```text
10 + 45 * P(balanced) + 35 * min(P(strongest), P(balanced))
```

Scores below 35 route to economical, scores from 35 through 74 route to
balanced, and scores at or above 75 route to strongest.

Run from the repository root:

```powershell
python scripts\learned_router\train_evaluate.py `
  --twin-repo .external\TwinRouterBench `
  --output-dir results\learned_router
```

Use `--skip-official` when TwinRouterBench's tokenizer dependency is not
installed. Grouped tier-quality evaluation and both model implementations use
only the Python standard library.

The reported five-fold predictions group all prefixes of one trajectory in the
same fold. Leave-one-benchmark-out results are also written. Final model JSON
files are fitted on all rows only after evaluation predictions are complete.

## Optimized deterministic policy

`optimize_deterministic.py` tunes 15 interpretable weight families and the two
routing thresholds in five-point increments. It uses nested trajectory-grouped
evaluation, regularizes toward the frozen policy, preserves monotonicity and
safety floors, and runs a leave-one-benchmark-out stress test.

```powershell
python scripts\learned_router\optimize_deterministic.py `
  --twin-repo .external\TwinRouterBench `
  --output-dir results\optimized_deterministic
```

The deployment experiment configuration is written to
`results/optimized_deterministic/optimized_policy.json`. The nested outer-fold
metrics, rather than its in-sample full-data fit, are the evaluation estimate.

## Deployable hybrid v3

`scripts/hybrid_router.py` is a separate, versioned production entry point. It
uses the optimized deterministic weights, conservative hard floors, a five-point
threshold margin, OOD escalation, and the ordinal logistic model as a one-way
second opinion. The learned model can escalate uncertainty but cannot downgrade
the deterministic choice. The initial choice remains sticky for the trajectory;
mid-trajectory switching requires a capability failure or repeated attributable
failure so provider-prefix caching is preserved.

```powershell
python scripts\hybrid_router.py request_metrics.json --pretty
python scripts\hybrid_router.py --self-test
```

The policy is in `scripts/router_policies/hybrid_v3.json`. The frozen
`deterministic_router.py` remains the v2 comparison baseline.

Generate the nested-OOF score frontier and aggregate deployment audit with:

```powershell
python scripts\learned_router\hybrid_frontier.py
python scripts\learned_router\audit_hybrid_router.py
```

The frontier's full sweep uses the optimizer's documented 0.5/3.5/25
failure-aware tier-price proxy so it stays laptop-safe. The already-computed
Twin official two-mapping score for the selected nested policy is retained as a
separate reference. The sweep is threshold sensitivity, not fresh validation
for a threshold selected after viewing the curve.
