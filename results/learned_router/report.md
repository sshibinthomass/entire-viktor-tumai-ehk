# Learned-router evaluation

All headline predictions are out-of-fold by complete trajectory. The final saved models were trained only after evaluation predictions were complete.

## Grouped five-fold results

| Model | Exact tier | Safe steps | Under-routed | Over-routed | Row-weighted trajectory pass | Failure-aware cost saving | Official combined |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinal_logistic | 77.2% | 94.6% | 5.4% | 17.4% | 87.0% | 46.8-53.0% | 76.1% |
| ordinal_boosted_trees | 78.8% | 94.4% | 5.6% | 15.7% | 84.9% | 41.2-50.0% | 74.5% |
| frozen deterministic | 48.0% | 82.1% | 17.9% | 34.0% | 62.7% | n/a | n/a |

## Leave-one-benchmark-out stress test

| Model | Exact tier | Safe steps | Under-routed | Row-weighted trajectory pass |
|---|---:|---:|---:|---:|
| ordinal_logistic | 57.2% | 72.0% | 28.0% | 60.8% |
| ordinal_boosted_trees | 68.5% | 71.3% | 28.7% | 59.5% |

### Safety when each benchmark is completely unseen

| Held-out benchmark | Logistic safe steps | Boosted-tree safe steps |
|---|---:|---:|
| bfcl | 97.6% | 97.6% |
| mtrag | 94.8% | 94.8% |
| pinchbench | 91.7% | 85.4% |
| qmsum | 91.7% | 91.0% |
| swebench | 28.6% | 28.0% |

Selected model: **ordinal_logistic**.
Selection criterion: mean TwinRouterBench official combined score across both middle-tier mappings.
This is the experiment winner, not a deployment approval: both learned models fall below the deterministic router's overall safety when the benchmark family is unseen.

## Important limitations

- Twin target tiers are pool- and protocol-specific cheapest-sufficient estimates.
- Prefixes from one trajectory are correlated; grouped evaluation prevents them crossing train/test boundaries.
- Leave-one-benchmark-out performance is the more pessimistic estimate of transfer to unseen task families.
- The 35/75 thresholds and score formula were frozen before this evaluation.
