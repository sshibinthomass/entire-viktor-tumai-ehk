# Optimized deterministic-router evaluation

Every optimized headline prediction is from an outer fold whose labels were not used to select its weights. Weights were searched only inside the corresponding training partition.

| Policy | Exact tier | Safe steps | Under-routed | Over-routed | Row-weighted trajectory pass | Mean official combined |
|---|---:|---:|---:|---:|---:|---:|
| Frozen current | 48.0% | 82.1% | 17.9% | 34.0% | 62.7% | 59.2% |
| Nested optimized | 54.1% | 97.4% | 2.6% | 43.3% | 95.1% | 75.6% |

## Leave-one-benchmark-out

Overall safety: 81.8%; row-weighted trajectory pass: 60.7%.

| Held-out benchmark | Safe steps | Exact tier | Row-weighted trajectory pass |
|---|---:|---:|---:|
| bfcl | 97.2% | 71.8% | 93.5% |
| mtrag | 99.0% | 41.5% | 99.0% |
| pinchbench | 85.4% | 66.7% | 47.9% |
| qmsum | 93.8% | 49.7% | 93.8% |
| swebench | 54.8% | 25.6% | 2.1% |

## Final configuration

```json
{
  "analytic_intent_weight": 15,
  "chained_action_weight": 10,
  "compare_intent_weight": 15,
  "context_unit": 5,
  "debug_intent_weight": 30,
  "deep_intent_weight": 25,
  "economical_threshold": 40,
  "history_weight": 5,
  "light_intent_weight": 5,
  "reasoning_modifier_weight": 0,
  "scope_unit": 5,
  "simple_action_weight": 0,
  "software_engineering_interaction": 5,
  "special_requirement_unit": 5,
  "specialist_interaction": 0,
  "strongest_threshold": 75,
  "testing_weight": 5
}
```

The final configuration is fitted on all public rows for deployment experiments. Its in-sample performance is not used as an evaluation result.
