# TwinRouterBench frozen-router backtest

The router thresholds (economical <35, balanced 35-74, strongest >=75) were frozen before evaluation.
Twin `mid` and `mid_high` are merged into the router's `balanced` tier for classification metrics.

| Policy | Exact tier | Safe steps | Under-routed | Trajectory pass (unweighted) | Trajectory pass (row-weighted) | Failure-aware cost saving |
|---|---:|---:|---:|---:|---:|---:|
| Independent step | 48.0% | 82.1% | 17.9% | 88.7% | 62.7% | 49.7-50.8% |
| Stateful keep/upgrade | 50.2% | 81.2% | 18.8% | 88.7% | 62.7% | 54.3-55.4% |

Unweighted trajectory pass gives every trajectory one vote. Twin's official score is row-weighted, so long failed trajectories matter more. The cost range is the sensitivity result from mapping our single `balanced` tier to Twin's `mid` versus `mid_high` tier.

## Independent policy by benchmark

| Benchmark | Rows | Exact tier | Safe steps | Row-weighted trajectory pass |
|---|---:|---:|---:|---:|
| bfcl | 248 | 71.4% | 97.2% | 93.5% |
| mtrag | 193 | 42.5% | 99.0% | 99.0% |
| pinchbench | 48 | 66.7% | 85.4% | 47.9% |
| qmsum | 145 | 62.1% | 93.8% | 93.8% |
| swebench | 336 | 25.3% | 55.7% | 7.7% |

The main weakness is SWE-bench: 55.7% safe steps and 7.7% row-weighted trajectory pass. The other four benchmark slices each exceed 85% safe steps.

## Scope

- Rows: 970
- Trajectories: 520
- TwinRouterBench commit: `7cbb0deac8f697b5faa8489c309560e53d2ef088`
- Question-bank SHA-256: `7e2870b5e2e5c801f6444c05a4311c9c9010e965016f6938f0bb5abc226252d0`
- Official cost tokenizer: TwinRouterBench documented cl100k_base fallback for all tiers; native HuggingFace tokenizers not installed
- Static fixed-prefix evaluation only; no model inference was performed.

## Interpretation limits

- Twin labels are cheapest-sufficient-tier estimates under its fixed pool and downgrade protocol, not universal model optima.
- The stateful simulation carries our tier decision forward over recorded prefixes; it cannot generate the counterfactual prefix our chosen model would have produced.
- Twin's two middle tiers have no exact one-to-one mapping to our single balanced tier, so official cost scores are reported under both mappings.
- This validates tier calibration on an external agentic benchmark; it does not identify the best anonymized Viktor model.
