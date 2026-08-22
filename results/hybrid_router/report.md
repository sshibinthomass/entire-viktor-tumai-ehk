# Hybrid-v3 router

`hybrid-v3-optimized-guarded-2026-08-22` is a trajectory-level, interpretable
router. It chooses once at the first request and keeps that model unless a
capability failure or repeated model-attributable failure justifies paying the
cache-reset cost.

## Decision stack

1. Validate context, tool, image, structured-output, language, and data-tag capabilities.
2. Score the request with the optimized monotonic deterministic policy.
3. Apply explicit threshold-margin, risk, complex-software, and OOD guards.
4. Consult the ordinal logistic model only as a one-way uncertainty signal; it cannot downgrade.
5. Select the cheapest eligible model at or above the resulting tier.

The policy and every escalation reason are returned in `policy_metadata`.

## Evidence

The optimized score's nested trajectory-grouped out-of-fold result on 970
TwinRouterBench rows was 97.4% safe steps and 95.1% row-weighted trajectory
pass. Its previously computed official failure-aware saving was 56.7–57.3%,
depending on how this router's one balanced tier maps to Twin's two middle tiers.

The runtime score thresholds plus five-point margin correspond to the 35/70
point in the score-only sensitivity sweep: 97.6% safe steps, 95.5% row-weighted
trajectory pass, and 55.2% failure-aware tier-price-proxy saving. The hard and
OOD guards are not reconstructed in that OOF score-only point.

The complete deployed policy's full-fit Twin diagnostic is 97.8% safe steps,
95.6% row-weighted trajectory pass, and 49.2% proxy saving. This is deliberately
labelled in-sample and is not the headline evaluation.

On 1,000 real Viktor trajectories, the aggregate-only routing diagnostic is:

| Tier | Count | Share |
|---|---:|---:|
| Economical | 268 | 26.8% |
| Balanced | 482 | 48.2% |
| Strongest | 250 | 25.0% |

Viktor has no counterfactual quality labels, so this distribution is not a
quality result. The learned second opinion chose economical for 988/1,000
Viktor trajectories, confirming domain shift; it is therefore never allowed to
downgrade the deterministic route.

## Known weakness

Leave-one-benchmark-out safety for the optimized deterministic policy was
81.8% overall and only 54.8% on unseen SWE-bench. The guards reduce that known
failure mode at additional cost, but their true Viktor quality effect cannot be
identified from the logged export alone. The frontier sweep is also exploratory
threshold sensitivity on the same OOF predictions, not fresh validation of a
threshold chosen after inspecting the curve.
