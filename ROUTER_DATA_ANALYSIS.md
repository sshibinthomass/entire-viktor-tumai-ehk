# Router Dataset Audit

This audit uses the 1,000 requests in the real chunk and excludes the 153 requests
created by `scripts/make_synthetic_sample.py`. No raw challenge text is reproduced.

## What changed

- The old loader hashed only the first 2,000 characters of the first user message.
  Long memory prefixes collided and produced 19 false mixed-model groups. Hashing
  the complete opening system plus first user message produces 1,000 real groups,
  all singletons and no mixed-model violations.
- The old extractor scored the complete user envelope. Its median length was
  9,995.5 characters. Active-task slicing reduces the median scored text to 2,064
  characters (median 21.56% of the envelope).
- The generated synthetic sample was mixed into earlier baseline runs. It is now
  recognized by its exact generator marker and excluded from comparison/audit.

## Current cost-only comparison

| Policy | Estimated input cost | Change vs logged |
|---|---:|---:|
| Logged route | $146.32 | — |
| Starter baseline | $112.10 | -23.4% |
| Deterministic v2 | $64.99 | -55.6% |

Deterministic v2 is 42.0% cheaper than the starter baseline under the same assumed,
cache-aware, input-only pricing. Its routes are 331 economical, 623 balanced, and
46 strongest.

This does **not** establish that v2 is better: the export contains no final output,
usage, acceptance, success, or counterfactual quality labels. A policy can always
look cheaper by sending everything to the cheapest model.

## Signals that are weak or misleading here

| Signal | Prevalence | Historical tier observation | Why it is unsafe as evidence |
|---|---:|---|---|
| Multi-write/chained tools | 75.0% | Mean assumed logged tier 1.237 when present vs 1.244 absent | Almost no association; it describes the agent runtime more than model need. |
| History over 30 items | 28.7% | 1.227 present vs 1.244 absent | Long history may be memory/tool residue; it is context load, not reasoning difficulty. |
| Context >=24k estimated tokens | 29.9% | 1.090 present vs 1.302 absent | The direction is opposite the intuitive claim; likely selection/confounding, not proof that long context needs a smaller model. |
| Testing/verification | 15.4% | 1.221 present vs 1.242 absent | No positive historical association. Tests can make a task easier to verify even if execution is longer. |
| Multi-file/stateful | 41.5% | 1.169 present vs 1.289 absent | Strongly overlaps cross-system detection and does not imply a larger historical model. |
| High-reasoning intent | 25.9% | 1.324 present vs 1.209 absent | Direction is plausible but small; logged selection is still not a success label. |

The strongest redundancy is between multi-file/stateful and cross-system flags
(`phi = 0.740`). Long context and long history also overlap (`phi = 0.609`). Adding
both signals independently without caps would double-count the same underlying fact.

## Dataset limitations that affect optimization

1. **All 1,000 real reconstructed groups are singletons.** There is prior assistant
   and tool history embedded in nearly every request, but no exact next request in
   this chunk to reveal the current call's output. The promised recovered-output
   method cannot be applied to these real rows as linked request pairs.
2. **Logged model is treatment, not truth.** Agreement with the historical route
   measures imitation of an unknown policy, not quality or correct routing.
3. **No counterfactuals.** We do not observe how another model would answer the same
   request. Individual route quality requires replay or judge-scored multi-model
   outputs.
4. **No measured tokens or output cost.** Characters/4 and assumed input prices can
   distort savings, particularly across model families and long contexts.
5. **Tier order is assumed.** `fable/sonnet/opus` and especially `sol/terra/luna`
   must be confirmed by organizers or empirical replay. Default GPT prices are equal,
   so the current GPT cost comparison cannot distinguish those routes.
6. **Regex semantics remain brittle.** Negation handling and active-task slicing fix
   obvious cases, but quoted text, conditional actions, unusual envelopes, and domain
   synonyms can still be misclassified.
7. **Risk is not model complexity.** Security, destructive, or high-stakes work needs
   policy gates, permissions, review, and validation; routing upward alone is not a
   safety control.
8. **Baseline leakage depends on future data.** The starter policy sums the completed
   trajectory. Every real group currently has one request, but the policy would use
   unavailable future size if later chunks contain real multi-call trajectories.

## What to optimize next

Do not tune weights against cost or historical-model agreement alone. First collect
a small stratified replay set across score bands and candidate models. Grade it with
deterministic task checks plus blind human/judge review, calibrate an adequacy
probability per tier, and choose the cheapest eligible tier whose lower confidence
bound meets the quality target. Keep this dataset audit as the honest limitation of
the current cost frontier.

Machine-readable audit: `results/router_signal_analysis.json`.
