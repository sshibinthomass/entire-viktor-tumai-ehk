# Deterministic Router — Exact Formulas

Let `I(condition) = 1` when true, otherwise `0`.

## 1. Token estimate

```text
T = supplied estimated_input_tokens
    else floor(serialized_input_characters / 4)
```

## 2. Complexity score

```text
BaseScore = Context + Reasoning + Execution + Scope + Special
```

### Context: 0–20

```text
TokenPoints =  0  if T < 8,000
               5  if 8,000 <= T < 24,000
              10  if 24,000 <= T < 64,000
              15  if T >= 64,000

HistoryPoints = 5 * I(input_item_count > 30)

Context = min(20, TokenPoints + HistoryPoints)
```

### Reasoning: 0–35

Exactly one intent value is selected:

| Primary intent | Points |
|---|---:|
| Extract or classify | 0 |
| Rewrite or format | 5 |
| Summarize | 5 |
| Compare | 10 |
| Analyze | 15 |
| Research or synthesize | 20 |
| Plan or design | 20 |
| Debug, prove, or diagnose | 30 |

```text
ComplexModifier = 5 * I(
    has_tradeoffs
    OR has_uncertainty_or_scenarios
    OR has_contradiction_or_ambiguity
)

Reasoning = min(35, IntentPoints + ComplexModifier)
```

The modifier is added once even when several conditions are true.

### Execution: 0–15

Exactly one action value is selected:

| Expected action | Points |
|---|---:|
| No tool | 0 |
| One read-only operation | 0 |
| Search or multiple reads | 5 |
| One local write | 5 |
| Multiple writes or chained tools | 10 |

```text
VerificationPoints = 5 * I(requires_testing_or_verification)

Execution = min(15, ActionPoints + VerificationPoints)
```

### Scope: 0–20

Apply the first matching condition:

```text
Scope = 20  if is_multi_file_modification OR has_stateful_coordination
        10  else if has_cross_file_or_system_dependency
                    OR artifact_object_count >= 4
                    OR deliverable_count >= 4
         5  else if artifact_object_count >= 2
                    OR deliverable_count >= 2
         0  otherwise
```

### Special requirements: 0–10

```text
N = I(requires_strict_schema_or_machine_output)
  + I(requires_citations_or_source_traceability)
  + I(requires_compatibility_style_template_or_behavior_preservation)
  + I(requires_mutually_consistent_outputs)
  + I(has_exact_numerical_or_factual_acceptance_criteria)
  + I(requires_ocr_spatial_or_chart_reasoning)
  + I(requires_layout_sensitive_artifact)

Special =  0  if N = 0
           5  if N = 1
          10  if N >= 2
```

## 3. Score floors

```text
Floor = 35 if any condition is true:
    action_risk = destructive/public/deployment/permission-changing
    is_high_stakes_domain
    involves_security/secrets/authentication/permissions
    is_multi_file_modification AND requires_testing_or_verification

Floor = 0 otherwise

FinalScore = max(BaseScore, Floor)
```

## 4. Tier decision

```text
Economical  if FinalScore < 35
Balanced    if 35 <= FinalScore < 75
Strongest   if FinalScore >= 75
```

## 5. Capability filter

A model is eligible only if all are true:

```text
model_context_window >= T
supports_images             if input_image_count > 0
supports_tools              if expected_action != no_tool
supports_structured_output  if strict output is required
supports_layout_tasks       if layout work is required
supports_required_language
allows_all_required_data_tags
```

From eligible models with `model_tier >= target_tier`, select the minimum tuple:

```text
(tier_rank, uncached_input_price, model_id)
```

With only a configured tier-to-model map, return the model assigned to the target tier.

## 6. Mid-trajectory override

Each condition contributes once when present:

```text
FailureScore = 8  * I(malformed_tool_or_structured_response_count > 0)
             + 8  * I(repeated_failed_tool_and_arguments_count > 0)
             + 10 * I(explicit_user_correction_count > 0)
             + 8  * I(consecutive_model_attributable_failure_count >= 2)
             + 6  * I(failed_verification_after_claimed_completion_count > 0)
             + 6  * I(no_progress_two_call_window_count > 0)
```

```text
Capability failure                         -> CAPABILITY_CHANGE_REQUIRED
FailureScore >= 16 and below strongest     -> upgrade at least one tier
FailureScore >= 16 and already strongest   -> REVIEW_REQUIRED
FailureScore < 16                          -> KEEP current tier
```

Environmental tool errors add `0` points.

## 7. Active text extraction

```text
1. Use text after the final </system> when non-empty.
2. Otherwise use the final "# === Thread info ===" block.
3. Otherwise use the full user text.
```

Negated actions such as `do not delete` are excluded from intended-action detection.
