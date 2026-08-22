# Deterministic Router v2

The router assigns a reproducible **0–100 routing-demand score** without calling
another model. It is a heuristic rank, not a probability that a model will
succeed. Policy version: `complexity-v2-simple`.

Implementation: `scripts/deterministic_router.py`  
Metric extraction: `scripts/extract_router_metrics.py`

## Scorecard

Only five components contribute points. Their caps sum to 100.

| Component | Maximum | Exact scoring |
|---|---:|---|
| Context | 20 | Estimated input tokens: `<8k` = 0, `8k–23,999` = 5, `24k–63,999` = 10, `>=64k` = 15. Add 5 only when history has more than 30 items. |
| Reasoning | 35 | One intent: extract/classify = 0, rewrite/format = 5, summarize = 5, compare = 10, analyze = 15, research/synthesize = 20, plan/design = 20, debug/prove/diagnose = 30. Add 5 once if the task explicitly contains trade-offs, uncertainty/scenarios, or ambiguity/contradictions. |
| Execution | 15 | One action: no tool = 0, one read = 0, search/multiple reads = 5, one local write = 5, multi-write/chained tools = 10. Add 5 if testing or verification is explicitly required. |
| Scope | 20 | Choose one band: single object = 0; multiple objects/deliverables = 5; cross-system dependency or at least four objects = 10; multi-file modification or stateful coordination = 20. |
| Special requirements | 10 | Count strict output schema, required citations, compatibility/preservation, mutually consistent outputs, exact acceptance criteria, visual reasoning, and layout sensitivity. None = 0, one = 5, two or more = 10. |

Tier mapping:

| Score | Route |
|---:|---|
| 0–34 | Economical |
| 35–74 | Balanced |
| 75–100 | Strongest |

The tier-to-model mapping is configuration. Anonymized model names do not prove
their size, quality, or price order.

## Gates outside the score

Capability filters remove models that lack the required context window, image,
tool, structured-output, layout, language, or data-handling support.

Destructive/public actions, high-stakes domains, and security/authentication tasks
have a minimum score of 35 (balanced). This is a review floor, not a claim that a
larger model makes an unsafe action safe. Multi-file modification with required
verification also has a balanced floor.

Mid-trajectory failure handling remains separate. A failure score of at least 16
recommends one tier of escalation; environmental errors do not count by themselves.
A mid-task downgrade is disabled because switching models loses the shared cache.

## Active-task extraction

Real requests place long memories and channel instructions inside the first user
message. The extractor scores:

1. Text after the final `</system>` when present (usually the live event).
2. Otherwise the final `# === Thread info ===` block (usually the scheduled task).
3. Otherwise the full user text as a conservative fallback.

Negated actions such as “do not delete” are not treated as intended actions. An
input JSON object is not treated as a strict JSON output requirement.

## Deliberately excluded from the score

- Available tool count and tool-schema size: capability/overhead, not difficulty.
- Bullet and numbered-step count: 85.3% of the original real-task extraction had
  a complex-step flag, mostly because operational prompts are checklists.
- Raw occurrences of `must`, `only`, and `never`: memories and safety rules make
  these common without proving model difficulty.
- Code, URLs, file paths, and placeholder counts by themselves.
- Logged model, later calls, historical errors, retries, recovered outputs, or
  final trajectory length: these are outcome leakage for an initial router.
- Risk as additive “complexity”: it is enforced as a separate floor and safety gate.

All raw metrics remain in telemetry for later calibration, even when they do not
affect this score.

## Run and verify

```bash
python scripts/deterministic_router.py --example
python scripts/deterministic_router.py request_metrics.json --pretty
python scripts/deterministic_router.py --self-test
python scripts/compare_routers.py export/
python scripts/analyze_router_signals.py export/
```

Token counts use serialized characters divided by four and are estimates, not
measured usage. Cost comparisons use assumed input prices, are cache-aware, and
exclude output cost.
