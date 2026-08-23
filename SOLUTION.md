# Solution: Tier Router + Trajectory Evaluator

Two independent systems, judged against each other, with a live lab to tune every
hyperparameter. Built on `export_linked/trajectories_v1_01.jsonl`
(1,000 requests → 953 reconstructed trajectories).

```
                 routing time                        hindsight
        ┌───────────────────────────┐      ┌────────────────────────────┐
task ──►│ PART 1 · ROUTER           │      │ PART 2 · EVALUATOR         │◄── full
        │ first user msg + system   │      │ tool calls, LLM calls,     │    trajectory
        │ prompt + tools ONLY       │      │ reasoning tokens, errors…  │
        │ → Tier 1 / 2 / 3          │      │ → Difficulty 1 / 2 / 3     │
        └────────────┬──────────────┘      └─────────────┬──────────────┘
                     └────────────► compare ◄────────────┘
                agreement · confusion · under-routing · cost–quality frontier
                             (dashboard/index.html)
```

The two parts read **disjoint slices of each trajectory**: the router sees only the
earliest logged call's opening (what a dispatcher would know), the evaluator counts
everything inside the deepest logged call (which embeds the full history). Neither
ever uses the logged `model` id as ground truth — the historical dispatch policy is
unknown and would leak provider bias, not difficulty.

## Part 1 — Router (`router/`)

`router/features.py` extracts ~29 routing-time features from exactly three sources:
the system prompt, the first user message, and the tool definitions. They are
percentile-ranked (robust to heavy-tailed token counts) and grouped into four signed
groups — **the signs are the routing insight**, measured against observed effort:

| group | sign | what it holds | why |
|---|---|---|---|
| `ask` | + | entity refs, questions, action verbs, coordination markers, URLs, length, attachments, media | a dense ask is hard (`usr_n_pii_refs` alone: ρ ≈ +0.37) |
| `harness` | + | system-prompt sections, feature flags, skills, subagent tool | a heavily configured workspace hosts complex work |
| `breadth` | − | number of tools, tools tokens, background/memory tools | broad generic toolsets go with quick conversational turns (ρ ≈ −0.21) |
| `midthread` | − | truncated context, auto-read blocks, thread-activity trigger | mid-thread nudges mean most work already happened (ρ ≈ −0.23) |

`router/tiering.py` turns the signed weighted group score into tiers, two
label-free ways:

- **score** — cut the composite at dataset percentiles (default p55 / p85)
- **kmeans** — k-means (k = 3) in the same weighted rank space, clusters ordered by
  mean composite (unsupervised structure discovery; sign is irrelevant to the
  clustering itself since reflections preserve distances)

`router/ml.py` adds the supervised head that won the two-round method ladder:
**word(1–2) + char_wb(3–5) TF-IDF of the first user message, concatenated with the
numeric rank features (23 group features + 19 v2 lexical/structural features),
into cumulative ordinal logistic regression** — trained on **evaluator labels**
(never the logged model id), producing per-task sufficiency probabilities
P(D≤1), P(D≤2). Three dispatch rules turn probabilities into tiers:

- **blend + cuts** (production default) — α·rank(ML difficulty) + (1−α)·heuristic,
  cut at percentiles. The balanced winner: improves BOTH the per-task and the
  token-weighted frontier over the heuristic.
- **τ-sufficiency** — cheapest tier with P(D≤t) ≥ τ. Size-blind; the cheapest way
  to 90 % served.
- **λ-Bayes** — argmin_t cost$(t) + λ·P(D>t). Best tasks-per-dollar; *known
  weakness:* it buys count-served by starving token-heavy tasks (watch the
  weighted-served tile).

## Part 2 — Evaluator (`evaluator/`)

`evaluator/metrics.py` counts effort inside the deepest call of each trajectory:
LLM responses (contiguous runs of model-emitted items + the final unlogged one),
tool calls, tool errors, model-generated tokens, reasoning tokens, tool-output
tokens, context size, distinct tools, max same-tool retry streak, user turns,
logged-call count.

`evaluator/difficulty.py` grades rule-based difficulty:

1. percentile-rank each metric across the dataset
2. difficulty score = weighted mean (defaults: tool calls .22, LLM calls .18,
   generated tokens .16, context .10, errors .10, …)
3. cut at percentiles (default p55 / p85) → **Difficulty 1/2/3**
4. hard overrides promote pathologies to D3 (≥5 tool errors, or ≥40 tool calls)

## Deep analysis & tuning (`tune_router.py`)

Everything below is **out-of-fold** under GroupKFold(5) on the workspace
fingerprint (random splits leak tenant identity), against the **frozen** default
evaluator, under **cache-aware** costs. Reference points: oracle (tier =
difficulty) serves 100 % at 64.6 % cost; always-T2 serves 85 % at 40 % cost.

| method | frontier AUC | served @50 % budget | cost @90 % served | weighted AUC |
|---|---|---|---|---|
| heuristic (score/default) | 0.861 | 0.818 | 0.730 | **0.715** |
| heuristic, CV-tuned weights | 0.843 | 0.802 | 0.766 | 0.691 |
| k-means / GMM | 0.45 / 0.58 | — | — | 0.36 / 0.40 |
| ordlog + τ-sufficiency | 0.869 | 0.867 | **0.577** | 0.682 |
| ordlog + λ-Bayes | 0.893 | 0.858 | 0.717 | 0.588 |
| TF-IDF+logreg + λ-Bayes | **0.903** | 0.865 | 0.642 | 0.613 |
| **blend α=0.5 + cuts** | 0.871 | 0.833 | 0.690 | **0.727** |
| random baseline | 0.842 | 0.794 | 0.771 | 0.675 |

Findings that drive the shipped defaults:

1. **Group-weight tuning overfits.** Per-fold optima disagree wildly and the OOF
   AUC lands *below* the hand-set defaults (0.843 vs 0.861; in-search 0.874 → the
   gap is measured selection bias). The defaults stay.
2. **Clustering is dominated** — it finds workspace structure, not difficulty.
3. **λ-Bayes games the count metric**: best per-task AUC, but weighted AUC
   collapses (0.59–0.61) because it starves the few token-heavy tasks that
   dominate cost. Legitimate if the objective is strictly tasks-per-dollar;
   reported with its weakness named.
4. **The blend (α = 0.5) is the balanced winner** — the only method that beats the
   heuristic on *both* the per-task and the token-weighted frontier, so it is the
   pipeline default. τ-sufficiency is the pick when a served-% target must be hit
   cheaply (90 % served at ~58 % of all-T3 cost vs 73 % for the heuristic).
5. **Robustness:** across 30 randomly perturbed evaluators (weights ×0.5–2, cuts
   ±5 pts) the winner's served@50 %budget moves 0.865 → 0.859 ± 0.018. Not an
   artifact of one weighting.

### Round 2: breaking the feature ceiling (`experiments.py`, `exp_text.py`, `exp_final.py`)

Diagnosis first: always-T1 alone scores 55 % exact (D1 is 55 % of labels), tasks
within 0.03 of a difficulty cut agree at coin-flip rates, and every numeric-only
learner (ridge, HGB-regressor, stacks) plateaued at 55–57 % — a feature ceiling,
not a model problem. Ruled out honestly: workspace priors (median 1 task per
workspace in this export — LOO signal is nil), natural-breaks cuts (changes the
task, not the quality), argmax dispatch (−6 pts: uncertainty collapses to T2),
boosted trees (overfit 953 rows: 50–56 %), nested stacking (58.6 %).

What broke the ceiling: **text**. The method ladder (all OOF, GroupKFold):

| candidate | exact | balanced | D3 recall |
|---|---|---|---|
| ordlog, base numeric features | 57.7 % | 50.2 % | 37.8 % |
| ordlog, + v2 lexical features | 58.6 % | 51.3 % | 39.2 % |
| LightGBM / XGBoost / CatBoost / mord | 50–58 % | 40–51 % | 19–38 % |
| TF-IDF(SVD-80) + LR | 61.9 % | 54.6 % | 42.7 % |
| **word+char TF-IDF + numeric, ordinal LR** | **62.4 %** | **55.1 %** | **43.4 %** |

**Honest final number (nested selection — the config is re-picked by inner CV
inside every training fold, so zero grid optimism): 61.4 % exact, 54.2 %
balanced, 95.6 % adjacent, D3 recall 43.4 %, ρ 0.577.** All five folds picked
the same config independently; measured selection bias ≈ 1 pt. Versus the
round-1 baseline: exact 56.0→61.4 %, balanced 48→54 %, confusion diagonal
378/106/50 → 400/123/62, two-tier misses 59 → 42. Chance with these marginals
is 41.5 %; two reasonable evaluator configurations agree ~93 % with each other,
which bounds any router.

Recommended operating points (dashboard defaults; cache-aware costs):

- **Balanced (default):** blend α=0.85 (ML head + heuristic), cuts p55/p85 →
  exact agreement 61.4 %, adjacent 95.8 %, Spearman 0.583, 80.9 % served, 66.2 %
  token-weighted, 48.6 % cost saved.
- **Quality-safe:** ML τ = 0.80 → 91.9 % served, 8.1 % under-routed, 36.7 % saved
  (77.0 % token-weighted).
- **Max tasks-per-dollar:** ML λ = $0.30 → 81.7 % served at 78.2 % saved
  (weighted served 42.9 % — the named sacrifice).

## Judging the router

`run_pipeline.py` joins both parts per trajectory and reports agreement,
confusion, under/over-routing, served (per-task and token-weighted) and cost
under both cost models; `results/comparison.json` holds the numbers,
`results/tuning_report.json` the full benchmark with every frontier.

The **cost–quality frontier** (headline artifact) sweeps the active method's own
knob and plots estimated cost against both served definitions — the dashboard
recomputes it live.

## The lab (`dashboard/index.html`)

Serve `dashboard/` statically (`python -m http.server -d dashboard`, or the
`dashboard` entry in `.claude/launch.json`) and open it. Everything recomputes
client-side in ~10 ms per change:

- router: 5 methods (blend / heuristic cuts / ML τ / ML λ / k-means), α, τ, λ,
  4 group weights, 2 tier cuts, k-means seed
- evaluator: 10 metric weights, 2 difficulty cuts, 2 override thresholds
- cost model (cache-aware vs naive) + pricing per tier ($/1M in, cached & out)
- live outputs: stat tiles, score histograms with cut lines, tier mix, confusion
  matrix, cost–quality frontier with the current operating point, router-vs-observed
  scatter (every task hoverable), per-feature signal chart, largest-disagreement
  table, CSV export

Data loads dynamically from `dashboard/data.js` — rerun `python run_pipeline.py`
(new export chunk, new features) and reload the page; no dashboard edits needed.

## Deck-requirement closures (round 3)

- **Inferred model tier order** (`infer_tiers.py` → `results/model_tiers.json`):
  within matched difficulty buckets, weaker models show more tool errors and
  longer retry streaks. Bootstrap-stable extremes: fable-5 / opus-5 top (94 %),
  luna bottom (100 %); fuzzy middle flagged honestly. Independent of the price
  sheet, yet reproduces its extremes. **Null result worth naming:** served
  difficulty is flat across models — the logged dispatch was not
  difficulty-aware; that headroom is what the router exploits.
- **The cache trap, demonstrated** (`cache_trap.py` → `results/cache_trap.json`):
  a per-call "cheap model for tool loops" policy claims 96 % savings under naive
  costing; priced with cache resets over full trajectories it pays ~$148 in
  resets alone — over half the ~$259 all-Tier-3 input budget. The measured
  multi-call subset is a floor (sampling hides switch points); per-task routing
  pays zero resets by construction.
- **Matched cross-model check** (`matching_check.py` →
  `results/matching_check.json`): low-tier models show +0.6 pt error rate and
  +1.33 retry streak on matched hard tasks (difference-in-differences).
  Directionally supportive, **not significant at n=953** — the "served"
  assumption stays an assumption, stated as such.
- **Policy comparison on the frontier** (dashboard + deck): our frontier is
  drawn against the two references the challenge illustrates — expected
  **random routing** (dashed, pure expectation, no RNG) and the **logged
  dispatch** (what actually ran, repriced via the inferred tiers), plus the
  always-top-tier dot. Result: the logged dispatch sits at 89.9 % served /
  69.2 % cost; at that same budget our τ-frontier serves **93.4 %**
  (token-weighted 80.4 % vs 75.4 %) — strict domination on both quality
  definitions.
- **The 5-minute defense deck** (`dashboard/present.html`): follows the
  make-presentation template (brand, fixed slide order, keys/fullscreen) but
  every number and chart is **computed live** from `data.js`/`findings.js` —
  including an auto-playing τ-sweep replay on the frontier slide. Team
  name/members are the only fill-in slots left.
- **Dashboard explainability layer**: every tile, slider, method, cost model,
  chart and finding now explains itself on hover (delegated tooltip layer);
  pipeline findings render as three cards below the live charts.

Reproduction order:
`run_pipeline.py` → `infer_tiers.py` → `cache_trap.py` → `matching_check.py`
→ `build_findings.py`; then open `dashboard/index.html` (lab) and
`dashboard/present.html` (deck) — both work statically, no server needed.

## Honesty notes (the known weaknesses)

- **Effort ≠ difficulty.** The evaluator measures observed effort; a task can be
  long-and-easy or short-and-hard. Effort is also shaped by the model that ran it
  (weaker models retry more), so the yardstick is partly policy-dependent.
- **All token counts are chars/4 estimates** — the export has no `usage`.
- **Pricing is an assumption** (anonymized model ids), and the cost estimate
  ignores prompt-cache discounts; a model switch resetting the cache would make
  naive per-call savings look better than they are. Quote relative comparisons,
  not absolute dollars.
- **`reasoning_tokens` exist only for the gpt family** — its default weight is
  small on purpose; raise it and you grade providers, not tasks.
- **Sampling truncation.** The deepest *logged* call can still be mid-task, so all
  effort counts are lower bounds; ~1 call per task means history size is noisy.

## Files

- `router/features.py`, `router/tiering.py`, `router/ml.py` — Part 1
- `evaluator/metrics.py`, `evaluator/difficulty.py` — Part 2
- `run_pipeline.py` — end-to-end; writes `results/router_features.jsonl`,
  `results/evaluator_metrics.jsonl`, `results/tiers.jsonl`,
  `results/comparison.json`, `dashboard/data.js`
- `tune_router.py` — round-1 frontier benchmark (`results/tuning_report.json`)
- `experiments.py`, `exp_text.py` — round-2 method ladder
  (`results/experiments_report.json`, `results/exp_text_report.json`)
- `exp_final.py` — nested honest validation (`results/final_validation.json`)
- `dashboard/index.html` — the interactive lab
