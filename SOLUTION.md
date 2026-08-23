# Solution: Tier Router + Trajectory Evaluator

Two independent systems, judged against each other, with a live lab to tune every
hyperparameter. Built on the full enriched export (`export_linked/`, both chunks:
1,153 requests → **1,025 reconstructed trajectories**; chunk 01 is ~1 logged call
per task, chunk 00 carries the 25 genuine multi-call trajectories).

```
                 routing time                        hindsight
        ┌───────────────────────────┐      ┌────────────────────────────┐
task ──►│ PART 1 · ROUTER           │      │ PART 2 · EVALUATOR         │◄── full
        │ first user msg + system   │      │ tool calls, LLM calls,     │    trajectory
        │ prompt + tools ONLY       │      │ reasoning tokens, errors…  │
        │ → Tier 1 / 2 / 3          │      │ → Difficulty 1 / 2 / 3    │
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

## Trajectory reconstruction is a checked invariant, not an assumption

The export has no trajectory ids. `scripts/load_trajectories.py` groups requests by
a hash of the **system prompt + full first user text** and then a **nesting
validator** asserts each call's input is an item-level prefix of the next, splitting
any group that fails. This matters more than it sounds: an earlier key (first 2,000
chars of the user text only) silently merged *distinct* tasks that share an opening
template — 19 of its 24 "multi-call trajectories" mixed models, which is exactly the
premise violation the loader is supposed to flag. Under the fixed key, all 25 real
multi-call trajectories nest perfectly and every one ran on a single model — the
one-model-per-trajectory premise holds *exactly* once reconstruction is right.

## Part 1 — Router (`router/`)

`router/features.py` extracts ~42 routing-time features from exactly three sources:
the system prompt, the first user message, and the tool definitions. They are
rank-transformed with the repo's **one canonical transform** (`rank_utils.py`,
right-ECDF, fit per CV fold on train rows; the dashboard ports the same semantics
to JS, so lab and benchmarks agree to the digit) and grouped into four signed
groups — **the signs are the routing insight**, measured against observed effort:

| group | sign | what it holds | why |
|---|---|---|---|
| `ask` | + | entity refs, questions, action verbs, coordination markers, URLs, length, attachments, media | a dense ask is hard (`usr_n_pii_refs` alone: ρ ≈ +0.39) |
| `harness` | + | system-prompt sections, feature flags, skills, subagent tool | a heavily configured workspace hosts complex work |
| `breadth` | − | number of tools, tools tokens, background/memory tools | broad generic toolsets go with quick conversational turns (ρ ≈ −0.17) |
| `midthread` | − | truncated context, auto-read blocks, thread-activity trigger | mid-thread nudges mean most work already happened (ρ ≈ −0.19) |

`router/ml.py` adds the supervised head: **word(1–2) + char_wb(3–5) TF-IDF of the
first user message, concatenated with the numeric rank features, into cumulative
ordinal logistic regression** — trained on **evaluator labels** (never the logged
model id), producing per-task sufficiency probabilities P(D≤1), P(D≤2). All shipped
probabilities are **out-of-fold** under GroupKFold(5) on the workspace fingerprint
(710 distinct workspaces; the fingerprint joins parts-list system content — an
earlier bug collapsed 76% of rows into one group and degenerated the folds; every
number below postdates that fix).

Three dispatch rules turn probabilities into tiers (all sweeps committed —
`sweep_defaults.py` → `results/sweeps.json`):

- **blend + cuts** (production default) — α·rank(ML difficulty) + (1−α)·heuristic,
  cut at percentiles. α = 0.85; the exact-agreement surface is flat between
  α 0.75–1.0 (65–66%), so the default sits mid-plateau rather than on a lucky peak.
- **τ-sufficiency** — cheapest tier with P(D≤t) ≥ τ. Size-blind; the cheapest way
  to a served-% target.
- **λ-Bayes** — argmin_t ĉost(t) + λ·P(D>t), where **ĉost is a routing-time
  prediction** (OOF regressors on the same routing features). The earlier version
  plugged each task's *realized* cost into this rule — unknowable at dispatch, so
  those numbers were an oracle upper bound, not a router. Both variants are in the
  benchmark, labeled; the oracle rows are excluded from winner selection.

**Deployment path (`freeze_router.py` / `apply_frozen.py`):** the interactive
pipeline is transductive by design (ranks and cuts recomputed per batch — right for
analysis, not a router). The freeze step serializes every learned transform
(feature ECDFs, TF-IDF vocabularies, logit coefficients, fixed cut values, cost
regressors, and the evaluator yardstick) so a held-out chunk — or a **single task**
— is routed with zero refitting. Demonstrated by freezing on chunk 01 and applying
to chunk 00 (`results/heldout_chunk00.json`; n=25 and heavily shifted — a mechanism
demo, not a headline; the real held-out run awaits chunk 02).

## Part 2 — Evaluator (`evaluator/`)

`evaluator/metrics.py` counts effort inside the deepest call of each trajectory:
LLM responses, tool calls, tool errors, model-generated tokens, reasoning tokens,
tool-output tokens, context size, distinct tools, max same-tool retry streak
(resets on user/assistant turns), user turns, logged-call count. Tool errors use
word-boundary matching with negation handling ("12 passed, 0 failed" and
`"errors": []` no longer count) over the head *and tail* of each output
(tracebacks end outputs); spot-check with `scripts/check_error_markers.py` —
reviewed samples are dominated by genuine tracebacks/timeouts with residual noise
from code listings that mention errors. Image placeholders are counted at a stated
1,000 tokens (the redacted URL is ~6).

`evaluator/difficulty.py` grades rule-based difficulty: percentile-rank each metric
(canonical ECDF; constant metrics rank a neutral 0.5), weighted mean, cut at
p55/p85, hard overrides to D3 (≥5 tool errors or ≥40 tool calls). An optional
**per-tier residualization** (`run_pipeline.py --residualize`) subtracts each
inferred model tier's median rank from the two policy-sensitive metrics — the
correction for "weaker models retry more" that the matched check quantifies.

## Benchmarks (all OOF, GroupKFold(workspace), frozen evaluator, cache-aware costs)

Reference points: oracle (tier = difficulty) serves 100% at 66.2% of all-Tier-3
cost; always-T2 serves 84.5% at 40% cost. Full table in
`results/tuning_report.json`; the deployable-vs-oracle λ distinction is explicit:

| method | frontier AUC | served @50% budget | cost @90% served | weighted AUC |
|---|---|---|---|---|
| heuristic (score/default) | 0.856 | 0.811 | 0.739 | 0.707 |
| heuristic, CV-tuned weights | 0.850 | 0.800 | 0.711 | 0.682 |
| k-means / GMM | 0.48 / 0.55 | — | — | 0.37 / 0.40 |
| ordlog + τ-sufficiency | 0.876 | 0.832 | 0.644 | **0.714** |
| ordlog + λ-Bayes (predicted cost) | 0.881 | 0.855 | 0.608 | 0.641 |
| **TF-IDF+logreg + λ-Bayes (predicted cost)** | **0.892** | 0.845 | **0.590** | 0.690 |
| *TF-IDF+logreg + λ-Bayes (oracle cost)* | *0.913* | *0.891* | *0.574* | *0.640* |
| random baseline (100 perms) | 0.849 [0.839, 0.862] | 0.829 | 0.712 | 0.642 |

Honest reading of that table:

1. **Raw AUC flatters everyone.** 55% of tasks are D1 and any tier serves them, so
   even random routing scores 0.849 ± 0.011. The separation lives at the
   quality-critical end — cost@90%-served: random needs 71% of the all-T3 budget,
   the deployable λ router 59% — and in the token-weighted metric.
2. **Oracle vs deployable λ is a measured 2.1-point AUC gap** (0.913 → 0.892):
   that is the price of not knowing each task's size in advance, and the reason
   the oracle rows are labeled, not headlined.
3. **Group-weight tuning still overfits** (0.850 OOF vs 0.856 defaults; in-search
   0.872 → the gap is measured selection bias). The hand-set defaults stay.
4. **Clustering is dominated** — it finds workspace structure, not difficulty.
5. **Robustness:** across 30 perturbed evaluators (weights ×0.5–2, cuts ±5pts) the
   winner's served@50%-budget moves to 0.838 ± 0.015 (min 0.81), and the perturbed
   evaluators agree with the frozen labels **94.9%** on average — that agreement
   bounds any router; nothing above it is measurable.

### The quotable agreement numbers (train-calibrated dispatch — no test-label leakage)

Dispatch cuts are calibrated **inside each training fold** (`cut_dispatch_oof`);
the old pooled-marginal cuts used the test set's label mix and are now quoted only
as a labeled reference ("known deployment marginals"). Method ladder
(`experiments.py`, deployable exact / known-marginals reference):

| candidate | exact | balanced | D3 recall |
|---|---|---|---|
| ordlog, base numeric features | 57.9% / 58.0% | 50.9% | 41.9% |
| ordlog, + v2 lexical features | 60.5% / 61.1% | 54.2% | 47.5% |
| HistGradientBoosting | 62.0% / 62.6% | 55.0% | 44.4% |
| **word+char TF-IDF + numeric LR** | **66.2% / 66.0%** | **60.1%** | **53.1%** |
| nested stack | 64.8% / 65.5% | 57.8% | 49.4% |

**Honest final number (nested selection over the FULL 30-config grid, deployable
cuts): 66.0% exact, 59.6% balanced, 96.3% adjacent, D3 recall 52.5%, ρ 0.695**
(`results/final_validation.json`) — the inner CV re-picks the config inside every
training fold (the five folds picked FOUR different configs, stated as such), the
outer cuts come from train labels only, and the grid-selection bias is
*measured*, not asserted: the optimistic pooled-grid best is 67.1%, so selection
bias = **+1.2 pts**. The known-marginals reference (transductive cuts) is 66.4%.
The earlier "zero grid optimism / all five folds agreed" claims were wrong: the
old 8-config grid had been pre-filtered on the same folds, and the fold picks
never matched the artifact. Chance with these marginals is ~41%; always-T1 gets
55.0%; the evaluator-agreement ceiling is 94.9%.

Recommended operating points (defaults; cache-aware costs; from
`results/sweeps.json` + `results/comparison.json`):

- **Balanced (default):** blend α=0.85, cuts p55/p85 → **65.3% exact, 97.1%
  adjacent, Spearman 0.693, 82.2% served (68.8% token-weighted), 45.5% cost
  saved** — 53.6% saved if Tier 3 is priced at the fable-5 row instead of opus-5
  (tier-price sensitivity in `comparison.json`).
- **Quality-safe:** ML τ = 0.80 → **94.7% served at 72.5% of all-T3 cost**
  (89.4% token-weighted), 5.3% under-routed.
- **Max tasks-per-dollar:** λ-Bayes on predicted costs — best cost@90%-served
  (0.59) of any deployable method; its known weakness (starving token-heavy
  tasks) is visible in the weighted column and named on the deck.

## Deck-requirement closures

- **Inferred model tier order** (`infer_tiers.py` → `results/model_tiers.json`):
  within matched routing-difficulty quintiles, weaker models show more tool errors
  and longer retry streaks. Bootstrap-stable extremes: **claude-sonnet-5 /
  claude-opus-5 least struggle (T3, 99%/96% stable), gpt-5.6-luna most (T1, 94%)**;
  the middle band (fable-5 at 57%, opus-4-8 at 62%) is flagged as genuinely fuzzy —
  terciles force three ids per tier regardless of spread. **Null result worth
  naming:** served difficulty is flat across models — the logged dispatch was not
  difficulty-aware; that headroom is what the router exploits.
- **Matched cross-model check** (`matching_check.py`, circularity-guarded: tier map
  fit on the even half of trajectories, validated on the odd half only; both
  interactions bootstrapped, 5,000 resamples on unrounded arrays): on matched hard
  tasks, low-tier models show **+6.7pt error rate (95% CI [+2.5, +10.8]) and +2.7
  extra retry streak (95% CI [+1.6, +4.0]) — both significant.** Verdict:
  **SUPPORTED on both metrics — under-routing has a measurable cost.** (The earlier
  writeup called this "not significant" after testing only one of the two effects.)
- **The cache trap** (`cache_trap.py` → `results/cache_trap.json`), measured and
  modeled layers separated: on the 25 genuinely multi-call trajectories the
  per-call policies produce real switches (25–36) and naive costing overstates the
  greedy policy's savings by 35pts (48.5% → 13.2%). **MODELED** over all 1,025
  trajectories (three stated assumptions), a cheap-for-tool-loops policy pays
  ~$160 in cache resets against a **$221 all-Tier-3 input-only budget** — its
  naive savings claim goes *negative* once resets are priced. Per-task routing
  pays zero resets by construction. Dollar figures are input-only and labeled so;
  tier-price sensitivity included.
- **Policy comparison on the frontier** (dashboard + deck): the frontier is drawn
  against expected **random routing** (analytic expectation, per quality metric)
  and the **logged dispatch repriced under our inferred tier map** — graded
  *without* the error/streak metrics for that comparison, since those metrics also
  drove the tier inference (grading with them would mechanically depress logged
  served%). The tooltip carries a tercile-boundary sensitivity. We claim "better
  under our inferred tier map and fair grading", not "strict domination".
- **The 5-minute deck** (`dashboard/present.html`): seven timed slides plus a Q&A
  appendix — claim (0:20) · setup in three facts (1:00) · the frontier (2:00) · the
  decision rule (2:45) · the honest slide (3:45) · one known weakness and a week of
  work (4:20) · close (5:00). Each slide carries its time window and `t` starts a
  presenter timer that turns amber past the segment. Slides are deliberately
  text-minimal — headline sentence, numbers with denominators and intervals, one
  chart; the prose lives in speaker notes toggled with `n` and never shown to the
  room. Every number is computed live from `data.js`/`findings.js` (committed
  artifacts), including the bootstrap CIs (2,000 resamples of the 1,025 tasks)
  and the evaluator-perturbation band. The
  frontier draws the router as a curve against four labeled baselines — always-cheap
  (all T1), always-strong (all T3), the repriced logged dispatch, and **random at
  matched cost** — with error bars on the operating point and on random. The τ=0.80
  point is pinned; the replay does one sweep and parks on it ('p' pauses). The rule
  slide prints the dispatch rule and its fallbacks verbatim; it also shows the rule
  firing on one real trace (loaded from the gitignored local previews — verbatim
  dataset text is never committed).
- **Judge-model rescoring** (`judge_rescore.py`, starter idea 3): matched
  low-vs-high-tier call pairs are built from recovered replies (call i's output
  lives in call i+1's input) — 8 pairs exist at the current data size; scoring
  them needs the team's own API key and is the highest-ceiling next step for the
  off-policy special prize.

## Reproduction

```
scripts/enrich_dataset.py export/ export_linked/   (ids + nesting validation)
run_pipeline.py → experiments.py run → exp_text.py → exp_final.py
tune_router.py → sweep_defaults.py
infer_tiers.py → cache_trap.py → matching_check.py → build_findings.py
freeze_router.py  (→ apply_frozen.py for held-out chunks / single tasks)
```

A clean checkout works: dependencies are declared in `pyproject.toml`, no path is
hardcoded, `experiments.py` rebuilds its cache automatically, and the raw export
is never modified. Files that embed dataset text (`export*/`,
`dashboard/previews.js`, `results/exp_cache.npz`, `results/frozen_router.pkl`)
are gitignored — the dataset is challenge-use-only (see
`docs/LICENSE_CONTAINMENT.md`).

## Honesty notes (the known weaknesses)

- **Effort ≠ difficulty.** The evaluator measures observed effort; a task can be
  long-and-easy or short-and-hard. Effort is also shaped by the model that ran it —
  now *quantified* by the matched check (significant on both metrics) and
  correctable via `--residualize`; the default pipeline stays unresidualized so the
  headline and the correction are separately inspectable.
- **All token counts are chars/4 estimates** — the export has no `usage`. Context
  tokens include JSON overhead, generated tokens don't; images are counted at an
  assumed 1,000 tokens. Fine under ranking; stated wherever dollars are quoted.
- **Pricing is an assumption** (anonymized ids). Headline savings carry a
  tier-price sensitivity (opus-priced vs fable-priced Tier 3). Quote ratios, not
  dollars.
- **`reasoning_tokens` exist only for the gpt family** — its weight stays small on
  purpose; raise it and you grade providers, not tasks.
- **Sampling truncation.** ~1 logged call per task in chunk 01: the deepest logged
  call may be mid-task, so effort counts are lower bounds, and the cache-trap
  *measured* table is a floor (the modeled layer exists because of this, with its
  assumptions stated).
- **The evaluator ceiling:** perturbed evaluator configs agree 94.9% with the
  frozen one — router agreement above that is unmeasurable.

## Files

- `router/features.py`, `router/tiering.py`, `router/ml.py`, `rank_utils.py` — Part 1
- `evaluator/metrics.py`, `evaluator/difficulty.py` — Part 2
- `scripts/load_trajectories.py` (reconstruction + nesting validator),
  `scripts/enrich_dataset.py` (ids), `scripts/cost_model.py`
- `run_pipeline.py` — end-to-end; `tune_router.py` — frontier benchmark;
  `experiments.py` / `exp_text.py` / `exp_final.py` — method ladder + nested
  validation; `sweep_defaults.py` — the sweeps behind the defaults
- `infer_tiers.py`, `matching_check.py`, `cache_trap.py`, `judge_rescore.py` —
  off-policy analyses; `build_findings.py` — bundles them for the UI
- `freeze_router.py` / `apply_frozen.py` — the deployable router
- `dashboard/index.html` — the interactive lab; `dashboard/present.html` — the deck
