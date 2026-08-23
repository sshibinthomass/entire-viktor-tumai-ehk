# Held-out test on `trajectories_v1_02` (chunk 02)

A second data drop arrived after the solution was frozen. This is the write-up of
running it — the one test the solution was built to take: **1,000 tasks nobody in
this repo has seen, routed and graded with transforms fitted on chunk 01, zero
refitting.**

Headline: **the frozen router transfers without loss** — 67.8% exact / 97.2%
adjacent agreement with the frozen evaluator, 83.1% of tasks served, 45.0%
estimated saving vs always-top-tier. That is *slightly better* than the fit
chunk's own in-sample numbers (66.3% / 96.9% / 82.9%), and the distribution
check says why: chunk 02 and chunk 01 are the same population.

What did **not** fully replicate: the under-routing cost claim. The retry-streak
half holds (+1.48 streak, CI [+0.24, +2.77]); the error-rate half flips sign and
loses significance (−0.041, CI [−0.097, +0.013]). Details and reading below.

Artifacts: `results/heldout_chunk02*.json` (Mode A, frozen),
`results/heldout_shift_chunk02.json` (distribution check),
`results/chunk02/` (Mode B, full refit on chunk 02).

## 1. What arrived

```bash
curl -L -o trajectories_v1_02.jsonl.tar.gz https://ref.viktor.com/tumai-data-02
mkdir -p export && tar xzf trajectories_v1_02.jsonl.tar.gz -C export/
```

30.1 MB archive (redirects to the same `viktor-tumai` R2 bucket as part 01) →
`export/trajectories_v1_02.jsonl`, 100.6 MB, **1,000 lines**, plus a `LICENSE`
identical to part 01's (challenge use only, no redistribution — the file stays
in gitignored `export/`).

Schema is unchanged: every line has exactly `model`, `input`, `tools` — no
`output`, no `usage`, no ids, no labels. Redaction placeholders behave as before.

**Shape (checked, not assumed):** 1,000 requests reconstruct to 1,000
trajectories of one call each, and each of those calls is deep — median 18 input
items (p25 11, p75 30, max 461), 998/1,000 contain `function_call` items. So the
chunk ships **one sampled (late) call per task**, exactly like chunk 01; chunk 00
is the odd one out, shipping whole trajectories (153 requests → 25 tasks).

That matters twice, and both times it is fine:

- **Routing features are unaffected.** `router/features.py` reads only the system
  prompt, the first user message and the tool definitions. Those items are a
  *prefix* of every later call's input, so features computed from a deep call are
  byte-identical to what a router would have seen at dispatch time.
- **Effort metrics are unaffected.** `evaluator/metrics.py` counts inside the
  deepest call's input and recovers `n_llm_calls` from the item structure
  (contiguous runs of model-emitted items), not from how many lines were logged.
  Only `n_logged_calls` is 1 by construction — it is display-only, never graded.

**It really is held out.** Task fingerprints (system + full first user text) and
first-user texts overlap chunk 01 in **0** cases; 1 of 927 system prompts is
shared (harness reuse, not task reuse).

**Model mix** is stable (requests per anonymized id):

| id | chunk 01 | chunk 02 |
|---|---|---|
| claude-opus-5 | 331 | 306 |
| claude-sonnet-5 | 281 | 269 |
| gpt-5.6-terra | 113 | 159 |
| gpt-5.6-sol | 112 | 97 |
| claude-opus-4-8 | 69 | 80 |
| claude-fable-5 | 71 | 62 |
| gpt-5.6-luna | 20 | 25 |
| claude-opus-4-6 | 2 | 2 |
| claude-sonnet-4-6 | 1 | 0 |

No new model id appeared, so the inferred tier map covers the chunk.

## 2. Is it the same population? (`scripts/heldout_shift.py`)

New script. It compares the fit chunk and a held-out chunk on the two things the
solution consumes — the 42 routing features and the 10 graded evaluator metrics —
reporting fit/new medians, the two-sample KS statistic, and the mean ECDF rank of
the new values *under the fit distribution* (0.5 = aligned). Ranks are what the
frozen transforms actually apply, so that last column is the shift that could
move tiers.

```bash
python scripts/heldout_shift.py export_linked/trajectories_v1_02.jsonl \
    --fit export_linked/trajectories_v1_01.jsonl
```

Result: **no detectable shift.** 0 of 42 routing features move by more than 0.15
rank; median |shift| is 0.006. The largest KS statistic anywhere (features or
metrics) is 0.062, against a 5%-level critical value of ≈0.061 for n = 1000 vs
1000 — i.e. at the noise floor, not above it. Sampling shape matches too (100%
single-call trajectories on both sides), so the effort counters are comparable.

This is the honest context for section 3: the frozen numbers transfer because the
new chunk is drawn from the same distribution, **not** because the router is
robust to shift. Shift robustness is untested — chunk 00 (n=25, differently
sampled) remains the only shifted probe, and there the frozen evaluator graded
everything Tier 1 (`results/heldout_chunk00.json`).

## 3. Mode A — frozen router, zero refitting

```bash
python scripts/enrich_dataset.py export/ export_linked/
python freeze_router.py export_linked/trajectories_v1_01.jsonl \
    --out results/frozen_chunk01.pkl
python apply_frozen.py export_linked/trajectories_v1_02.jsonl \
    --frozen results/frozen_chunk01.pkl --out results/heldout_chunk02.json
```

Refreezing matters: a `frozen_*.pkl` produced before commit `5b1a11c` embeds the
old tier prices, and `apply_frozen.py` prices from the artifact. Everything below
uses a pickle refit on chunk 01 with the shipped `TIER_PRICES`.

| method | exact | adjacent | served | served (token-wtd) | under | est. cost | saved |
|---|---|---|---|---|---|---|---|
| **blend** (default) | **67.8%** | **97.2%** | 83.1% | 74.4% | 16.9% | $102.46 | **45.0%** |
| score (heuristic only) | 53.7% | 91.8% | 79.9% | 62.6% | 20.1% | $94.13 | 49.4% |
| ml-τ | 56.2% | 95.7% | 94.6% | 91.6% | 5.4% | $132.53 | 28.8% |
| ml-λ | 55.0% | 95.3% | 87.0% | 70.4% | 13.0% | $95.11 | 48.9% |

n = 1000, all-Tier-3 reference $186.18, Spearman(router score, evaluator score)
= 0.681. Tier mix under blend: T1 583 / T2 297 / T3 120. Confusion (rows =
router tier, cols = evaluator difficulty):

```
        D1   D2   D3
T1     462  110   11
T2      95  154   48
T3      17   41   62
```

The method ranking is the same one the fit chunk produced, including the shape of
the trade-off: ml-τ buys the highest served share (94.6%) by spending most of the
saving, the heuristic alone saves the most and agrees the least, blend sits where
the deck claims it does.

The single-task inference path works on raw (un-enriched) chunk-02 lines too:
`python apply_frozen.py --one request.json` → `Tier 2  P(D<=1)=0.345
P(D<=2)=0.932`.

## 4. Mode B — full refit on chunk 02

Everything recomputed *on* chunk 02 (`results/chunk02/`), which answers a
different question: would the same method, fitted from scratch on this chunk,
reach the same place?

| | fit chunk 01 (n=1000) | held-out chunk 02, refit (n=1000) | held-out chunk 02, frozen |
|---|---|---|---|
| exact | 66.3% | 65.4% | 67.8% |
| adjacent | 96.9% | 95.8% | 97.2% |
| Spearman | 0.697 | 0.656 | 0.681 |
| served | 82.9% | 82.1% | 83.1% |
| est. saving | 39.9% | 42.0% | 45.0% |

(The chunk-01 column is a fresh run at the shipped prices —
`results/chunk02/fit_chunk01_comparison_shipped_prices.json` — not the committed
`results/comparison.json`; see the caveat in section 6.)

**Tier map replicates.** `infer_tiers.py` on chunk 02 reproduces the committed
tier assignment for all 8 shared ids, and the revealed-preference null result
(served-difficulty flat across models → the logged dispatcher was not
difficulty-aware) holds again:

| id | chunk 01 struggle-z → tier | chunk 02 struggle-z → tier |
|---|---|---|
| claude-sonnet-5 | −0.238 → T3 | −0.277 → T3 |
| claude-opus-5 | −0.178 → T3 | −0.142 → T3 |
| claude-opus-4-8 | −0.135 → T3 | −0.287 → T3 |
| claude-fable-5 | −0.113 → T2 | −0.085 → T2 |
| claude-opus-4-6 (n=2) | +0.346 → T2 | +0.565 → T2 |
| gpt-5.6-sol | +0.486 → T2 | +0.308 → T2 |
| gpt-5.6-terra | +0.568 → T1 | +0.570 → T1 |
| gpt-5.6-luna | +0.902 → T1 | +0.990 → T1 |

Only within-tier order moves (opus-4-8 ↔ sonnet-5 inside T3; sol ↔ the n=2
opus-4-6 inside T2, which the script already flags as low-confidence). Tier
buckets: identical.

**The under-routing cost claim only half replicates.** Matched difficulty
quintiles, tier map fit on the even half and validated on the odd half
(`results/chunk02/matching_check.json`, 500 validation rows):

| interaction (low tier × hard task) | chunk 00+01 | chunk 02 |
|---|---|---|
| extra error rate | **+0.067**, CI [+0.025, +0.108], significant | **−0.041**, CI [−0.097, +0.013], not significant |
| extra retry streak | **+2.73**, CI [+1.56, +3.99], significant | **+1.48**, CI [+0.24, +2.77], significant |

Read it as: *under-routing costs retries* survives a fresh sample at roughly half
the effect size; *under-routing costs errors* does not. The error-rate DiD on
chunk 02 is driven by the easy-task cell (easy tasks on low-tier models show a
10.1% error rate here vs 6.0% in chunk 00+01), which inverts the difference of
differences. Since the error-rate signal is also one of the two inputs to the
struggle ranking that produces the tier map, the honest statement after this drop
is the weaker one: the served-vs-cost penalty is supported by retry behaviour and
is *not* established on error rate.

**Cache trap: the measured layer is not computable on this chunk.** With no
trajectory carrying ≥2 logged calls there is no intra-task call pair whose prefix
overlap could be measured, so `cache_trap.py` now reports the layer unavailable
instead of crashing (it used to die in `np.quantile` on an empty array). The
modeled layer, which needs only per-trajectory metrics, replicates closely: a
$112.94 cache-reset penalty against a $152.68 input-only all-Tier-3 budget =
**74.0% of it** (fit chunks: 72.3%). Per-task routing still pays zero resets by
construction.

## 5. Code changes this test forced

| file | change |
|---|---|
| `scripts/heldout_shift.py` | new — the section 2 distribution check |
| `cache_trap.py` | measured layer is now conditional: a chunk with no multi-call trajectory reports `measured_available: false` with a reason instead of an `IndexError`; the modeled layer still runs |
| `matching_check.py` | the PARTIALLY-SUPPORTED verdict used to hard-code "directionally positive" for the non-significant metric; on chunk 02 that metric is negative, so the verdict now names the actual sign (and says so when the other metric is significant with the opposite sign) |
| `run_pipeline.py` | `tier3_price_sensitivity.note` still quoted the pre-`5b1a11c` Tier-3 row ($5/$0.5/$25); it now names the shipped assumption |

Both analysis-script changes were regression-checked against the committed
chunk 00+01 artifacts: `matching_check.py` reproduces `results/matching_check.json`
byte-for-byte, and `cache_trap.py` reproduces the measured layer's structure and
switch counts (25 switches, per-call greedy 36) — its dollar figures differ only
because `TIER_PRICES` changed in `5b1a11c`.

## 6. Caveats that carry over

- **Prices are an assumption** (anonymized ids). Every dollar and saving figure
  here uses the shipped `TIER_PRICES`; the fable-priced Tier-3 sensitivity is in
  `results/chunk02/comparison.json` (42.0% → 52.9%).
- **Tokens are chars/4 estimates.** No `usage` field exists in the export.
- **Agreement is agreement, not quality.** Both sides are constructed: the router
  score and the evaluator's effort-based difficulty. There are no human labels
  and no observed outputs for the final call of any trajectory.
- **Same-distribution transfer only** (section 2). Nothing here shows robustness
  to a genuinely shifted chunk.
- **The committed chunk-01 artifacts predate the pricing commit.** `5b1a11c`
  changed `TIER_PRICES` without regenerating them, so the shipped
  `results/comparison.json` (45.5% saved, n=1025) and the figures quoted from it
  in `docs/JUDGES.md` are at the old Tier-3 row. At the shipped prices the same
  pipeline gives 39.9% on chunk 01. Regenerating the full artifact chain
  (`experiments.py` → `exp_text.py` → `exp_final.py` → `tune_router.py` →
  `sweep_defaults.py` → `build_findings.py` → dashboard) under the new prices is
  a separate job; it is not done on this branch.

## 7. Reproducing this exactly

```bash
# 1. data (never committed; export/ and *.tar.gz are gitignored)
curl -L -o trajectories_v1_02.jsonl.tar.gz https://ref.viktor.com/tumai-data-02
mkdir -p export && tar xzf trajectories_v1_02.jsonl.tar.gz -C export/
python scripts/load_trajectories.py export/
python scripts/enrich_dataset.py export/ export_linked/

# 2. is it the same population?
python scripts/heldout_shift.py export_linked/trajectories_v1_02.jsonl \
    --fit /path/to/export_linked/trajectories_v1_01.jsonl \
    --out results/heldout_shift_chunk02.json

# 3. Mode A — frozen, zero refitting (the honest test)
python freeze_router.py /path/to/export_linked/trajectories_v1_01.jsonl \
    --out results/frozen_chunk01.pkl
python apply_frozen.py export_linked/trajectories_v1_02.jsonl \
    --frozen results/frozen_chunk01.pkl --out results/heldout_chunk02.json
for m in score ml-tau ml-lambda; do
  python apply_frozen.py export_linked/trajectories_v1_02.jsonl \
      --frozen results/frozen_chunk01.pkl --method $m \
      --out results/heldout_chunk02_$m.json
done

# 4. Mode B — full refit on chunk 02 (writes results/ + dashboard/data.js;
#    the chunk-02 copies of the outputs are committed under results/chunk02/)
python run_pipeline.py export_linked/trajectories_v1_02.jsonl
python infer_tiers.py && python matching_check.py && python cache_trap.py
```

Mode B overwrites the committed chunk-01 artifacts in place. On this branch they
were copied to `results/chunk02/` and the tracked files restored with
`git checkout -- results dashboard/data.js`; re-running Mode B twice produced
byte-identical `tiers.jsonl`, `evaluator_metrics.jsonl` and
`router_features.jsonl`, so the pipeline is deterministic on fixed input.
