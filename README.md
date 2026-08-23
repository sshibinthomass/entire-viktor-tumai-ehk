# Viktor Challenge — Build the Router

Our solution for the **Viktor Challenge** at the TUM.ai hackathon (Munich, 22–23 Aug 2026):
a two-part **tier router + trajectory evaluator** with an honest off-policy evaluation.
The full writeup is in [SOLUTION.md](SOLUTION.md). **Judging this? Start with
[docs/JUDGES.md](docs/JUDGES.md)** — the 15-minute evaluation path, the
claim-to-artifact map, and how to run it on your own data. **Setting this up on a new
machine? Start with [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — a step-by-step
checklist with verification points and troubleshooting. License-containment
status is in [docs/LICENSE_CONTAINMENT.md](docs/LICENSE_CONTAINMENT.md).

## Reproduce from a clean checkout

(The condensed version — [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) has the same steps
with prerequisites, expected output at each stage, and a troubleshooting table.)

```bash
# 0. dependencies (numpy / scipy / scikit-learn — see pyproject.toml)
python -m venv .venv && .venv/Scripts/pip install -e .        # or: uv sync

# 1. dataset (challenge use only — NEVER commit it): extract the posted archives
#    into export/ ; no dataset yet? scripts/make_synthetic_sample.py fakes the shape
mkdir -p export && tar xzf trajectories_v1_01.jsonl.tar.gz -C export/

# 2. sanity-check + reconstruct trajectories (grouping key + nesting validator):
python scripts/load_trajectories.py export/

# 3. assign ids (writes export_linked/, never touches export/):
python scripts/enrich_dataset.py export/ export_linked/

# 4. the whole solution, in dependency order:
python run_pipeline.py                    # features -> tiers -> grades -> dashboard data
python experiments.py run                 # method ladder (auto-builds its cache)
python exp_text.py                        # text-model sweep
python exp_final.py                       # nested validation -> THE quotable number
python tune_router.py                     # cost-quality frontier benchmark
python sweep_defaults.py                  # the sweeps behind the shipped defaults
python infer_tiers.py && python cache_trap.py && python matching_check.py
python build_findings.py                  # bundle findings for dashboard + deck
python freeze_router.py                   # deployable frozen-router artifact

# 5. held-out chunk (when it drops) — zero refitting:
python apply_frozen.py export_linked/trajectories_v1_02.jsonl
```

Open the lab and the deck with `python -m http.server 8017 -d dashboard`
(or the `dashboard` entry in `.claude/launch.json`): `index.html` is the
interactive lab, `present.html` the 5-minute deck.

## For judges: run it on YOUR data

Any export in the challenge format works — one JSONL line per LLM request with
`model`, `input` (Responses-format item list), `tools`. Two modes:

**Mode A — held-out scoring (the honest test: nothing refits on your data).**

```bash
# put YOUR chunk(s) in export/ and rebuild the id layer
mkdir -p export && tar xzf trajectories_v1_02.jsonl.tar.gz -C export/
python scripts/load_trajectories.py export/            # reconstruction + validator stats
python scripts/enrich_dataset.py export/ export_linked/

# freeze the router on OUR fit chunk once (regenerates the gitignored pickle —
# it embeds TF-IDF vocabulary, so the license keeps it out of git):
python freeze_router.py export_linked/trajectories_v1_01.jsonl

# apply frozen to your chunk: every transform (feature ECDFs, TF-IDF, logits,
# cut values, the evaluator yardstick) is fixed from the fit chunk
python apply_frozen.py export_linked/trajectories_v1_02.jsonl
# -> prints exact/adjacent/served/savings, writes results/heldout_report.json

# or route ONE request (single-task inference path):
python apply_frozen.py --one some_request.json
```

**Mode B — the full analysis on your data.** Replace the chunks in `export/`,
rerun steps 2–4 of the reproduction above, and reload the dashboard: every
number, chart, finding card and deck slide recomputes from your data — nothing
in `dashboard/` is hand-typed. The defaults live in `run_pipeline.py`
(`TIER_PRICES`, α/τ/λ) if you want to test other assumptions, and every knob is
also live in the lab UI.

Both modes were run on the second data drop (`trajectories_v1_02`, 1,000 unseen
tasks): **67.8% exact / 97.2% adjacent agreement, 83.1% served, 45.0% saved with
zero refitting** — plus one claim that only half replicated. The worked example,
with the distribution check that makes those numbers readable, is
[docs/HELDOUT_CHUNK02.md](docs/HELDOUT_CHUNK02.md).

Caveats that carry over to any data: token counts are chars/4 estimates (no
`usage` in the format), prices are an assumption for anonymized ids, and
Mode A's evaluator grades your chunk with the FIT chunk's yardstick — that is
the point, but expect distribution shift on small or unusual chunks
(see `results/heldout_chunk00.json` for a worked n=25 example, and
`scripts/heldout_shift.py` to measure the shift on yours before reading the
agreement numbers).

## Using a coding agent

Point Claude Code / Codex / Cursor / opencode at this repo — `AGENTS.md` briefs your agent.
In Claude Code you also get slash commands:

- `/setup` — set up everything needed to participate
- `/make-presentation` — build a Viktor-branded presentation of your solution
- `/prepare-submission` — package your solution into a formal submission

## What's here

| Path | What |
|---|---|
| `AGENTS.md` | Agent briefing: dataset shape, the cache trap, judging, starter ideas |
| `docs/DEPLOYMENT.md` | Set-up checklist for a new machine: look → route → reproduce, with verification and troubleshooting |
| `docs/HELDOUT_CHUNK02.md` | The second data drop, run end to end: what transferred, what only half replicated, what the shift check says |
| `skills/` | The three guided workflows above (plain Markdown, readable by humans too) |
| `scripts/` | Loader + trajectory reconstruction, baseline router, cache-aware cost model (estimated tokens), frontier plot, held-out distribution-shift check, synthetic sample |
| `templates/presentation.html` | Self-contained branded slide template |

## Rules that matter

- **License:** challenge use only — no redistribution of the dataset. Full terms ship with the download.
- No GPU or API keys needed. Judge-model rescoring is allowed (credits announced at kickoff).
- Questions → the challenge Discord; the Viktor team answers there all weekend.
