# Viktor Challenge — Build the Router

Our solution for the **Viktor Challenge** at the TUM.ai hackathon (Munich, 22–23 Aug 2026):
a two-part **tier router + trajectory evaluator** with an honest off-policy evaluation.
The full writeup is in [SOLUTION.md](SOLUTION.md); the license-containment status is in
[docs/LICENSE_CONTAINMENT.md](docs/LICENSE_CONTAINMENT.md).

## Reproduce from a clean checkout

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
| `skills/` | The three guided workflows above (plain Markdown, readable by humans too) |
| `scripts/` | Loader + trajectory reconstruction, baseline router, cache-aware cost model (estimated tokens), frontier plot, synthetic sample |
| `templates/presentation.html` | Self-contained branded slide template |

## Rules that matter

- **License:** challenge use only — no redistribution of the dataset. Full terms ship with the download.
- No GPU or API keys needed. Judge-model rescoring is allowed (credits announced at kickoff).
- Questions → the challenge Discord; the Viktor team answers there all weekend.
