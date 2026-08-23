# Deployment checklist

How a person who has never seen this repo gets the whole system running on their
own machine. Three levels — do them in order, each one is independently useful:

| Level | You get | Needs the dataset? | Time |
|---|---|---|---|
| **0 — Look** | The lab + the 5-minute deck, every number live | No | ~5 min |
| **1 — Route** | A deployable frozen router; score a held-out chunk or one request | Yes (1 chunk) | ~20 min |
| **2 — Reproduce** | Every committed artifact rebuilt from scratch | Yes | 1–2 h (mostly unattended) |

Level 0 works from a bare clone because `dashboard/data.js`, `dashboard/findings.js`
and all of `results/*.json` are committed. The dataset is **not** committed and never
will be — read [Guardrails](#guardrails) before you touch it.

---

## Prerequisites

- [ ] **Python ≥ 3.11** — `python --version`. The repo pins 3.13 in `.python-version`; 3.11/3.12 are fine.
- [ ] **git** — `git --version`
- [ ] ~500 MB free disk (dataset chunk 01 is 32 MB compressed, 107 MB extracted; derived artifacts add ~50 MB)
- [ ] **No GPU. No API keys.** Everything runs on a laptop, offline. (`judge_rescore.py` is the one
      optional script that uses a key — skip it and nothing breaks.)
- [ ] Read access to the repo (it is **private** — the dataset license required that)

Not required: Docker, node, a database, any cloud account.

---

## Level 0 — Look at the results (no dataset)

- [ ] **Clone**

  ```bash
  git clone https://github.com/sshibinthomass/entire-viktor-tumai-ehk.git
  ```

- [ ] **Serve the dashboard** from the repo root — stdlib only, no dependencies needed for this level:

  ```bash
  python -m http.server 8017 -d dashboard
  ```

  In Claude Code / VS Code you can instead launch the `dashboard` entry in `.claude/launch.json`.

- [ ] **Open both pages**
  - <http://localhost:8017/present.html> — the 5-minute deck
  - <http://localhost:8017/> — the interactive lab (every hyperparameter live)

- [ ] **Verify it loaded real data**, not an empty shell. On the lab's default settings the
      headline tiles must read:

  | Tile | Expected |
  |---|---|
  | n (trajectories) | **1025** |
  | Exact agreement | **65.3 %** |
  | Adjacent | **97.1 %** |
  | Served | **82.2 %** |
  | Est. savings | **45.5 %** |

  These reproduce `results/comparison.json` to the digit — that match *is* the
  Python ↔ JS parity check. A blank or off tile means the page did not load `data.js`.

- [ ] **Expected console warning:** `previews.js` 404s. That is correct and harmless — it holds
      verbatim dataset text, so it is gitignored, and both pages load it behind an `onerror`
      handler. You lose only the hover-preview snippets.

- [ ] **Orient yourself:** read [docs/JUDGES.md](JUDGES.md) — the claim→artifact table says which
      committed file backs every number, and the rigor checklist says how to attack them.

**Stop here** if you only need to review the work. Levels 1–2 exist to run it on *new* data.

---

## Level 1 — Deploy the router (needs the dataset)

### 1.1 Install dependencies

- [ ] Create the environment. Either tool works:

  ```bash
  uv sync
  ```

  ```bash
  python -m venv .venv && .venv/Scripts/pip install -e .
  ```

  (macOS / Linux: `.venv/bin/pip install -e .`) This installs numpy, scipy and
  scikit-learn from `pyproject.toml`. `matplotlib` is optional — `pip install -e ".[plots]"`
  only if you want the frontier PNG.

- [ ] **Verify:**

  ```bash
  python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
  ```

  Reference machine: `2.5.2 1.18.1 1.9.0`.

- [ ] **Run everything from the repo root, with the venv's interpreter.** The scripts do bare
      `import rank_utils` / `import router`, so another cwd breaks them. On Windows that is
      `.venv/Scripts/python <script>`. If the venv was created by `uv` it has **no pip** —
      add packages with `uv pip install <pkg> -p .venv`.

### 1.2 Get the dataset

- [ ] Obtain the archives — challenge use only, links posted at kickoff / in the challenge
      Discord. The signed URL in `skills/setup/SKILL.md` step 2 has almost certainly expired;
      ask in Discord for a fresh one.
- [ ] Extract into `export/`:

  ```bash
  mkdir -p export && tar xzf trajectories_v1_01.jsonl.tar.gz -C export/
  ```

- [ ] Read `export/LICENSE` — it ships inside the archive.
- [ ] **No dataset access?** You can still exercise every code path on a shape-identical fake:

  ```bash
  python scripts/make_synthetic_sample.py
  ```

  Numbers will not match the committed ones — it is synthetic, that is expected.

### 1.3 Sanity-check and reconstruct

- [ ] ```bash
  python scripts/load_trajectories.py export/
  ```

  Prints request counts, per-model distribution, reconstructed-trajectory stats and a schema
  check. **Verify:** each line has exactly `model`, `input`, `tools` — no `output`, no `usage`.
  The validator asserts item-prefix nesting; grouping is on system prompt + full first-user text.

- [ ] **Expected shape** for the official chunks: chunk 01 is pure single-call sampling
      (1000 requests → 1000 trajectories); the only real multi-call trajectories (25, all
      single-model, perfectly nested) live in chunk 00. A multi-model group means your
      grouping key is too loose — not that the one-model-per-trajectory premise broke.

- [ ] Assign trajectory ids — writes a copy, never mutates `export/`:

  ```bash
  python scripts/enrich_dataset.py export/ export_linked/
  ```

  Every downstream script reads `export_linked/` by default.

### 1.4 Freeze the router

The pickles are gitignored (they embed TF-IDF vocabulary built from dataset text), so
**every new machine must regenerate them** — this step is not optional.

- [ ] ```bash
  python freeze_router.py export_linked/trajectories_v1_01.jsonl
  ```

  Writes `results/frozen_router.pkl` (~7 MB): per-feature ECDFs, heuristic weights and fixed
  cut values, both TF-IDF vectorizers, the ordinal logits, α/τ/λ, the routing-time cost
  regressors, and the frozen evaluator yardstick.

### 1.5 Verify the router runs

- [ ] **Score a held-out chunk** — zero refitting, this is the honest test:

  ```bash
  python apply_frozen.py export_linked/trajectories_v1_00.jsonl
  ```

  Prints exact / adjacent / served / savings and writes `results/heldout_report.json`.
  **Compare against the committed reference** `results/heldout_chunk00.json`: n=25,
  adjacent 1.0, served 1.0, savings 60 %, exact 0.0. Yes — exact is zero. Chunk 00 is a
  shifted n=25 sample graded with chunk 01's yardstick; it is a *mechanism* demo, not a
  headline. Matching those numbers means your freeze is faithful.

- [ ] **Route a single request** (the production inference path):

  ```bash
  python apply_frozen.py --one some_request.json
  ```

  where `some_request.json` is one JSONL line (`model`, `input`, `tools`) saved to a file.
  Prints the tier plus P(D≤1), P(D≤2).

At this point the system is deployed — you can route new traffic. Level 2 only re-derives
the analysis behind it.

---

## Level 2 — Full reproduction

Run in this order; later steps read earlier steps' `results/` files. Times are from a laptop
CPU. Background the two long ones.

- [ ] `python run_pipeline.py` — features → tiers → grades → comparison → `dashboard/data.js`
      (+ local-only `previews.js`). *~2 min.* Also writes `results/comparison.json`,
      `tiers.jsonl`, `evaluator_metrics.jsonl`, `router_features.jsonl`.
- [ ] `python experiments.py run` — the method ladder. Auto-builds `results/exp_cache.npz`
      (~15 MB, gitignored — holds verbatim first-user text) on first run. *~5 min.*
- [ ] `python exp_text.py` — text-model sweep (vectorizers / channels / ordinal head). *~5 min.*
- [ ] `python exp_final.py` — nested validation, the quotable number. **Tens of minutes —
      background it.** Re-picks the config by inner CV per fold over the full 30-config grid.
- [ ] `python tune_router.py` — cost–quality frontier benchmark. **~10 min — background it.**
      `--fast` uses fewer search draws.
- [ ] `python sweep_defaults.py` — the α/τ/λ sweeps behind the shipped defaults.
- [ ] `python infer_tiers.py` — inferred tier order + per-model bootstrap stability. Reads
      `results/tiers.jsonl` + `evaluator_metrics.jsonl`, so it must follow `run_pipeline.py`.
- [ ] `python cache_trap.py` — measured vs modeled cache-reset cost.
- [ ] `python matching_check.py` — matched cross-model validation. Reads
      `results/model_tiers.json`, so it must follow `infer_tiers.py`.
- [ ] `python build_findings.py` — bundles the findings into `dashboard/findings.js`.
      **Run this last** — the dashboard and the deck render what it writes.
- [ ] `python freeze_router.py` — refresh the deployable artifact from the rebuilt fit.
- [ ] **Reload the dashboard** and re-check the Level 0 tile table.

### Verify the reproduction

- [ ] `results/comparison.json` → exact 0.6527, adjacent 0.9707, served 0.8224, savings 45.5 %, n 1025
- [ ] `results/final_validation.json` → nested exact **0.6595**, measured selection bias **+1.17 pt**, 30 configs in grid
- [ ] `results/tuning_report.json` → λ router AUC 0.892 vs its oracle bound 0.913; random baseline 0.849, CI [0.839, 0.862]
- [ ] `results/matching_check.json` → +6.7 pt error rate, CI [+2.5, +10.8]; +2.7 retry streak, CI [+1.6, +4.0]

Drift in the last decimal is fine (BLAS/threading nondeterminism). A different *first*
decimal means you fed it different data: `run_pipeline.py` defaults to the whole
`export_linked/` directory, so the committed numbers assume chunks 00 **and** 01 are present.

### Running it on your own export

Any export in the challenge format works. Two modes, spelled out in
[docs/JUDGES.md](JUDGES.md#run-it-on-your-own-data): **Mode A** is Level 1 pointed at your
chunk (nothing refits on your data — the honest test); **Mode B** is Level 2 with your chunks
in `export/` (every chart, tile and finding recomputes; nothing in `dashboard/` is hand-typed).
Defaults to override live in `run_pipeline.py` (`TIER_PRICES`, α/τ/λ), and every knob is also
live in the lab UI.

---

## Guardrails

Non-negotiable, ordered by how much damage getting it wrong does:

- [ ] **Never commit the dataset, or anything derived that embeds its text.** `export/`,
      `export_linked/`, `*.tar.gz`, `*.zip`, `dashboard/previews.js`, `results/exp_cache.npz`
      and `results/frozen_*.pkl` are all in `.gitignore` — leave them there. Challenge use
      only, no redistribution. This repo already needed one history purge over exactly this;
      see [docs/LICENSE_CONTAINMENT.md](LICENSE_CONTAINMENT.md).
- [ ] **`git status` before every commit.** If the dataset or a pickle shows up as stageable,
      fix `.gitignore` rather than reaching for `git add -A`.
- [ ] **Never `git push --all`.** History was rewritten on 2026-08-23; old clones' local-only
      branches still carry pre-purge objects, and pushing them re-publishes the dataset.
- [ ] **Re-clone rather than pull** if your clone predates the purge — every commit hash changed.
- [ ] **Token counts are chars/4 estimates** (there is no `usage` in the export) and prices are
      an assumption for anonymized ids. Quote ratios, not dollars, and say "estimated".

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: rank_utils` / `router` | wrong cwd | run from the repo root |
| `ModuleNotFoundError: numpy` | system python, not the venv | `.venv/Scripts/python <script>`, or activate the venv |
| `pip: command not found` inside `.venv` | venv was created by `uv` (no pip) | `uv pip install <pkg> -p .venv` |
| "Extract the dataset to export/ and run enrich_dataset.py first" | `export_linked/` missing or unenriched | `python scripts/enrich_dataset.py export/ export_linked/` |
| `FileNotFoundError: results/frozen_router.pkl` | gitignored on purpose, never shipped | `python freeze_router.py export_linked/trajectories_v1_01.jsonl` |
| `results/exp_cache.npz` missing | gitignored, rebuilt on demand | `python experiments.py build` (or just `run` — it auto-builds) |
| `infer_tiers.py` / `matching_check.py` can't find `tiers.jsonl` / `model_tiers.json` | ran out of order | `run_pipeline.py` → `infer_tiers.py` → `matching_check.py` |
| Dashboard tiles blank or all zeros | `data.js` didn't load | check the browser console; confirm you served the `dashboard/` dir, not the repo root |
| `previews.js` 404 in console | intended — it's gitignored | ignore, or regenerate locally via `run_pipeline.py` |
| Port 8017 already in use | another server | pick another: `python -m http.server 8018 -d dashboard` |
| `exp_final.py` looks hung | it genuinely takes tens of minutes | background it; use `tune_router.py --fast` for a quicker frontier |

---

## Using a coding agent

Point Claude Code / Codex / Cursor / opencode at the repo — [AGENTS.md](../AGENTS.md) is the
briefing (dataset shape, the cache trap, the hard rules). In Claude Code, `/setup` walks the
environment and dataset steps interactively; `/make-presentation` and `/prepare-submission`
cover the deck and the submission package.
