# Dataset license containment — status

The challenge dataset is **challenge-use-only, no redistribution** (AGENTS.md hard
rule). An audit found it publicly reachable through this repo. Containment status:

## Done (2026-08-23)

- **Repo set to private** (was public).
- **History PURGED and force-pushed**: `git filter-repo` removed from every
  branch's history: `trajectories_v1_01.jsonl.tar.gz` (31.7 MB raw blob),
  `export/trajectories_v1_01.jsonl` (LFS pointer), `export/LICENSE`, all
  historical versions of `dashboard/data.js` and `results/router_features.jsonl`
  (two versions of each carried verbatim task previews), and the starter zip.
  Current CLEAN versions of the two artifacts (numbers/ids/model names only,
  verified zero preview text) were re-committed. All six remote branches were
  force-updated; repo pack shrank 34 MB → 3.2 MB. A full-blob scan across all
  branches confirmed no other historical file carries dataset text (needles:
  preview fields, redaction markers, image placeholders, prompt phrases).
- **Working tree**: `export/`, `export_linked/`, archives, `dashboard/previews.js`,
  `results/exp_cache.npz`, `results/frozen_*.pkl` are untracked and gitignored —
  all dataset-derived text stays local.
- No tags, no LFS objects referenced by the rewritten history.

## Remaining actions (owner)

1. **Anyone with an old clone must re-clone.** Old clones re-introduce the purged
   blobs if pushed. This includes teammates' machines (branches `ivan_branch`,
   `router_models_mark` were force-updated).
2. **This machine still holds pre-purge history in LOCAL-ONLY branches** —
   `claude/evaluator-architecture-c36cdb`, `claude/trajectories-router-data-2eb29b`,
   `claude/tumai-challenge-analysis-cb55d5`, `claude/tumai-challenge-solution-602ea6`,
   `claude/tumai-router-evaluator-1022a9`, `feat/router-pipeline` (several pinned
   by `.claude/worktrees/`). Local possession is licensed; the hazard is a future
   `git push --all`, which would re-upload the blobs. When those experiments are
   no longer needed: `git worktree remove <path>` for each stale worktree, then
   `git branch -D <branch>`, then `git gc --prune=now`.
3. **GitHub server-side remnants**: force-pushed-away commits can stay reachable
   by SHA in GitHub's cache, and the old LFS object may remain in the repo's LFS
   storage. Ask GitHub Support (sensitive-data-removal flow,
   https://support.github.com/) to run gc and drop cached views; check
   Settings → Git LFS for the 105 MB object. Anyone who forked while the repo
   was public holds an independent copy — support can detach/clear forks.

Keep the repo private regardless: previews are dataset content, and the license
does not allow public redistribution in any form.
