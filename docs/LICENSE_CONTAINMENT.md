# Dataset license containment — status and remaining steps

The challenge dataset is **challenge-use-only, no redistribution** (AGENTS.md hard
rule). An audit found it publicly reachable through this repo. This file tracks
the containment.

## Done

- **Repo set to private** (was public) — the dataset is no longer publicly
  reachable through GitHub's UI.
- **Working tree clean**: `export/`, `export_linked/`, `*.tar.gz`, `*.zip` are
  untracked and gitignored.
- **Committed artifacts stripped**: `results/router_features.jsonl` and
  `dashboard/data.js` no longer carry verbatim `_preview`/`_trigger` text; the
  previews live only in the gitignored `dashboard/previews.js`. The experiment
  cache (`results/exp_cache.npz`, TF-IDF-bearing `results/frozen_router.pkl`)
  are gitignored because they embed dataset-derived text.

## Still required: purge the git HISTORY (destructive — confirm before running)

The pushed history still contains:

- `trajectories_v1_01.jsonl.tar.gz` — 31.7 MB raw blob (commit `c21aece`)
- `export/trajectories_v1_01.jsonl` — Git-LFS pointer to the 105 MB JSONL
- historical versions of `dashboard/data.js` and `results/router_features.jsonl`
  with verbatim first-user-message previews

The later "remove from tracking" commit (`261c20c`) removed them from the
working tree only, not from history. Anyone who cloned/forked while public may
still hold copies. To purge:

```bash
# from a FRESH clone (git-filter-repo refuses to run in a dirty original)
pip install git-filter-repo
git clone https://github.com/sshibinthomass/entire-viktor-tumai-ehk.git purge-clone
cd purge-clone
git filter-repo --invert-paths \
  --path trajectories_v1_01.jsonl.tar.gz \
  --path export/trajectories_v1_01.jsonl \
  --path export/trajectories_v1_00.jsonl \
  --path dashboard/data.js \
  --path results/router_features.jsonl
# re-add the CURRENT (clean) versions of the two regenerated artifacts
# (copy them from your working repo, commit), then:
git remote add origin https://github.com/sshibinthomass/entire-viktor-tumai-ehk.git
git push origin --force --all
git push origin --force --tags
```

Then:

1. Ask GitHub support to clear cached views and detach any forks:
   https://support.github.com/ (the "sensitive data removal" flow).
2. Every local clone must be re-cloned (old clones re-introduce the blobs on push).
3. Delete any Git-LFS objects from the repo's LFS storage
   (Settings → Git LFS, or `git lfs prune` after the rewrite).

Keep the repo private regardless — the dataset license does not allow public
redistribution even without the raw blobs, and previews are dataset content.
