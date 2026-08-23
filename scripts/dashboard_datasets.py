"""Multi-dataset support for the lab: one data file per chunk + a manifest.

The lab used to load a single `dashboard/data.js`, so looking at a second chunk
meant overwriting the first. Instead each pipeline run also drops its own copy
under `dashboard/data/<slug>.js` (plus `<slug>.findings.js` and the local-only
`<slug>.previews.js`), and the manifest `dashboard/datasets.js` lists whatever
is in that directory. The dropdown in `dashboard/index.html` is built from the
manifest, so a chunk appears there as soon as the pipeline has been run on it —
nothing to register by hand.

`dashboard/data.js` / `findings.js` / `previews.js` are still written exactly as
before: they are the fallback when no manifest exists (a fresh checkout, or a
dashboard copied somewhere on its own).

License note: the per-dataset data and findings files carry counts only, like
the originals, so they are safe to commit. `<slug>.previews.js` holds verbatim
task text and is gitignored — same rule as `dashboard/previews.js`.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
DATA_DIR = DASH / "data"
MANIFEST = DASH / "datasets.js"

SUFFIX = {"data": ".js", "findings": ".findings.js", "previews": ".previews.js"}


def slug_of(source):
    """Dataset id from an export name: 'trajectories_v1_02.jsonl' -> 'trajectories_v1_02'."""
    stem = Path(str(source)).name
    for ext in (".jsonl", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem) or "dataset"


def write(source, kind, payload, var):
    """Write one per-dataset file; returns its path. `payload` is JSON-serializable."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / (slug_of(source) + SUFFIX[kind])
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"window.{var} = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    return path


def _meta_of(path):
    """Read a data file's meta block without keeping the whole payload around."""
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1]).get("meta")
    except json.JSONDecodeError:
        return None


def rebuild_manifest():
    """Scan dashboard/data/ and (re)write dashboard/datasets.js. -> list of entries."""
    entries = []
    if DATA_DIR.is_dir():
        for p in sorted(DATA_DIR.glob("*.js")):
            if p.name.endswith((".findings.js", ".previews.js")):
                continue
            meta = _meta_of(p)
            if not meta:
                continue
            entries.append({
                "id": p.stem,
                "source": meta.get("source", p.stem),
                "n": meta.get("n"),
                "label": f"{meta.get('source', p.stem)} — {meta.get('n', '?')} tasks",
            })
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("window.VIKTOR_DATASETS = ")
        json.dump(entries, f, separators=(",", ":"))
        f.write(";\n")
    return entries
