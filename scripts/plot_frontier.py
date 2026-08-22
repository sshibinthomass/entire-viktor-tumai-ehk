#!/usr/bin/env python3
"""Cost–quality frontier from results/routes.jsonl.

Each model's quality proxy is the mean of its LiveBench Global Average and
Reasoning Average. Overall quality is the call-weighted mean of those scores.

Usage: python scripts/plot_frontier.py results/routes.jsonl
Writes results/frontier.csv (+ frontier.png if matplotlib is installed).
"""
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROUTE_MODELS_PATH = REPOSITORY_ROOT / "benchmarks/route-models.json"
MODEL_MAP_PATH = REPOSITORY_ROOT / "benchmarks/route-model-livebench-map.json"
LEADERBOARD_PATH = REPOSITORY_ROOT / "benchmarks/livebench-leaderboard.csv"


@cache
def load_model_quality_scores() -> dict[str, float]:
    """Load and validate the route-model to LiveBench quality scores."""
    route_models = json.loads(ROUTE_MODELS_PATH.read_text(encoding="utf-8"))
    model_map = json.loads(MODEL_MAP_PATH.read_text(encoding="utf-8"))

    expected = set(route_models)
    mapped = set(model_map)
    if expected != mapped:
        missing = sorted(expected - mapped)
        extra = sorted(mapped - expected)
        raise ValueError(f"model mapping mismatch: missing={missing}, extra={extra}")

    with LEADERBOARD_PATH.open(newline="", encoding="utf-8") as leaderboard_file:
        reader = csv.DictReader(leaderboard_file)
        required = {"Model", "Global Average", "Reasoning Average"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"leaderboard missing columns: {', '.join(sorted(missing_columns))}"
            )
        leaderboard = {row["Model"]: row for row in reader}

    scores = {}
    for route_model in route_models:
        livebench_model = model_map[route_model]
        if livebench_model not in leaderboard:
            raise ValueError(
                f"LiveBench model not found for {route_model}: {livebench_model}"
            )
        row = leaderboard[livebench_model]
        global_average = float(row["Global Average"])
        reasoning_average = float(row["Reasoning Average"])
        scores[route_model] = (global_average + reasoning_average) / 2
    return scores


def quality(
    models: Sequence[str],
    model_scores: Mapping[str, float] | None = None,
) -> float:
    """Return the average LiveBench-derived quality for the routed calls."""
    if isinstance(models, str) or not models:
        raise ValueError("models must be a non-empty sequence of model names")
    scores = model_scores if model_scores is not None else load_model_quality_scores()
    unknown = sorted(set(models) - set(scores))
    if unknown:
        raise ValueError(f"no quality score for model(s): {', '.join(unknown)}")
    return sum(scores[model] for model in models) / len(models)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "results/routes.jsonl"
    with open(src, encoding="utf-8") as routes_file:
        recs = [json.loads(line) for line in routes_file if line.strip()]

    for record in recs:
        if len(record["route"]) != record["n_calls"]:
            raise ValueError(
                f"trajectory {record['trajectory']}: route length does not match n_calls"
            )

    model_scores = load_model_quality_scores()
    total_calls = sum(r["n_calls"] for r in recs)
    # Sweep how much of the router's proposal to adopt: adopt for the cheapest X% of
    # trajectories first (they are the safest bets), keep the rest on the logged model.
    by_size = sorted(recs, key=lambda r: r["cost_logged_usd"])
    rows = []
    for frac in [i / 10 for i in range(0, 11)]:
        adopt = set(id(r) for r in by_size[: int(round(frac * len(recs)))])
        cost = sum(r["cost_routed_usd"] if id(r) in adopt else r["cost_logged_usd"] for r in recs)
        quality_total = 0.0
        for record in recs:
            models = (
                record["route"]
                if id(record) in adopt
                else [record["logged_model"]] * record["n_calls"]
            )
            quality_total += quality(models, model_scores) * len(models)
        rows.append({"adopt_frac": frac, "cost_usd": round(cost, 4),
                     "quality_score": round(quality_total / total_calls, 3)})
    out = Path(src).parent / "frontier.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    for r in rows: print(r)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot([r["cost_usd"] for r in rows], [r["quality_score"] for r in rows], "o-", color="#6748FD")
        plt.xlabel("cost (USD, estimated tokens, cache-aware)"); plt.ylabel("quality (LiveBench proxy)")
        plt.title("Cost–quality frontier"); plt.tight_layout()
        png = Path(src).parent / "frontier.png"; plt.savefig(png, dpi=150)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not installed — skipped PNG (CSV is enough)")

if __name__ == "__main__":
    main()
