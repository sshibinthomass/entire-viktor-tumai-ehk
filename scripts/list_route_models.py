#!/usr/bin/env python3
"""Write the unique model IDs referenced by a routes JSONL file."""

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "results/routes.jsonl"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "benchmarks/route-models.json"


def collect_models(source: Path) -> list[str]:
    models: set[str] = set()
    with source.open(encoding="utf-8") as routes:
        for line_number, line in enumerate(routes, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid JSON on line {line_number}: {error}") from error

            logged_model = record.get("logged_model")
            route = record.get("route")
            if not isinstance(logged_model, str) or not logged_model:
                raise SystemExit(f"line {line_number}: logged_model must be a non-empty string")
            if not isinstance(route, list) or not all(
                isinstance(model, str) and model for model in route
            ):
                raise SystemExit(
                    f"line {line_number}: route must be a list of non-empty strings"
                )

            models.add(logged_model)
            models.update(route)

    return sorted(models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    models = collect_models(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(models)} unique models to {args.output}")


if __name__ == "__main__":
    main()
