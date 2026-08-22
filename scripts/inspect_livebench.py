#!/usr/bin/env python3
"""Inspect the locally downloaded LiveBench coding Parquet dataset."""

import argparse
import json
from pathlib import Path


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/livebench-coding/data/test-00000-of-00001.parquet"
)
DEFAULT_PREVIEW_COLUMNS = (
    "question_id",
    "category",
    "question_title",
    "task",
    "release_date",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def open_dataset(path: Path):
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise SystemExit(
            "pyarrow is required; install it with: "
            "python -m pip install 'pyarrow>=18'"
        ) from error

    if not path.is_file():
        raise SystemExit(
            f"dataset not found: {path}\n"
            "download it with: hf download livebench/coding --repo-type dataset "
            "--local-dir benchmarks/livebench-coding"
        )
    return parquet.ParquetFile(path)


def validate_columns(dataset, columns: list[str]) -> None:
    available = set(dataset.schema_arrow.names)
    unknown = sorted(set(columns) - available)
    if unknown:
        raise SystemExit(
            f"unknown column(s): {', '.join(unknown)}\n"
            f"available columns: {', '.join(dataset.schema_arrow.names)}"
        )


def print_info(dataset, path: Path) -> None:
    print(f"path: {path}")
    print(f"rows: {dataset.metadata.num_rows}")
    print(f"row groups: {dataset.metadata.num_row_groups}")
    print(f"columns: {len(dataset.schema_arrow.names)}")


def print_columns(dataset) -> None:
    for name in dataset.schema_arrow.names:
        print(name)


def print_head(dataset, columns: list[str], limit: int) -> None:
    validate_columns(dataset, columns)
    batches = dataset.iter_batches(batch_size=limit, columns=columns)
    batch = next(batches, None)
    if batch is None:
        return
    for row in batch.to_pylist():
        print(json.dumps(row, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Parquet file to inspect (default: {DEFAULT_DATASET})",
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("info", help="show dataset dimensions")
    commands.add_parser("columns", help="list top-level column names")
    commands.add_parser("schema", help="show column names and Arrow types")
    head = commands.add_parser("head", help="print a JSONL row preview")
    head.add_argument("--limit", type=positive_int, default=3)
    head.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_PREVIEW_COLUMNS),
        help="columns to include in the preview",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = open_dataset(args.path)
    command = args.command or "info"

    if command == "info":
        print_info(dataset, args.path)
    elif command == "columns":
        print_columns(dataset)
    elif command == "schema":
        print(dataset.schema_arrow)
    elif command == "head":
        print_head(dataset, args.columns, args.limit)


if __name__ == "__main__":
    main()
