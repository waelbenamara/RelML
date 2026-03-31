#!/usr/bin/env python3
"""
Load any RelBench dataset and save all tables to CSV files.

Usage:
    python scripts/load_relbench_to_csv.py <dataset_name> [--output-dir DIR] [--download]
    python scripts/load_relbench_to_csv.py --list-datasets

Examples:
    python scripts/load_relbench_to_csv.py rel-hm --download
    python scripts/load_relbench_to_csv.py rel-f1 --output-dir data/rel-f1-data --download
"""

import argparse
import sys
from pathlib import Path

from relbench.datasets import get_dataset, get_dataset_names

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Load a RelBench dataset and save all tables to CSV"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (e.g. rel-hm, rel-f1, rel-amazon). Use --list-datasets to see all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for CSV files (default: data/<dataset>-data under repo root)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from RelBench server if not cached",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available RelBench dataset names and exit",
    )
    args = parser.parse_args()

    if args.list_datasets:
        names = get_dataset_names()
        print("Available RelBench datasets:")
        for name in sorted(names):
            print(f"  {name}")
        return 0

    if not args.dataset:
        parser.error("dataset name required (or use --list-datasets)")
        return 1

    if args.dataset not in get_dataset_names():
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        print("Use --list-datasets to see available names.", file=sys.stderr)
        return 1

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _REPO_ROOT / "data" / f"{args.dataset}-data"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} dataset...")
    dataset = get_dataset(args.dataset, download=args.download)
    db = dataset.get_db()

    print(f"Saving {len(db.table_dict)} tables to {out_dir.absolute()}...")
    for table_name, table in db.table_dict.items():
        path = out_dir / f"{table_name}.csv"
        table.df.to_csv(path, index=False)
        print(f"  {table_name}.csv ({len(table.df):,} rows)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
