#!/usr/bin/env python3
"""
Give a RelBench dataset name; the script downloads it and creates a folder
of CSV files (one per table), e.g. rel-hm -> data/rel-hm-data/*.csv.

Usage:
    python scripts/import_rel_trial.py <dataset_name>
    python scripts/import_rel_trial.py --list-datasets   # show available names
"""

import argparse
import sys
from pathlib import Path

from relbench.datasets import get_dataset, get_dataset_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a RelBench dataset and save each table as CSV in data/<dataset>-data/"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (e.g. rel-trial, rel-hm). Use --list-datasets to see all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder for CSV files (default: data/<dataset>-data under repo root)",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List available dataset names and exit")
    args = parser.parse_args()

    if args.list_datasets:
        names = get_dataset_names()
        print("Available RelBench datasets:")
        for n in sorted(names):
            print(f"  {n}")
        sys.exit(0)

    if not args.dataset:
        parser.error("dataset name required (or use --list-datasets)")
        sys.exit(1)

    try:
        dataset = get_dataset(args.dataset, download=True)
    except KeyError:
        print(f"Unknown dataset: {args.dataset!r}", file=sys.stderr)
        print("Available names:", ", ".join(sorted(get_dataset_names())), file=sys.stderr)
        print("Run with --list-datasets to list them.", file=sys.stderr)
        sys.exit(1)

    db = dataset.get_db()
    _repo_root = Path(__file__).resolve().parents[1]
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _repo_root / "data" / f"{args.dataset}-data"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(db.table_dict)} tables to {out_dir.absolute()}...")
    for table_name, table in db.table_dict.items():
        path = out_dir / f"{table_name}.csv"
        table.df.to_csv(path, index=False)
        print(f"  {table_name}.csv ({len(table.df):,} rows)")
    print("Done.")
