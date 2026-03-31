#!/usr/bin/env python3
"""
Run the MovieLens 1M benchmark (same setup as relml C++ test_training).

Loads rel_ml1m dataset and rating-prediction task, prints task stats,
and evaluates a constant baseline. For full GNN training, use relbench's
gnn_entity.py pattern with this dataset (see README).

Usage:
  python run_ml1m.py [--data-dir PATH] [--threshold 4.0]
  RELBENCH_ML1M_DATA=/path/to/data/ml-1m-data python run_ml1m.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add benchmarks dir so we can import rel_ml1m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_ML1M = _REPO / "data" / "ml-1m-data"

import numpy as np

from rel_ml1m import ML1MDataset, RatingPredictionTask


def main():
    parser = argparse.ArgumentParser(
        description="Run RelBench ML1M benchmark (same as relml C++ example)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("RELBENCH_ML1M_DATA", str(_DEFAULT_ML1M)),
        help="Path to directory containing users.csv, movies.csv, ratings.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expanduser("~/.cache/relbench/rel-ml1m"),
        help="Cache directory for database and task tables",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Rating threshold for binary label (>= threshold -> 1)",
    )
    args = parser.parse_args()

    print("Loading ML1M dataset (same schema as relml C++)...")
    dataset = ML1MDataset(cache_dir=args.cache_dir, data_dir=args.data_dir)
    db = dataset.get_db()

    print("\nBuilding rating task (>= {} stars)...".format(args.threshold))
    task = RatingPredictionTask(
        dataset=dataset,
        cache_dir=args.cache_dir,
        threshold=args.threshold,
    )

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    n_train, n_val, n_test = len(train_table.df), len(val_table.df), len(test_table.df)
    n_pos_train = int(train_table.df[task.target_col].sum())
    n_pos = int(
        train_table.df[task.target_col].sum()
        + val_table.df[task.target_col].sum()
        + test_table.df[task.target_col].sum()
    )
    n_total = n_train + n_val + n_test
    print(
        "  Ratings: {}  positive rate: {:.3f}  (rating >= {})".format(
            n_total, n_pos / n_total, args.threshold
        )
    )
    print(
        "  Samples — train: {}  val: {}  test: {}".format(n_train, n_val, n_test)
    )
    print(
        "  Train label rate: {:.3f}".format(n_pos_train / n_train if n_train else 0)
    )

    # Baseline: predict train positive rate for all test rows
    test_pred = np.full(len(test_table.df), n_pos_train / n_train, dtype=np.float32)
    test_metrics = task.evaluate(test_pred)
    print("\nBaseline (constant = train pos rate) test metrics:")
    for k, v in test_metrics.items():
        print("  {}: {:.4f}".format(k, v))

    return 0


if __name__ == "__main__":
    sys.exit(main())
