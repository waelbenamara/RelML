import sys
from pathlib import Path

import duckdb
import guepard.qwery.relml as relml

conn = duckdb.connect(":memory:")

_repo_root = Path(__file__).resolve().parents[2]
_default_ml1m = _repo_root / "data" / "ml-1m-data"
data_dir = sys.argv[1] if len(sys.argv) > 1 else str(_default_ml1m)

print(f"Loading data from {data_dir}...")
conn.execute(f'CREATE TABLE users   AS SELECT * FROM read_csv_auto("{data_dir}/users.csv",   header=true)')
conn.execute(f'CREATE TABLE movies  AS SELECT * FROM read_csv_auto("{data_dir}/movies.csv",  header=true)')
conn.execute(f'CREATE TABLE ratings AS SELECT * FROM read_csv_auto("{data_dir}/ratings.csv", header=true)')

for t in ["users", "movies", "ratings"]:
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {count} rows")

task = relml.TaskSpec(
    sql="""
        SELECT
            ratingId,
            userId,
            movieId,
            CASE WHEN rating >= 4.0 THEN 1.0 ELSE 0.0 END AS label,
            timestamp
        FROM ratings
    """,
    task_table_name = "ratings_task",
    target_column   = "label",
    task_type       = "binary_classification",
    label_transform = {"kind": "threshold", "threshold": 0.5, "inclusive": True},
    split_strategy  = "temporal",
    time_col        = "timestamp",
    inference_mode  = "entity_synthesis",
    entity_refs     = {"userId": "1", "movieId": "1193"},
)

print("\nStarting training...")
model = relml.train(
    conn,
    task,
    channels   = 64,
    gnn_layers = 2,
    hidden     = 64,
    dropout    = 0.3,
    lr         = 3e-4,
    epochs     = 3,
    batch_size = 0,
)

print("\nEntity synthesis inference:")
prob = model.predict_entity({"userId": "1", "movieId": "1193"})
print(f"  P(user 1 likes movie 1193) = {prob:.4f}")

print("\nScoring all rows:")
all_preds = model.predict_all()
n_pos = sum(1 for p in all_preds if p > 0.5)
print(f"  {n_pos} / {len(all_preds)} predicted positive ({100*n_pos/len(all_preds):.1f}%)")
