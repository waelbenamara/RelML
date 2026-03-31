import sys
from pathlib import Path

import duckdb
import guepard.qwery.relml as relml

_repo_root = Path(__file__).resolve().parents[2]
_default_sunny = _repo_root / "data" / "sunny-side-data" / "database_sunnyside"
data_dir = sys.argv[1] if len(sys.argv) > 1 else str(_default_sunny)

print(f"Loading Sunny Side slot data from {data_dir}...")
conn = duckdb.connect(":memory:")

conn.execute(f'CREATE TABLE dim_slot           AS SELECT * FROM read_csv_auto("{data_dir}/dim_slot.csv",           header=true)')
conn.execute(f'CREATE TABLE slot_week_learning AS SELECT * FROM read_csv_auto("{data_dir}/slot_week_learning.csv", header=true)')

for t in ["dim_slot", "slot_week_learning"]:
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {count} rows")

# Same pattern as demand: cast the TEXT label column to DOUBLE in SQL.
# Rows where txn_count_next_week is NULL are excluded from all splits.
task = relml.TaskSpec(
    sql="""
        SELECT
            row_id,
            slot_entity_id,
            day_of_week_num,
            time_slot,
            iso_year,
            iso_week,
            transaction_count,
            txn_lag1,
            txn_lag2,
            txn_lag3,
            txn_lag4,
            txn_rolling4_mean,
            temp_max_week_avg,
            temp_min_week_avg,
            precip_week_sum,
            wind_week_max,
            week_start,
            TRY_CAST(txn_count_next_week AS DOUBLE) AS visits_next_week
        FROM slot_week_learning
    """,
    task_table_name = "swl_task",
    target_column   = "visits_next_week",
    task_type       = "regression",
    label_transform = {"kind": "normalize"},
    split_strategy  = "temporal",
    time_col        = "week_start",
    inference_mode  = "row_based",
    inference_agg   = "mean",
)

print("\nStarting training...")
model = relml.train(
    conn,
    task,
    channels   = 32,
    gnn_layers = 2,
    hidden     = 64,
    dropout    = 0.2,
    lr         = 3e-4,
    epochs     = 3000,
    batch_size = 0,
)

print("\nScoring all slot x week rows...")
all_preds = model.predict_all()

# Build entity info map from dim_slot
slots = conn.execute("""
    SELECT slot_entity_id, day_of_week, slot_label
    FROM dim_slot
""").fetchall()
entity_info = {row[0]: (row[1], row[2]) for row in slots}

# Get the most recent iso_year/week that has a label
max_year, max_week = conn.execute("""
    SELECT iso_year, iso_week
    FROM swl_task
    WHERE visits_next_week IS NOT NULL
    ORDER BY iso_year DESC, iso_week DESC
    LIMIT 1
""").fetchone()

# Fetch rows for the most recent labeled week
recent_rows = conn.execute(f"""
    SELECT slot_entity_id, iso_year, iso_week, visits_next_week, row_id
    FROM swl_task
    WHERE iso_year = {max_year} AND iso_week = {max_week}
      AND visits_next_week IS NOT NULL
    ORDER BY slot_entity_id
""").fetchall()

# Map row_id to prediction index
all_row_ids = conn.execute("SELECT row_id FROM swl_task").fetchall()
rowid_to_idx = {row[0]: i for i, row in enumerate(all_row_ids)}

DAY_ORDER = {"Tuesday":1,"Wednesday":2,"Thursday":3,
             "Friday":4,"Saturday":5,"Sunday":6}

forecast = []
for slot_eid, yr, wk, actual, rid in recent_rows:
    idx = rowid_to_idx.get(rid)
    pred = all_preds[idx] if idx is not None and idx < len(all_preds) else 0.0
    day, slot_lbl = entity_info.get(slot_eid, ("?", "?"))
    forecast.append((DAY_ORDER.get(day, 99), day, slot_lbl, pred, actual))

forecast.sort(key=lambda x: (x[0], x[2]))

print(f"\nStaffing forecast — week {max_week} / {max_year}:")
print("-" * 62)
print(f"{'Day':<12} {'Slot':<22} {'Predicted':>12} {'Actual':>8}")
print("-" * 62)
for _, day, slot_lbl, pred, actual in forecast:
    print(f"{day:<12} {slot_lbl:<22} {pred:>12.1f} {actual:>8.1f}")
