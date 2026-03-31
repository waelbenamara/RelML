import sys
from pathlib import Path

import duckdb
import guepard.qwery.relml as relml

_repo_root = Path(__file__).resolve().parents[2]
_default_sunny = _repo_root / "data" / "sunny-side-data" / "database_sunnyside"
data_dir = sys.argv[1] if len(sys.argv) > 1 else str(_default_sunny)

print(f"Loading Sunny Side data from {data_dir}...")
conn = duckdb.connect(":memory:")

conn.execute(f'CREATE TABLE dim_product              AS SELECT * FROM read_csv_auto("{data_dir}/dim_product.csv",              header=true)')
conn.execute(f'CREATE TABLE fact_product_week_orders AS SELECT * FROM read_csv_auto("{data_dir}/fact_product_week_orders.csv", header=true)')
conn.execute(f'CREATE TABLE product_week_learning    AS SELECT * FROM read_csv_auto("{data_dir}/product_week_learning.csv",    header=true)')

for t in ["dim_product", "fact_product_week_orders", "product_week_learning"]:
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {count} rows")

# The label column order_count_next_week is stored as TEXT in the CSV
# (to prevent the C++ encoder from seeing it as a feature).
# We cast it to DOUBLE here in SQL — rows where it is NULL (the most
# recent week for each product) are excluded automatically by relml
# because build_split skips null labels.
task = relml.TaskSpec(
    sql="""
        SELECT
            product_id,
            iso_year,
            iso_week,
            order_count,
            order_count_lag1,
            order_count_lag2,
            order_count_lag3,
            order_count_lag4,
            temp_max_week_avg,
            temp_min_week_avg,
            precip_week_sum,
            wind_week_max,
            week_start_date,
            TRY_CAST(order_count_next_week AS DOUBLE) AS demand_next_week
        FROM product_week_learning
    """,
    task_table_name = "pwl_task",
    target_column   = "demand_next_week",
    task_type       = "regression",
    label_transform = {"kind": "normalize"},
    split_strategy  = "temporal",
    time_col        = "week_start_date",
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
    epochs     = 500,
    batch_size = 0,
)

print("\nScoring all product x week rows...")
all_preds = model.predict_all()

# Load product names for display
products = conn.execute("""
    SELECT product_id, item_name, category
    FROM dim_product
""").fetchall()
id_to_info = {row[0]: (row[1], row[2]) for row in products}

# Get product_id column from the task table
rows = conn.execute('SELECT product_id FROM pwl_task').fetchall()
pid_sum = {}
pid_cnt = {}
for i, row in enumerate(rows):
    if i >= len(all_preds):
        break
    pid = row[0]
    pid_sum[pid] = pid_sum.get(pid, 0.0) + all_preds[i]
    pid_cnt[pid] = pid_cnt.get(pid, 0) + 1

ranked = sorted(
    [(pid, pid_sum[pid] / pid_cnt[pid]) for pid in pid_sum],
    key=lambda x: x[1], reverse=True
)

print("\nTop 10 products by mean predicted next-week demand:")
print(f"{'Product':<28} {'Category':<24} Predicted demand")
print("-" * 72)
for pid, pred in ranked[:10]:
    name, cat = id_to_info.get(pid, ("?", "?"))
    print(f"{name[:27]:<28} {cat[:23]:<24} {pred:.1f} orders/week")
