# Creating a Task in RelML

This document explains how to take a raw relational database, define a
prediction task, and train a model. Every example in `src/example_tasks/`
follows this exact pattern. RelML also ships a Python binding that lets you
drive the full pipeline from Python using DuckDB as the data layer.

---

## Choosing your interface

| | C++ | Python |
|---|---|---|
| Data loading | `CSVLoader` reads CSV files directly | DuckDB — load from CSV, Parquet, SQL, anything |
| Schema declaration | Explicit `TableSchema` structs | Auto-detected from DuckDB metadata + name matching |
| FK detection | `FKDetector::detect(db)` | Automatic via `FKDetector` called inside the binding |
| Label injection | Manual C++ column construction | SQL — derive the label directly in the `TaskSpec` query |
| Training | `Trainer::fit(split, db, graph)` | `relml.train(conn, task, ...)` |
| Inference | `predict_all`, `synthesize_prediction` | `model.predict_all()`, `model.predict_entity(...)` |

Both interfaces compile and train the exact same C++ pipeline. The Python
layer is a thin pybind11 wrapper that converts Python dicts and DuckDB query
results into `relml::Database` objects and hands them to the C++ engine.

---

## Python quick start

### Prerequisites

```bash
# RHEL 8
sudo yum install python39-devel

# Ubuntu / Debian
sudo apt install python3-dev

# macOS
brew install python
```

### Build the extension

```bash
cd build

cmake .. \
  -DPython_EXECUTABLE=$(which python) \
  -DPython_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))") \
  -DPython_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").so

make -j$(nproc) _relml_core
```

You should see `-- Python bindings: enabled` in the cmake output. The compiled
`.so` is automatically copied into `python/guepard/qwery/relml/` after each
build.

### Install the Python package

```bash
cd python
pip install -e .
```

### Verify

```bash
python -c "import guepard.qwery.relml; print('OK')"
```

---

## Python API

### `TaskSpec`

```python
import guepard.qwery.relml as relml

task = relml.TaskSpec(
    sql             = "SELECT ...",   # any SPJA query — result becomes the training table
    task_table_name = "my_task",      # name for the materialized result in DuckDB
    target_column   = "label",        # which column in the result is the label
    task_type       = "binary_classification",  # or "regression" / "multiclass_classification"
    label_transform = {"kind": "threshold", "threshold": 0.5, "inclusive": True},
    split_strategy  = "temporal",     # or "random"
    time_col        = "timestamp",    # required when split_strategy="temporal"
    inference_mode  = "row_based",    # or "entity_synthesis"
    inference_agg   = "none",         # "mean", "fraction", "count"
    entity_refs     = {},             # only for entity_synthesis
)
```

**`label_transform` options**

| `kind` | When to use | Extra fields |
|---|---|---|
| `threshold` | Binary classification | `threshold` (float), `inclusive` (bool) |
| `normalize` | Regression | none — mean/std fitted from training rows |
| `buckets` | Multiclass | `buckets` (sorted list of boundaries) |

### `relml.train`

```python
model = relml.train(
    conn,           # open DuckDB connection with data already loaded
    task,           # TaskSpec defined above
    channels   = 64,
    gnn_layers = 2,
    hidden     = 64,
    dropout    = 0.3,
    lr         = 3e-4,
    epochs     = 30,
    batch_size = 0,   # 0 = full batch
)
```

Returns a `TrainedModel` object.

### `TrainedModel`

```python
# Score every row in the target table
preds = model.predict_all()   # list[float]

# Entity synthesis — score a specific combination of entities
prob = model.predict_entity({"userId": "1", "movieId": "1193"})   # float
```

---

## Python task patterns

### Pattern A — label already exists

```python
import duckdb
import guepard.qwery.relml as relml

conn = duckdb.connect(":memory:")
conn.execute('CREATE TABLE users   AS SELECT * FROM read_csv_auto("users.csv",   header=true)')
conn.execute('CREATE TABLE movies  AS SELECT * FROM read_csv_auto("movies.csv",  header=true)')
conn.execute('CREATE TABLE ratings AS SELECT * FROM read_csv_auto("ratings.csv", header=true)')

task = relml.TaskSpec(
    sql="""
        SELECT ratingId, userId, movieId, rating, timestamp
        FROM ratings
    """,
    task_table_name = "ratings_task",
    target_column   = "rating",
    task_type       = "binary_classification",
    label_transform = {"kind": "threshold", "threshold": 4.0, "inclusive": True},
    split_strategy  = "temporal",
    time_col        = "timestamp",
    inference_mode  = "entity_synthesis",
    entity_refs     = {"userId": "1", "movieId": "1193"},
)

model = relml.train(conn, task, channels=64, epochs=30)
prob  = model.predict_entity({"userId": "1", "movieId": "1193"})
print(f"P(user 1 likes movie 1193) = {prob:.4f}")
```

### Pattern B — label derived in SQL

When the label does not exist as a column, derive it directly in the SQL
query passed to `TaskSpec`. This replaces the C++ `inject_*` functions.

```python
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
)
```

### Pattern C — label stored as text, cast in SQL

Some pipelines store the target column as `TEXT` to prevent the C++ encoder
from treating it as a numerical feature. Cast it to `DOUBLE` in the SQL query
using `TRY_CAST` so that null rows are excluded automatically from all splits.

```python
task = relml.TaskSpec(
    sql="""
        SELECT
            product_id,
            iso_year,
            iso_week,
            order_count,
            order_count_lag1,
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
)
```

### Pattern D — multiclass classification

Use `buckets` as the label transform. For a 3-class outcome (0 / 1 / 2),
the boundaries `[0.5, 1.5]` map raw values correctly:
raw 0 → class 0, raw 1 → class 1, raw 2 → class 2.

```python
task = relml.TaskSpec(
    sql="""
        SELECT
            fixture_id,
            home_team_id,
            away_team_id,
            CASE
                WHEN goals_home > goals_away THEN 0.0
                WHEN goals_home = goals_away THEN 1.0
                ELSE 2.0
            END AS outcome,
            date
        FROM games
        WHERE goals_home IS NOT NULL
    """,
    task_table_name = "games_task",
    target_column   = "outcome",
    task_type       = "multiclass_classification",
    label_transform = {"kind": "buckets", "buckets": [0.5, 1.5]},
    split_strategy  = "temporal",
    time_col        = "date",
    inference_mode  = "entity_synthesis",
    entity_refs     = {"home_team_id": "42", "away_team_id": "50"},
)
```

---

## How FK detection works in Python

DuckDB does not enforce or store foreign key constraints — tables created via
`CREATE TABLE AS SELECT` carry no FK metadata. The Python binding handles this
in two steps:

1. `_detect_fks()` queries `information_schema.referential_constraints`.
   This works for tables created with explicit `FOREIGN KEY` clauses.

2. For the task table (which is always created from a SQL query), the binding
   calls `FKDetector::detect(db)` from C++ after the database is assembled.
   This applies the same name-matching heuristic as the CSV loader:
   if a column named `userId` exists and `users` has a PK column also named
   `userId`, a FK is inferred automatically.

You never need to declare FKs manually in Python. As long as your column names
follow the pattern `<table>Id` or match the PK column name of another table,
detection is fully automatic.

---

## Running the Python examples

```bash
cd python

# MovieLens-1M — binary classification, entity synthesis
python examples/run_ml1m.py /path/to/ml-1m-data

# Sunny Side café — demand regression
python examples/run_sunnyside_demand.py /path/to/sunny-side-data/database_sunnyside

# Sunny Side café — staffing slot regression
python examples/run_sunnyside_slots.py /path/to/sunny-side-data/database_sunnyside
```

---

## The five steps (C++)

1. Load the CSV files and declare the schema
2. Detect foreign keys
3. Build the heterogeneous graph
4. Define the task (the part that requires the most thought)
5. Train and run inference

---

## Step 1 — Load the CSV files

```cpp
std::unordered_map<std::string, TableSchema> schemas = {
    {"users", {
        .pkey_col     = "user_id",          // primary key column
        .time_col     = std::nullopt,        // no timestamp on this table
        .foreign_keys = {},                  // declared or auto-detected
        .columns      = {
            // override only when type inference is wrong
            {.name = "country", .type = ColumnType::CATEGORICAL},
            {.name = "bio",     .type = ColumnType::TEXT},
        }
    }},
    {"orders", {
        .pkey_col     = "order_id",
        .time_col     = "created_at",        // used for temporal split
        .foreign_keys = {
            {.column = "user_id", .target_table = "users"},
        },
        .columns = {}                        // all types inferred automatically
    }},
};

Database db = CSVLoader::load_database("./my-data", "mydb", schemas);
```

**Type inference rules.**
`TypeInferrer` scans each column and assigns a type automatically:

| What it sees | Assigned type |
|---|---|
| All values parse as numbers | `NUMERICAL` |
| All values match `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS` | `TIMESTAMP` |
| String column with fewer than 5% unique values | `CATEGORICAL` |
| String column with 5%+ unique values | `TEXT` |

You only need to declare a column in `columns` when inference gets it wrong.
For example, an `age` column stored as integers `{1, 18, 25, 35}` will infer
as `NUMERICAL`. If those are actually bucket codes you want one-hot encoded,
override it as `CATEGORICAL`.

**What gets skipped by the encoder.**
The primary key column is never encoded. `TEXT` columns are always skipped.
`TIMESTAMP` columns declared as `time_col` are used for temporal splits but
not encoded. Foreign key columns are used to build graph edges but their raw
values (integer IDs) carry no semantic signal after z-scoring, so you should
override them as `TEXT` on tables where the FK is the only column you do not
want encoded.

---

## Step 2 — Detect foreign keys

```cpp
FKDetector::detect(db);
```

This writes FK relationships directly into each `Table::foreign_keys`. It
uses two checks: the column name must match the pattern
`singular(target_table) + "Id"` (e.g. `userId` → `users`) and at least 99%
of non-null values must exist in the target table's primary key column.

When auto-detection cannot find a FK — for example because the column is named
`home_team_id` instead of `teamId` — declare it explicitly in the schema:

```cpp
.foreign_keys = {
    {.column = "home_team_id", .target_table = "teams"},
    {.column = "away_team_id", .target_table = "teams"},
}
```

Running `FKDetector::detect` after explicit declarations is harmless: it
skips columns that are already in `foreign_keys`.

---

## Step 3 — Build the graph

```cpp
HeteroGraph graph = GraphBuilder::build(db);
graph.print_summary();
```

For every FK relationship it finds, `GraphBuilder` emits two edge types:

```
orders --[user_id]--> users          (forward)
users  --[rev_user_id]--> orders     (reverse)
```

The reverse edges let a user node aggregate information from all its orders
during message passing.

---

## Step 4 — Define the task

This is where all the thought goes. A task has three components:

**A. The target table and column**

The table that contains one row per prediction unit. The column must be
`NUMERICAL` and will be used as the label.

**B. The label transform**

How the raw column value is converted to a training label.

| Kind | When to use | Example |
|---|---|---|
| `Threshold` | Binary classification on a numerical signal | rating >= 4 → positive |
| `Normalize` | Regression | predict the raw value after z-scoring |
| `Buckets` | Multiclass | outcome ∈ {0,1,2} from goals scored |

**C. The split strategy**

`Temporal` — sort rows by `time_col` and split 70/15/15. Use this whenever
the table has a meaningful time ordering. It prevents future leakage.

`Random` — deterministic Fisher-Yates shuffle. Use this when there is no
meaningful time ordering (e.g. a synthetic task table you materialised).

---

## Task patterns

### Pattern A — label already exists in a table

```cpp
TaskSpec spec;
spec.target_table  = "ratings";
spec.target_column = "rating";
spec.task_type     = TaskSpec::TaskType::BinaryClassification;

spec.label_transform.kind      = LabelTransform::Kind::Threshold;
spec.label_transform.threshold = 4.0f;
spec.label_transform.inclusive = true;

spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
spec.split_time_col = "timestamp";
```

### Pattern B — label must be injected

```cpp
static void inject_churn_label(Database& db) {
    // find the latest timestamp in transactions
    // compute cutoff = max_ts - 7 * 86400
    // for each customer: label = 1 if last purchase <= cutoff, else 0
    // add Column("will_churn", NUMERICAL) to the customers table
}

inject_churn_label(db);

TaskSpec spec;
spec.target_table  = "customers";
spec.target_column = "will_churn";
spec.task_type     = TaskSpec::TaskType::BinaryClassification;

spec.label_transform.kind      = LabelTransform::Kind::Threshold;
spec.label_transform.threshold = 0.5f;
spec.label_transform.inclusive = true;

spec.split_strategy = TaskSpec::SplitStrategy::Random;
```

**Critical:** after `build_split` reads the label column, flip its type to
`TEXT` so `HeteroEncoder` does not encode it as a feature:

```cpp
TaskSplit split = spec.build_split(db);
db.get_table("customers").get_column("will_churn").type = ColumnType::TEXT;
```

### Pattern C — the task table does not exist in the database

```cpp
static void build_user_ad_candidates(Database& db) {
    // 1. find cutoff = max(ViewDate) - 4 * 86400
    // 2. collect positive (UserID, AdID) pairs: appeared in VisitStream >= cutoff
    // 3. for each active user: keep all positives + negatives
    // 4. store as Table "UserAdCandidates" with FK → UserInfo, FK → AdsInfo
}

build_user_ad_candidates(db);
HeteroGraph graph = GraphBuilder::build(db);

db.get_table("UserAdCandidates").get_column("UserID").type = ColumnType::TEXT;
db.get_table("UserAdCandidates").get_column("AdID").type   = ColumnType::TEXT;

TaskSpec spec;
spec.target_table   = "UserAdCandidates";
spec.target_column  = "will_visit";
spec.task_type      = TaskSpec::TaskType::BinaryClassification;
spec.split_strategy = TaskSpec::SplitStrategy::Random;
```

---

## Step 5 — Train

```cpp
TrainConfig cfg;
cfg.channels   = 64;
cfg.gnn_layers = 2;
cfg.hidden     = 64;
cfg.dropout    = 0.3f;
cfg.lr         = 3e-4f;
cfg.pos_weight = 1.f;    // auto-rebalanced from training data
cfg.epochs     = 20;
cfg.batch_size = 0;      // 0 = full batch; set to 4096+ for large tables
cfg.task       = spec;

Trainer trainer(cfg, db, graph);
trainer.fit(split, db, graph);
```

**Choosing `gnn_layers`.** Each layer expands the receptive field by one hop.
2 layers is correct for the vast majority of schemas. Adding more causes
over-smoothing: all node embeddings converge toward the same global average.

**Choosing `channels`.** 64 is a good default. Increase to 128 for large
datasets with complex schemas. Decrease to 32 for small datasets.

**`pos_weight`.** Leave it at `1.f`. `Trainer::fit` automatically computes
`pos_weight = n_neg / n_pos` from the training split.

---

## Inference

```cpp
// Score all rows
std::vector<float> preds = trainer.predict_all(db, graph);

// Filter and aggregate
spec.inference_mode = TaskSpec::InferenceMode::RowBased;
spec.inference_agg  = TaskSpec::AggType::Fraction;
InferenceFilter f;
f.column = "user_id"; f.op = "="; f.value = "42";
spec.inference_filters.push_back(f);
auto result = spec.apply_inference(db, preds);

// Entity synthesis
float p = trainer.synthesize_prediction(
    {{"user_id", "42"}, {"product_id", "1337"}},
    db, graph);
```

---

## Full example — regression on order value

```cpp
std::unordered_map<std::string, TableSchema> schemas = {
    {"users", {
        .pkey_col = "user_id",
        .columns  = {
            {.name = "country",     .type = ColumnType::CATEGORICAL},
            {.name = "signup_date", .type = ColumnType::TIMESTAMP},
        }
    }},
    {"products", {
        .pkey_col = "product_id",
        .columns  = {
            {.name = "name",     .type = ColumnType::TEXT},
            {.name = "category", .type = ColumnType::CATEGORICAL},
        }
    }},
    {"orders", {
        .pkey_col     = "order_id",
        .time_col     = "created_at",
        .foreign_keys = {
            {.column = "user_id",    .target_table = "users"},
            {.column = "product_id", .target_table = "products"},
        },
        .columns = {}
    }},
};

Database    db    = CSVLoader::load_database("./my-data", "mydb", schemas);
FKDetector::detect(db);
HeteroGraph graph = GraphBuilder::build(db);

TaskSpec spec;
spec.target_table             = "orders";
spec.target_column            = "total_value";
spec.task_type                = TaskSpec::TaskType::Regression;
spec.label_transform.kind     = LabelTransform::Kind::Normalize;
spec.split_strategy           = TaskSpec::SplitStrategy::Temporal;
spec.split_time_col           = "created_at";

TaskSplit split = spec.build_split(db);
db.get_table("orders").get_column("total_value").type = ColumnType::TEXT;

TrainConfig cfg;
cfg.channels = 64; cfg.gnn_layers = 2; cfg.hidden = 64;
cfg.dropout = 0.3f; cfg.lr = 3e-4f; cfg.epochs = 30;
cfg.task = spec;

Trainer trainer(cfg, db, graph);
trainer.fit(split, db, graph);

std::vector<float> preds = trainer.predict_all(db, graph);
```

---

## Common mistakes

**Forgetting to hide the label column.** If the target column is left as
`NUMERICAL` after `build_split`, `HeteroEncoder` encodes it as a node feature.
The model learns to copy the input to the output and achieves near-perfect
training accuracy. Always flip the label column to `TEXT` after `build_split`.
In Python this is handled automatically — the label column is derived in SQL
and never appears in the feature tables.

**Using a temporal split on a synthetic task table.** If you materialised a
task table where the temporal logic is already baked into the label, use
`Random` split. A temporal split on such a table splits by the order rows were
generated, which is arbitrary.

**Leaving FK integer IDs as NUMERICAL on a synthetic task table.** Override
them to `TEXT` or flip after `GraphBuilder` so the encoder does not z-score
raw IDs and use them as features.

**Declaring `gnn_layers > 2`.** More layers cause over-smoothing on typical
relational schemas. The signal you need is almost always within 2 hops.

---

## Building RelML

### Prerequisites

| Dependency | Required | Purpose |
|---|---|---|
| CMake >= 3.20 | yes | build system |
| C++20 compiler | yes | clang++ 14+ or g++ 12+ |
| libcurl | yes | Agent API calls |
| OpenMP | recommended | multi-threaded training |
| OpenBLAS | optional | faster matrix multiply |
| Python 3.9+ devel | optional | Python bindings |

**macOS**

```bash
brew install cmake curl libomp openblas python
```

**Ubuntu / Debian**

```bash
sudo apt install cmake libcurl4-openssl-dev libomp-dev libopenblas-dev python3-dev
```

**RHEL 8**

```bash
sudo yum install cmake libcurl-devel libomp-devel openblas-devel python39-devel
```

### Configure and build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

To build only the Python extension:

```bash
make -j$(nproc) _relml_core
```

### Running an example task

```bash
./ml1m_rating_classification ../data/ml-1m-data
./avito_user_ad_visit ../data/rel-avito-data
./avito_user_ad_visit ../data/rel-avito-data 38950 1938326
./pl_outcome ../data/premiere-league-data
```

### Running the gradient check

```bash
./test_grad_check
# expected output: "All gradients correct."
```

---

## Multi-threaded training

RelML parallelises the three most expensive operations using OpenMP:

- `Linear::forward` and `Linear::backward` — matrix multiplies over N rows
- `SAGELayer::forward` — neighbour sum accumulation and ReLU
- `Adam::step` — parameter updates across all tensors

```bash
OMP_NUM_THREADS=$(nproc) ./ml1m_rating_classification ../data/ml-1m-data
OMP_NUM_THREADS=4 ./avito_user_ad_visit ../data/rel-avito-data
OMP_NUM_THREADS=1 ./pl_outcome ../data/premiere-league-data
```

**Expected scaling** on 8 physical cores with OpenBLAS:

| Dataset | Channels | Epochs | OMP=1 | OMP=8 |
|---|---|---|---|---|
| MovieLens-1M | 64 | 30 | ~180s | ~35s |
| rel-avito (100k rows) | 64 | 20 | ~90s | ~20s |
| Premier League | 32 | 600 | ~240s | ~55s |