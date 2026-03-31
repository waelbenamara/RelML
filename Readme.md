# Creating a Task in RelML

This document explains how to take a raw relational database, define a
prediction task, and train a model. Every example in `src/example_tasks/`
follows this exact pattern.

---

## The five steps

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

The simplest case. A column in an existing table is directly the label.

```cpp
// Predict whether a rating is >= 4 stars
TaskSpec spec;
spec.target_table  = "ratings";
spec.target_column = "rating";
spec.task_type     = TaskSpec::TaskType::BinaryClassification;

spec.label_transform.kind      = LabelTransform::Kind::Threshold;
spec.label_transform.threshold = 4.0f;
spec.label_transform.inclusive = true;   // >= 4

spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
spec.split_time_col = "timestamp";
```

### Pattern B — label must be injected

The label does not exist in the database. You compute it and add it as a new
column before calling `build_split`.

```cpp
// Predict whether a customer will churn (no purchase in last 7 days)
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

Sometimes the right prediction unit is a pair or combination that has no
corresponding table. You materialise it from an SPJA query on the existing
tables, add it to the database, then treat it like any other table.

```cpp
// Predict whether user U will visit ad A in the next 4 days.
// This requires a table UserAdCandidates(UserID, AdID, will_visit)
// that does not exist in the raw database.

static void build_user_ad_candidates(Database& db) {
    // 1. find cutoff = max(ViewDate) - 4 * 86400
    // 2. collect positive (UserID, AdID) pairs: appeared in VisitStream >= cutoff
    // 3. for each active user: keep all positives + NEG_RATIO negatives
    // 4. store as Table "UserAdCandidates" with FK → UserInfo, FK → AdsInfo
    //    UserID and AdID as NUMERICAL (so GraphBuilder can resolve FKs)
    //    will_visit as NUMERICAL (0 or 1)
}

build_user_ad_candidates(db);
HeteroGraph graph = GraphBuilder::build(db);

// After graph is built: flip FK columns to TEXT so encoder skips them.
// Their only role was to carry edges. The integer IDs themselves are meaningless.
db.get_table("UserAdCandidates").get_column("UserID").type = ColumnType::TEXT;
db.get_table("UserAdCandidates").get_column("AdID").type   = ColumnType::TEXT;

TaskSpec spec;
spec.target_table  = "UserAdCandidates";
spec.target_column = "will_visit";
spec.task_type     = TaskSpec::TaskType::BinaryClassification;
spec.split_strategy = TaskSpec::SplitStrategy::Random;
// temporal logic already baked into the label derivation (the cutoff)
```

The key insight: the synthetic table has no self-features (FK columns are
skipped). All signal flows through the FK edges during GNN message passing.
For a row `(user U, ad A)`, the GNN embedding is dominated by
`W_neigh_u * h_UserInfo[U] + W_neigh_a * h_AdsInfo[A]`. This means the model
can score any `(user, ad)` pair at inference time, including ones the user has
never interacted with.

---

## Step 5 — Train

```cpp
TrainConfig cfg;
cfg.channels   = 64;     // embedding dimension for all nodes
cfg.gnn_layers = 2;      // 2 is correct for most schemas
cfg.hidden     = 64;     // MLP hidden layer size
cfg.dropout    = 0.3f;   // set higher (0.5) for small datasets
cfg.lr         = 3e-4f;
cfg.pos_weight = 1.f;    // auto-rebalanced from training data
cfg.epochs     = 20;
cfg.batch_size = 0;      // 0 = full batch; set to 4096+ for large tables
cfg.task       = spec;

Trainer trainer(cfg, db, graph);
trainer.fit(split, db, graph);
```

**Choosing `gnn_layers`.**
Each layer expands the receptive field by one hop. 2 layers is correct for
the vast majority of schemas. The useful signal is almost always within 2 hops
(e.g. order → user → user features). Adding more layers causes over-smoothing:
all node embeddings converge toward the same global average.

**Choosing `channels`.**
64 is a good default. Increase to 128 for large datasets with complex schemas.
Decrease to 32 for small datasets prone to overfitting.

**`pos_weight`.**
Leave it at `1.f`. `Trainer::fit` automatically computes
`pos_weight = n_neg / n_pos` from the training split and applies it to the
BCE loss. For multiclass tasks it computes per-class inverse-frequency weights.

---

## Inference

After training, there are three ways to query the model.

**Score all rows** — returns one prediction per row in the target table:

```cpp
std::vector<float> preds = trainer.predict_all(db, graph);
```

**Filter and aggregate** — apply row filters and return an aggregate:

```cpp
spec.inference_mode = TaskSpec::InferenceMode::RowBased;
spec.inference_agg  = TaskSpec::AggType::Fraction;  // fraction of positives

InferenceFilter f;
f.column = "user_id";
f.op     = "=";
f.value  = "42";
spec.inference_filters.push_back(f);

auto result = spec.apply_inference(db, preds);
// result.aggregate contains the fraction
```

**Entity synthesis** — score a specific combination of entities, including
ones that do not exist as rows in any table:

```cpp
float p = trainer.synthesize_prediction(
    {{"user_id", "42"}, {"product_id", "1337"}},
    db, graph);
```

This looks up `h_UserInfo[42]` and `h_Product[1337]` from the GNN, mean-pools
them, and passes the result through the MLP head. It works for any entity in
the respective tables, including entities with no prior interactions — they
have embeddings derived from their static features and their connections in
the graph.

---

## Full example — regression on order value

```cpp
// Predict the total value of an order.

std::unordered_map<std::string, TableSchema> schemas = {
    {"users", {
        .pkey_col = "user_id",
        .columns  = {
            {.name = "country",    .type = ColumnType::CATEGORICAL},
            {.name = "signup_date",.type = ColumnType::TIMESTAMP},
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
        .columns = {}  // total_value inferred NUMERICAL, created_at inferred TIMESTAMP
    }},
};

Database    db    = CSVLoader::load_database("./my-data", "mydb", schemas);
FKDetector::detect(db);
HeteroGraph graph = GraphBuilder::build(db);

TaskSpec spec;
spec.target_table  = "orders";
spec.target_column = "total_value";
spec.task_type     = TaskSpec::TaskType::Regression;

// Normalize fits mean/std on training rows and inverse-transforms predictions
spec.label_transform.kind = LabelTransform::Kind::Normalize;

spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
spec.split_time_col = "created_at";

TaskSplit split = spec.build_split(db);

// Hide label from encoder
db.get_table("orders").get_column("total_value").type = ColumnType::TEXT;

TrainConfig cfg;
cfg.channels   = 64;
cfg.gnn_layers = 2;
cfg.hidden     = 64;
cfg.dropout    = 0.3f;
cfg.lr         = 3e-4f;
cfg.epochs     = 30;
cfg.batch_size = 0;
cfg.task       = spec;

Trainer trainer(cfg, db, graph);
trainer.fit(split, db, graph);

// Predictions are returned in the original scale (inverse of z-score)
std::vector<float> preds = trainer.predict_all(db, graph);
```

---

## Full example — multiclass classification

```cpp
// Predict the outcome of a football match: 0 = home win, 1 = draw, 2 = away win.
// The outcome column does not exist — we inject it from goals_home and goals_away.

static void inject_outcome(Database& db) {
    Table& games = db.get_table("games");
    Column outcome("outcome", ColumnType::NUMERICAL);
    for (std::size_t i = 0; i < games.num_rows(); ++i) {
        double gh = /* read goals_home[i] */;
        double ga = /* read goals_away[i] */;
        outcome.data.push_back(gh > ga ? 0.0 : gh == ga ? 1.0 : 2.0);
    }
    games.add_column(std::move(outcome));
}

inject_outcome(db);

TaskSpec spec;
spec.target_table  = "games";
spec.target_column = "outcome";
spec.task_type     = TaskSpec::TaskType::MulticlassClassification;

// Buckets {0.5, 1.5} map raw values 0/1/2 to classes 0/1/2
spec.label_transform.kind    = LabelTransform::Kind::Buckets;
spec.label_transform.buckets = {0.5f, 1.5f};

spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
spec.split_time_col = "date";

TaskSplit split = spec.build_split(db);
db.get_table("games").get_column("outcome").type = ColumnType::TEXT;
```

---

## Common mistakes

**Forgetting to hide the label column.** If the target column is left as
`NUMERICAL` after `build_split`, `HeteroEncoder` encodes it as a node feature.
The model learns to copy the input to the output and achieves near-perfect
training accuracy. Always flip the label column to `TEXT` after
`build_split` returns.

**Using a temporal split on a synthetic task table.** If you materialised a
task table where the temporal logic is already baked into the label (a cutoff
separates training evidence from the label), use `Random` split. A temporal
split on such a table splits by the order rows were generated, which is
arbitrary.

**Leaving FK integer IDs as NUMERICAL on a synthetic task table.** A table
like `UserAdCandidates(UserID, AdID, will_visit)` has only FK columns and a
label. If `UserID` and `AdID` are left as `NUMERICAL`, the encoder z-scores
the raw IDs and includes them as features. The model learns ID correlations
that do not generalise. Override them to `TEXT` or flip after `GraphBuilder`.

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

**macOS**

```bash
brew install cmake curl libomp openblas
```

**Ubuntu / Debian**

```bash
sudo apt install cmake libcurl4-openssl-dev libomp-dev libopenblas-dev
```

---

### Configure and build

```bash
# from the project root
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)        # Linux
make -j$(sysctl -n hw.logicalcpu)   # macOS
```

`-DCMAKE_BUILD_TYPE=Release` enables `-O3` optimisations. Without it the
build defaults to `Debug` which is 5-10x slower for training.

If CMake finds OpenBLAS it prints `BLAS found: ...` and compiles with
`-DRELML_USE_BLAS`. If it finds OpenMP it links against it automatically.
Both are detected without any flag from you.

To build a specific target only:

```bash
make -j$(nproc) ml1m_rating_classification
make -j$(nproc) avito_user_ad_visit
make -j$(nproc) test_grad_check
```

---

### Running an example task

```bash
# from the build directory
./ml1m_rating_classification ../data/ml-1m-data

./avito_user_ad_visit ../data/rel-avito-data

# point query: will user 38950 visit ad 1938326 in the next 4 days?
./avito_user_ad_visit ../data/rel-avito-data 38950 1938326

./pl_outcome ../data/premiere-league-data
```

Data directories are passed as the first argument. The binary looks for
CSV files directly inside that directory. In this repository, datasets live
under `data/` (for example `data/ml-1m-data`, `data/rel-avito-data`); from
`build/` use paths like `../data/ml-1m-data`.

---

### Running the gradient check

Run this once after any change to a backward pass to verify correctness:

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

No code change is needed. The number of threads is controlled at runtime via
the standard OpenMP environment variable:

```bash
# use all available cores (default if OMP_NUM_THREADS is not set)
OMP_NUM_THREADS=$(nproc) ./ml1m_rating_classification ../data/ml-1m-data

# use 4 threads
OMP_NUM_THREADS=4 ./avito_user_ad_visit ../data/rel-avito-data

# single-threaded (useful for reproducible timing benchmarks)
OMP_NUM_THREADS=1 ./pl_outcome ../data/premiere-league-data
```

**Checking that OpenMP is active.**
If CMake printed `Found OpenMP` during configure, it is active. To confirm
at runtime:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -i openmp
# should print: "Found OpenMP: TRUE ..."
```

If OpenMP was not found, all `#pragma omp parallel for` directives compile
away silently and training runs single-threaded without errors.

**BLAS acceleration.**
When OpenBLAS is present, `Linear::forward` and `Linear::backward` use
`cblas_sgemm` instead of the scalar fallback. This is the single largest
speedup for large channel dimensions (128+). For channels=64 the improvement
is modest. For channels=128 on MovieLens-1M it roughly halves epoch time.

To verify BLAS is active:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -i blas
# should print: "BLAS found: /path/to/libopenblas.dylib" or similar
```

**Expected scaling.**
On a machine with 8 physical cores and OpenBLAS:

| Dataset | Channels | Epochs | OMP=1 | OMP=8 |
|---|---|---|---|---|
| MovieLens-1M | 64 | 30 | ~180s | ~35s |
| rel-avito (100k rows) | 64 | 20 | ~90s | ~20s |
| Premier League | 32 | 600 | ~240s | ~55s |

The speedup is sublinear because the encoder and GNN forward passes
parallelise well but the sequential scatter-gather in `mean_aggregate`
(which cannot be parallelised due to write conflicts) becomes the bottleneck
at high thread counts.

---