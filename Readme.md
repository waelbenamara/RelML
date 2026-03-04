# RelML

RelML is a C++ implementation of Relational Deep Learning for relational databases. Its purpose is to answer **predictive queries** — questions whose answers are not stored in any table and cannot be derived by any SQL expression, but can be learned from the relational structure of the database itself.


## The Problem: The Ceiling of SQL

A relational database stores entities and their interactions as tables linked by foreign-key constraints. SQL — and more specifically the Select-Project-Join (SPJ) fragment that underlies virtually all query answering — is the language of *retrieval*. It filters rows, joins tables, and computes aggregates over data that already exists.

SPJ has a hard ceiling. It cannot answer questions whose answers are not yet in the database. Consider the following queries:

- *Which customers are likely to stop purchasing in the next 30 days?*
- *Will user 5 enjoy movie 42?*
- *How much will this transaction cost?*

A text-to-SQL system handed the first question can, at best, return customers who have already stopped. It has no mechanism to *predict* who will. The answer to a predictive question is not stored anywhere and cannot be derived by any SQL expression, no matter how complex.

There is a precise relationship between SPJ and prediction that is worth making explicit. **Defining** a prediction task always requires an SPJ query. The task is: take a set of entity rows, attach a label to each, and order them by time. That is a table, and a table is the output of a SELECT statement. We call this the *task table*. Three examples cover the range of questions RelML can answer.

**Binary classification.** Will a given rating be positive (>= 4 stars)?

```sql
SELECT ratingId,
       CAST(rating >= 4 AS INT) AS label
FROM   ratings
ORDER  BY timestamp
```

**Regression.** What is the expected purchase price for this transaction?

```sql
SELECT transactionId,
       price AS label
FROM   transactions
ORDER  BY t_dat
```

**Link prediction.** Will customer u purchase article a in the next 7 days?

```sql
SELECT u.customer_id, a.article_id,
       CAST(EXISTS(
         SELECT 1 FROM transactions t
         WHERE  t.customer_id = u.customer_id
         AND    t.article_id  = a.article_id
         AND    t.t_dat BETWEEN :cutoff AND :cutoff + INTERVAL 7 DAYS
       ) AS INT) AS label
FROM   customer u CROSS JOIN article a
ORDER  BY :cutoff
```

In each case, the task table is entirely within SPJ. What SPJ cannot supply — and what RelML learns — is the label value for held-out or future rows. The label of a future rating is not in the database. Answering it requires generalizing from patterns spread across past ratings, users, and movies simultaneously.

A **predictive query** therefore pairs a task table with a learned model that estimates the label for unseen entity instances, using all relational context available in the database. The context of a rating — the user who gave it, the movie it concerns, other movies the same user has rated, other users who rated the same movie — cannot be expressed in the target row alone. It must be gathered by traversing foreign-key links across the relational graph.


## RelML as a Predictive Layer over Existing Database Engines

The task table — the ordered list of entity rows paired with labels that defines the prediction problem — is a standard SQL query. It can be materialized by any existing database engine: DuckDB, PostgreSQL, SQLite, or any other system that supports SPJ. There is no new database required. The database you already have is sufficient to define and materialize the task.

For example, the rating classification task table is produced by:

```sql
SELECT ratingId,
       CAST(rating >= 4 AS INT) AS label
FROM   ratings
ORDER  BY timestamp
```

Run this query in DuckDB or PostgreSQL and you have the task table as a CSV or an in-memory result. RelML reads it, builds the graph over the full schema, trains the GNN, and produces predictions. The database engine handles what it does well — storage, retrieval, joins, aggregation, label computation — and RelML handles what it cannot: learning from relational structure to answer questions about unseen or future rows.

This means the boundary between the two systems is clean and explicit. Everything that can be expressed as SPJ stays in the database engine. Everything that requires learning from graph structure goes to RelML. The TaskSpec in RelML is simply the C++ materialization of that SQL query — it selects the target table, computes the label transform, and orders rows by the time column, exactly mirroring what the SQL above does.

In a future version, the current C++ database layer — which reimplements CSV loading, type inference, and FK detection — would be replaced by a direct DuckDB connection, so that the task table SQL query is executed natively and its result is handed directly to the encoder without any intermediate representation. The GNN, prediction head, trainer, and optimizer would remain unchanged. The learning stack is already complete; only the data ingestion layer would change.


## How RelML Works

RelML implements the Relational Deep Learning framework described in Robinson et al. (RelBench, NeurIPS 2024) and Hamilton et al. (GraphSAGE, NeurIPS 2017). The pipeline has four stages that run automatically once you define a task.

**Stage 1: Schema and type inference.** Each column is assigned one of four types: NUMERICAL (z-score normalized), CATEGORICAL (one-hot encoded), TIMESTAMP (sin/cos cyclical decomposition), or TEXT (skipped). Types are inferred automatically from the data distribution and can be overridden per column in the schema declaration.

**Stage 2: Graph construction.** Every row in every table becomes a node. Every foreign-key value becomes a directed edge from the source row to the referenced primary-key row. Reverse edges are added automatically so information flows in both directions. The result is a heterogeneous graph where node types correspond to tables and edge types correspond to FK relationships.

**Stage 3: Relational message passing.** A multi-layer Graph SAGE network runs over the graph. Each layer allows every node to aggregate a summary of its neighbors' current embeddings. After L layers, each node's embedding encodes its own features plus the features of its entire L-hop relational neighborhood. A rating node after two layers implicitly encodes: the user's demographics, that user's full rating history, the movie's genre, and how other users have responded to that movie. None of this is written in the rating row itself.

**Stage 4: Prediction head and training.** A small MLP head reads the final embedding of each target node and produces a prediction. All parameters — projection matrices, GNN weights, MLP head — are trained end-to-end with Adam. The best checkpoint on the validation set is restored before test evaluation.

The value of relational context is empirical and substantial. On MovieLens 1M, a flat MLP that joins all three tables into one row per rating achieves a test AUC of 0.574 — barely above chance. RelML with two message-passing layers achieves 0.970. The gap is a direct measurement of what lives two relational hops away and is completely invisible to any single-row model.


## Building the Project

Requirements: a C++20 compiler, CMake >= 3.20, libcurl, and optionally OpenBLAS.

```bash
git clone <repo>
cd relml
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

To build a specific example task:

```bash
make ml1m_rating_classification -j$(nproc)
make hm_churn -j$(nproc)
```


## Writing a Task: Complete Walkthrough

Every task is a standalone `.cpp` file in `src/example_tasks/`. The minimum you need is three things: a schema, a `TaskSpec`, and a `TrainConfig`. RelML handles the rest.

We walk through the MovieLens 1M rating classification task step by step.

**Step 1: Declare the schema.**

The schema tells RelML how to interpret each table. You only need to declare what cannot be inferred automatically: primary keys, time columns, and any column type overrides.

```cpp
std::unordered_map<std::string, TableSchema> schemas = {
    {"users", {
        .pkey_col     = "userId",
        .time_col     = std::nullopt,
        .foreign_keys = {},
        .columns      = {
            {.name = "gender",     .type = ColumnType::CATEGORICAL},
            {.name = "occupation", .type = ColumnType::CATEGORICAL},
            {.name = "zip",        .type = ColumnType::TEXT},
        }
    }},
    {"movies", {
        .pkey_col = "movieId",
        .time_col = std::nullopt,
        .columns  = {
            {.name = "title",  .type = ColumnType::TEXT},
            {.name = "genres", .type = ColumnType::CATEGORICAL},
        }
    }},
    {"ratings", {
        .pkey_col     = "ratingId",
        .time_col     = "timestamp",
        .foreign_keys = {
            {.column = "userId",  .target_table = "users"},
            {.column = "movieId", .target_table = "movies"},
        },
    }},
};
```

**Step 2: Load the database, detect foreign keys, build the graph.**

```cpp
Database    db    = CSVLoader::load_database(data_dir, "ml-1m", schemas);
FKDetector::detect(db);
HeteroGraph graph = GraphBuilder::build(db);
graph.print_summary();
```

After these three lines you have a heterogeneous graph with three node types and four edge types. The graph builder prints a summary showing node counts and edge counts per type.

**Step 3: Define the prediction task.**

The `TaskSpec` is the C++ materialization of the task table SQL query. It specifies what to predict, how to transform the label, how to split the data, and how to run inference.

```cpp
TaskSpec spec;
spec.target_table              = "ratings";
spec.target_column             = "rating";
spec.task_type                 = TaskSpec::TaskType::BinaryClassification;
spec.label_transform.kind      = LabelTransform::Kind::Threshold;
spec.label_transform.threshold = 4.f;
spec.label_transform.inclusive = true;        // rating >= 4 is positive
spec.split_strategy            = TaskSpec::SplitStrategy::Temporal;
spec.split_time_col            = "timestamp";
spec.inference_mode            = TaskSpec::InferenceMode::EntitySynthesis;
spec.entity_refs               = {{"userId", "1"}, {"movieId", "1193"}};

TaskSplit split = spec.build_split(db);
```

`build_split` materializes the task table: it reads the rating column, applies the threshold to produce 0/1 labels, sorts all rows by timestamp, and cuts at 70% and 85% to produce train, val, and test sets.

**Step 4: Configure and train.**

```cpp
TrainConfig cfg;
cfg.channels   = 64;
cfg.gnn_layers = 2;
cfg.hidden     = 64;
cfg.dropout    = 0.3f;
cfg.lr         = 3e-4f;
cfg.epochs     = 30;
cfg.batch_size = 0;     // 0 = full batch
cfg.task       = spec;

Trainer trainer(cfg, db, graph);
trainer.fit(split, db, graph);
```

The trainer runs the full pipeline on every epoch: encode all nodes, run GNN message passing over the full graph, compute loss on the target table rows, backpropagate through the GNN and encoders, update all parameters with Adam. Validation metrics are printed after every epoch. The best checkpoint is saved and restored automatically.

**Step 5: Run inference.**

```cpp
// entity synthesis: predict for a specific (user, movie) pair
// that may have no existing rating row
float prob = trainer.synthesize_prediction(
    {{"userId", "1"}, {"movieId", "1193"}}, db, graph);
std::cout << "P(user 1 likes movie 1193) = " << prob << "\n";

// row-based: score all existing rating rows, report the positive fraction
std::vector<float> all_preds = trainer.predict_all(db, graph);
TaskSpec::InferenceResult result = spec.apply_inference(db, all_preds);
```

That is the complete task. Define the schema. Define the task. Call fit. Learning happens automatically.


## Schema Reference

### TableSchema fields

**pkey_col** — the primary key column name, or `std::nullopt`. Required for any table that other tables point to via a FK. Tables that are pure observation tables and are never themselves FK targets (like ratings or transactions) can have `std::nullopt`.

**time_col** — the time column name for temporal splits, or `std::nullopt`. Must be either a Unix integer stored as NUMERICAL or a date string in YYYY-MM-DD format.

**foreign_keys** — explicit FK declarations as a list of `{column, target_table}` pairs. If omitted, the FKDetector finds them automatically by name matching and value coverage verification. Explicit declaration is safer when coverage is lower than expected or naming conventions are irregular.

**columns** — per-column type overrides. Only columns where automatic inference would give the wrong type need to be listed.

### Column types

**NUMERICAL** — z-score normalized to zero mean and unit variance. Statistics are computed from the training split only to prevent leakage. Use for real-valued measurements: age, price, rating value, duration.

**CATEGORICAL** — one-hot encoded over the training vocabulary. Unseen values at inference time map to the all-zero vector. Use for columns with a bounded set of values: gender, country, occupation code, sales channel.

**TIMESTAMP** — encoded as a 5-dimensional vector: sin and cos of month, sin and cos of day-of-month, and normalized year. The sin/cos representation places calendar time on a circle so December and January are adjacent in feature space. Use for date and datetime columns.

**TEXT** — skipped by the encoder. Use for high-cardinality strings that would produce explosion in one-hot dimensions: names, descriptions, hashed identifiers, zip codes.

If no override is declared, the inferrer applies these rules: all-numeric values become NUMERICAL, ISO 8601 date-shaped values become TIMESTAMP, string columns with unique-value ratio below 5% become CATEGORICAL, everything else becomes TEXT.


## TaskSpec Reference

### task_type

**BinaryClassification** — predicts a probability between 0 and 1. Loss is binary cross-entropy with automatic positive-class reweighting. Use when the outcome has two values: liked/not liked, churned/active, defaulted/did not default.

**Regression** — predicts a continuous value. Loss is mean squared error. The target is normalized during training and denormalized at inference. Use when you want to estimate a magnitude: price, duration, rating value.

**MulticlassClassification** — predicts one of K classes. Loss is cross-entropy. Use when the outcome has more than two discrete values: star tier, price bucket, age group.

### label_transform

**Threshold** — converts raw value to 0 or 1.

```cpp
spec.label_transform.kind      = LabelTransform::Kind::Threshold;
spec.label_transform.threshold = 4.f;
spec.label_transform.inclusive = true;   // >= 4 is positive
```

**Normalize** — z-score normalizes the raw value. Used with Regression. Statistics are filled automatically by `build_split`.

```cpp
spec.label_transform.kind = LabelTransform::Kind::Normalize;
```

**Buckets** — maps raw value to a class index using sorted boundary values. Used with MulticlassClassification.

```cpp
spec.label_transform.kind    = LabelTransform::Kind::Buckets;
spec.label_transform.buckets = {2.f, 3.5f};  // 3 classes: <2, 2-3.5, >3.5
```

### split_strategy

**Temporal** — sorts by time column and cuts at 70% / 85%. This is the correct split whenever the data has timestamps. It simulates deployment: the model sees only past data during training and is evaluated on the future.

**Random** — shuffles with a fixed seed before splitting. Use when there is no meaningful time ordering.

### inference_mode

**EntitySynthesis** — predicts for hypothetical entity combinations that may not exist as rows. Provide `entity_refs` as a map from FK column name to entity ID string. The system looks up the GNN embeddings of the referenced entities, averages them, and passes the pooled vector through the head.

```cpp
spec.inference_mode = TaskSpec::InferenceMode::EntitySynthesis;
spec.entity_refs    = {{"userId", "5"}, {"movieId", "42"}};
```

**RowBased** — scores existing rows and aggregates the predictions. Provide `inference_filters` to restrict which rows are scored and `inference_agg` to combine the scores.

```cpp
spec.inference_mode = TaskSpec::InferenceMode::RowBased;
spec.inference_agg  = TaskSpec::AggType::Fraction;

InferenceFilter f;
f.column = "userId";
f.op     = "=";
f.value  = "42";
spec.inference_filters.push_back(f);
```

### inference_filters

Each filter has three string fields: `column`, `op` (one of `=`, `!=`, `>`, `>=`, `<`, `<=`), and `value`. Multiple filters are ANDed together.

### inference_agg

**None** — return all per-row scores individually.

**Fraction** — fraction of rows with predicted probability above 0.5. Use for *what proportion will churn?*

**Mean** — average predicted value across matching rows. Use for *what is the expected like-probability for user 42?*

**Count** — count of rows predicted positive. Use for *how many transactions will exceed $50?*


## TrainConfig Reference

**channels** — embedding dimensionality for every node in the graph. Primary memory knob. Approximate peak memory:

```
peak_memory = channels x total_rows x 4 bytes x gnn_layers x 6
```

```
channels = 32    safe for datasets with millions of rows on 16 GB RAM
channels = 64    good default for datasets up to ~1M rows
channels = 128   for small datasets under 100K rows with ample RAM
```

**gnn_layers** — number of message-passing rounds. Two layers is the standard. Each additional layer extends the receptive field by one hop but adds proportionally to memory and risks over-smoothing.

**hidden** — hidden layer width in the MLP head. Setting `hidden = channels` is a safe default.

**dropout** — dropout probability in the MLP head during training. Values between 0.2 and 0.5 work for most tasks.

**lr** — Adam learning rate. `3e-4` is a reliable default. Reduce to `1e-4` if training is unstable. Increase to `1e-3` if the loss barely moves in early epochs.

**pos_weight** — positive-class weight in binary cross-entropy. Set to `1.0` to let the system compute it automatically as `n_negative / n_positive`. Essential for imbalanced tasks like churn.

**epochs** — number of training passes. The best validation checkpoint is restored before test evaluation.

**batch_size** — mini-batch size for the MLP head backward pass. `0` uses the full training set. The GNN always operates on the full graph regardless of this setting.


## Injecting Synthetic Labels

Some tasks require a label that does not exist as a column anywhere and must be derived from related tables. The churn task is the canonical example: no table has a `will_churn` column, so it is computed from transaction recency.

The pattern is: compute the label, build a `Column` object with one entry per row in the target table, and add it before calling `build_split`.

```cpp
Column churn_col("will_churn", ColumnType::NUMERICAL);
for (std::size_t i = 0; i < customer.num_rows(); ++i) {
    double label = /* 1.0 if churned, 0.0 if active */;
    churn_col.data.push_back(label);
}

db.get_table("customer").add_column(std::move(churn_col));

spec.target_column = "will_churn";
TaskSplit split = spec.build_split(db);
```

The injected column must be NUMERICAL and must have exactly the same number of rows as the target table. This pattern is the C++ equivalent of a SQL WITH clause: any label expressible as an SPJA query over the database can be injected this way.


## Running the Example Tasks

```bash
cd build && cmake ..

# MovieLens 1M — predict whether a rating is >= 4 stars
make ml1m_rating_classification -j$(nproc)
./ml1m_rating_classification ../ml-1m-data

# H&M — predict which customers will churn in the next 7 days
make hm_churn -j$(nproc)
./hm_churn ../rel-hm-data
```

Expected directory layouts:

```
ml-1m-data/          rel-hm-data/
    users.csv             customer.csv
    movies.csv            article.csv
    ratings.csv           transactions.csv
```


## Adding a New Task

Three steps.

```bash
# 1. create the source file
touch src/example_tasks/my_task.cpp

# 2. add one line to CMakeLists.txt
echo "add_relml_task(my_task  src/example_tasks/my_task.cpp)" >> CMakeLists.txt

# 3. write the task (see walkthrough above)
```

Every task follows the same structure: declare schemas, load and build graph, optionally inject labels, define TaskSpec, call build_split, configure TrainConfig, construct Trainer, call fit.


## Natural Language Interface

For interactive exploration without writing a task file, `test_system` accepts plain-English queries and uses an LLM to parse them into `TaskSpec` objects automatically:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./test_system ../ml-1m-data "What would user 5 rate movie 589?"
./test_system ../ml-1m-data "What fraction of ratings are likely positive?"
./test_system ../ml-1m-data "How does user 42 tend to rate movies?"
```

Trained models are cached in `~/.relml/` indexed by a fingerprint of the task definition. Asking the same question a second time loads the saved weights and skips training.


## Memory Requirements

The dominant cost is GNN activation caches kept for backpropagation.

```
peak_memory = channels x total_rows_across_all_tables x 4 bytes x gnn_layers x 6
```

Concrete figures:

```
ml-1m  (~1M ratings + ~10K users and movies combined):
    channels=64     ~1.5 GB
    channels=128    ~6   GB

rel-hm (~15M transactions + ~1.5M customers + ~105K articles):
    channels=32     ~6   GB
    channels=64     ~24  GB
    channels=128    ~96  GB
```

If you run out of memory, reduce `channels` first. Reducing `gnn_layers` from 2 to 1 halves the cache at the expense of shorter-range relational reasoning.


## Future Directions

### DuckDB integration

The current database layer — CSVLoader, TypeInferrer, FKDetector, Column, Table, Database — is approximately 800 lines of C++ that reimplements functionality available natively in DuckDB. A future version of RelML would replace this layer entirely:

```cpp
duckdb::DuckDB db;
duckdb::Connection con(db);
con.Query("CREATE TABLE ratings AS SELECT * FROM read_csv_auto('ratings.csv')");
```

Type inference, CSV parsing, null handling, and FK detection all become SQL expressions. Synthetic label injection becomes a WITH clause. The HeteroEncoder would consume DuckDB result chunks directly instead of the custom Column struct.

The GNN, prediction head, trainer, and optimizer would remain unchanged. The boundary between the relational layer and the learning layer is clean: the database produces typed feature matrices, the GNN consumes them. Swapping the database backend requires no changes to any learning code.

More importantly, this would make RelML deployable as a true database extension. The PREDICT verb would be registered as a custom function in DuckDB or PostgreSQL, allowing predictive queries to be issued directly alongside relational queries with no external tooling.

### Neighbor sampling

The current GNN runs full-graph message passing every epoch, so peak memory scales with total row count across all tables. For datasets with tens of millions of rows this is prohibitive. Neighbor sampling — loading only the k-hop neighborhood of each batch of target nodes rather than the full graph — would reduce peak memory from O(total nodes) to O(batch size x fanout^L), making very large relational databases trainable on modest hardware.

### Text column encoding

Columns of type TEXT are currently skipped. Encoding them with a pretrained language model would allow RelML to use movie titles, product descriptions, and customer notes as features without manual preprocessing, which is particularly valuable for article and product catalog tables.


## References

Robinson, J. et al. *RelBench: A Benchmark for Deep Learning on Relational Databases*. NeurIPS 2024.

Hamilton, W. L., Ying, Z., Leskovec, J. *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS 2017.