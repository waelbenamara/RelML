// src/example_tasks/sunnyside_demand.cpp
//
// Task   : regression — predict next-week unit order count for each product
//          target = product_week_learning.demand_next_week
//                   (derived from the pre-computed order_count_next_week column)
//
// Dataset: The Sunny Side café, Paris
//          Square POS exports enriched with weekly Paris weather data.
//          Located in data/sunny-side-data/database_sunnyside/.
//
// Label derivation
//   order_count_next_week is pre-computed in product_week_learning.csv.
//   It is declared TEXT in the schema so the encoder never uses it as a
//   feature. inject_demand_labels() reads it back via get_text() and creates
//   a NUMERICAL column "demand_next_week" used as the regression target.
//   Rows where the value is null (the most recent week for each product,
//   where the future is unknown) are automatically excluded from
//   train/val/test by TaskSpec::build_split.
//
// Equivalent SQL
//
//   ALTER TABLE product_week_learning
//       ADD COLUMN demand_next_week FLOAT;
//   UPDATE product_week_learning
//       SET demand_next_week = CAST(order_count_next_week AS FLOAT)
//       WHERE order_count_next_week IS NOT NULL;
//
// Graph structure
//
//   product_week_learning  --[product_id]--> dim_product
//   fact_product_week_orders --[product_id]--> dim_product
//   (plus reverse edges for bidirectional message passing)
//
//   dim_date is intentionally excluded: calendar and weather features are
//   already aggregated into product_week_learning (iso_year, iso_week,
//   temp_max_week_avg, precip_week_sum, etc.), so adding dim_date would
//   be redundant.
//
//   With 2 GNN layers the information flow is:
//     Layer 1  dim_product absorbs all historical weekly order records from
//              fact_product_week_orders. After this layer each product node
//              carries an embedding informed by its full sales history.
//     Layer 2  product_week_learning nodes aggregate from the now-enriched
//              product embeddings. The model can learn "cappuccino follows a
//              similar seasonal pattern to allongé" style cross-product
//              signals implicitly from the graph structure.
//   The lag and weather features on product_week_learning nodes are encoded
//   directly by HeteroEncoder and serve as the primary predictive signal.
//
// Split
//   Temporal by week_start_date (70 / 15 / 15).
//   Train on the oldest weeks, validate and test on progressively more
//   recent ones. This mirrors real deployment: you always predict the next
//   unknown week forward in time.
//
// Leakage notes
//   * order_count (current week) is safe — it is observed before you
//     predict the following week.
//   * order_count_lag1..4 are the four prior-week counts — all safe.
//   * temp_max_week_avg, temp_min_week_avg, precip_week_sum, wind_week_max
//     are averages over the CURRENT week. For a strict future-week forecast
//     you would substitute next-week weather predictions. For this
//     demonstration, current-week weather is an acceptable proxy signal.
//   * order_count_next_week is declared TEXT — the encoder skips it.
//
// Inference
//   Row-based: predict demand_next_week for every row in
//   product_week_learning. The final section reports the top 10 products
//   by mean predicted next-week demand.

#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/training/Trainer.h"
#include "relml/training/TaskSpec.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace relml;

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, TableSchema> sunnyside_schemas() {
    return {

        // ── dim_product ──────────────────────────────────────────────────
        // Static product catalogue: one row per menu item.
        //   item_name  : free-text, too many unique values → TEXT (skip)
        //   category   : e.g. "CAFFEINE PLEASE", "CAKES" — declared TEXT
        //                so it can be read back for inference display;
        //                product-level category signal enters the model
        //                through sku (CATEGORICAL) and through GNN message
        //                passing from fact_product_week_orders.
        //   sku        : product code, low-cardinality → CATEGORICAL
        {"dim_product", {
            .pkey_col     = "product_id",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                {.name = "item_name", .type = ColumnType::TEXT},
                {.name = "category",  .type = ColumnType::TEXT},
                {.name = "sku",       .type = ColumnType::CATEGORICAL},
            }
        }},

        // ── fact_product_week_orders ──────────────────────────────────────
        // Historical weekly demand records — one row per (product × week).
        // These are NOT the target rows; they supply historical order volume
        // signal to dim_product nodes through GNN message passing.
        // No single-column PK; it is never a FK target.
        //
        // week_start is a date string ("YYYY-MM-DD"); it is stored as a
        // TIMESTAMP for time-awareness. We do NOT declare a FK from it to
        // dim_date because dim_date is excluded from this graph.
        {"fact_product_week_orders", {
            .pkey_col     = std::nullopt,
            .time_col     = "week_start",
            .foreign_keys = {
                {.column = "product_id", .target_table = "dim_product"},
            },
            .columns = {
                // iso_year / iso_week: treat as CATEGORICAL so the encoder
                // one-hot encodes them (year 2026 is not "larger" than 2025)
                {.name = "iso_year",     .type = ColumnType::CATEGORICAL},
                {.name = "iso_week",     .type = ColumnType::CATEGORICAL},
                // order_count: auto-inferred NUMERICAL — no override needed
            }
        }},

        // ── product_week_learning (TARGET TABLE) ─────────────────────────
        // One row per (product × observed week). The regression target is
        // demand_next_week (injected below from order_count_next_week).
        //
        // order_count_next_week is declared TEXT so HeteroEncoder skips it.
        // inject_demand_labels() reads it back via get_text() and creates
        // the "demand_next_week" NUMERICAL column.
        //
        // All lag columns (order_count, lag1..4) and weekly weather averages
        // (temp_max_week_avg, temp_min_week_avg, precip_week_sum,
        //  wind_week_max) are auto-inferred as NUMERICAL and kept as input
        // features — they carry the strongest predictive signal.
        {"product_week_learning", {
            .pkey_col     = std::nullopt,   // composite key: (product_id, iso_year, iso_week)
            .time_col     = "week_start_date",
            .foreign_keys = {
                {.column = "product_id", .target_table = "dim_product"},
            },
            .columns = {
                {.name = "iso_year",              .type = ColumnType::CATEGORICAL},
                {.name = "iso_week",              .type = ColumnType::CATEGORICAL},
                // ── TARGET — encoder must never see this ──────────────────
                {.name = "order_count_next_week", .type = ColumnType::TEXT},
                // lag features and weather aggregates are auto-inferred
                // as NUMERICAL (order_count, order_count_lag1..4,
                // temp_max_week_avg, temp_min_week_avg, precip_week_sum,
                // wind_week_max)
            }
        }},
    };
}

// ---------------------------------------------------------------------------
// inject_demand_labels
//
// Reads order_count_next_week (stored as TEXT to avoid encoder leakage)
// and adds a NUMERICAL column "demand_next_week" to product_week_learning.
// Rows where the value is missing or empty (the last known week per product,
// where no future is available) receive a null label and are excluded from
// all splits by TaskSpec::build_split.
// ---------------------------------------------------------------------------
static void inject_demand_labels(Database& db) {
    Table&        tbl      = db.get_table("product_week_learning");
    const Column& raw_col  = tbl.get_column("order_count_next_week");

    Column label_col("demand_next_week", ColumnType::NUMERICAL);
    label_col.data.reserve(tbl.num_rows());

    std::size_t n_valid = 0;
    std::size_t n_null  = 0;

    for (std::size_t i = 0; i < tbl.num_rows(); ++i) {
        if (raw_col.is_null(i)) {
            label_col.data.push_back(std::monostate{});
            ++n_null;
            continue;
        }
        const std::string& s = raw_col.get_text(i);
        if (s.empty()) {
            label_col.data.push_back(std::monostate{});
            ++n_null;
            continue;
        }
        try {
            label_col.data.push_back(std::stod(s));
            ++n_valid;
        } catch (...) {
            label_col.data.push_back(std::monostate{});
            ++n_null;
        }
    }

    tbl.add_column(std::move(label_col));

    std::cout << "  Labeled rows (future known) : " << n_valid << "\n"
              << "  Null rows (no future week)  : " << n_null  << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1)
        ? argv[1]
        : "./data/sunny-side-data/database_sunnyside";

    // -----------------------------------------------------------------------
    // Load tables
    // We load only the three tables needed for this task.
    // fact_transactions, transaction_payments, etc. are skipped because they
    // operate at a finer grain and add no signal beyond what is already
    // aggregated into product_week_learning and fact_product_week_orders.
    // -----------------------------------------------------------------------
    std::cout << "Loading Sunny Side database from: " << data_dir << "\n";
    Database db("sunny-side");

    const auto schemas = sunnyside_schemas();

    auto load_one = [&](const std::string& csv_name,
                        const std::string& table_name) {
        std::string path = data_dir + "/" + csv_name;
        std::cout << "  " << csv_name << " ... ";
        std::cout.flush();
        auto it = schemas.find(table_name);
        Table t = (it != schemas.end())
            ? CSVLoader::load_table(path, table_name, it->second)
            : CSVLoader::load_table(path, table_name, {});
        std::cout << t.num_rows() << " rows\n";
        db.add_table(std::move(t));
    };

    load_one("dim_product.csv",              "dim_product");
    load_one("fact_product_week_orders.csv", "fact_product_week_orders");
    load_one("product_week_learning.csv",    "product_week_learning");

    // -----------------------------------------------------------------------
    // FK auto-detection (sanity check)
    // The product_id → dim_product FK is declared in the schema.
    // FKDetector may re-confirm it; it will not overwrite declared FKs.
    // -----------------------------------------------------------------------
    std::cout << "\nRunning FK detector (sanity check)...\n";
    auto detected = FKDetector::detect(db);
    for (const auto& fk : detected)
        std::cout << "  auto-detected: " << fk.src_table << "." << fk.src_column
                  << " -> " << fk.dst_table
                  << "  (coverage " << fk.coverage * 100.f << "%)\n";
    if (detected.empty())
        std::cout << "  (none beyond explicitly declared FKs)\n";

    // -----------------------------------------------------------------------
    // Inject demand labels
    // -----------------------------------------------------------------------
    std::cout << "\nInjecting demand labels from order_count_next_week...\n";
    inject_demand_labels(db);

    // -----------------------------------------------------------------------
    // Build heterogeneous graph
    //
    // Edge types produced:
    //   product_week_learning  --[product_id]--> dim_product  (and reverse)
    //   fact_product_week_orders --[product_id]--> dim_product (and reverse)
    // -----------------------------------------------------------------------
    std::cout << "\nBuilding heterogeneous graph...\n";
    HeteroGraph graph = GraphBuilder::build(db);
    graph.print_summary();

    // -----------------------------------------------------------------------
    // TaskSpec
    //
    // target_table  = "product_week_learning"
    // target_column = "demand_next_week" (injected NUMERICAL column above)
    // task_type     = Regression
    //
    // label_transform = Normalize:
    //   Trainer::fit() computes mean and std of training labels and
    //   normalises to zero mean / unit variance before backprop. Predicted
    //   outputs are denormalised for RMSE / MAE / R² evaluation.
    //
    // split_strategy = Temporal on "week_start_date":
    //   Train on the oldest 70% of weeks; val on the next 15%; test on the
    //   most recent 15%. Ensures the model is evaluated on genuinely unseen
    //   future demand.
    // -----------------------------------------------------------------------
    TaskSpec spec;
    spec.target_table  = "product_week_learning";
    spec.target_column = "demand_next_week";
    spec.task_type     = TaskSpec::TaskType::Regression;

    spec.label_transform.kind = LabelTransform::Kind::Normalize;

    spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
    spec.split_time_col = "week_start_date";

    spec.inference_mode = TaskSpec::InferenceMode::RowBased;
    spec.inference_agg  = TaskSpec::AggType::Mean;

    std::cout << "\nBuilding temporal train/val/test split...\n";
    TaskSplit split = spec.build_split(db);
    std::cout << "  train: " << split.train.size() << " rows\n"
              << "  val  : " << split.val.size()   << " rows\n"
              << "  test : " << split.test.size()  << " rows\n";

    // -----------------------------------------------------------------------
    // TrainConfig
    //
    // channels=32  : small café dataset (~985 target rows, ~216 products).
    //                A 128-dim embedding would overfit immediately.
    // gnn_layers=2 : sufficient for the 2-hop path
    //                fact_product_week_orders → dim_product → product_week_learning.
    // batch_size=0 : full-batch — the dataset is tiny (< 1 000 target rows).
    // epochs=150   : more epochs compensate for the small batch size and the
    //                normalised label range [~-2, ~+2].
    // dropout=0.2  : light regularisation; the dataset is small.
    // -----------------------------------------------------------------------
    TrainConfig cfg;
    cfg.channels   = 32;
    cfg.gnn_layers = 2;
    cfg.hidden     = 64;
    cfg.dropout    = 0.2f;
    cfg.lr         = 3e-4f;
    cfg.pos_weight = 1.f;   // unused for regression
    cfg.epochs     = 500;
    cfg.batch_size = 0;     // full-batch
    cfg.task       = spec;

    std::cout << "\nBuilding Trainer...\n";
    Trainer trainer(cfg, db, graph);

    std::cout << "\nTraining...\n";
    trainer.fit(split, db, graph);

    // -----------------------------------------------------------------------
    // Inference: score every product × week row and report the top 10
    // products by mean predicted next-week demand.
    // -----------------------------------------------------------------------
    std::cout << "\nScoring all product × week rows...\n";
    std::vector<float> all_preds = trainer.predict_all(db, graph);

    const Table&  pwl       = db.get_table("product_week_learning");
    const Table&  dprod     = db.get_table("dim_product");
    const Column& pid_col   = pwl.get_column("product_id");
    const Column& dpk_col   = dprod.get_column("product_id");
    const Column& name_col  = dprod.get_column("item_name");
    const Column& cat_col   = dprod.get_column("category");

    // Build product_id -> (item_name, category) map
    std::unordered_map<int64_t, std::pair<std::string, std::string>> id_to_info;
    for (std::size_t i = 0; i < dprod.num_rows(); ++i) {
        if (dpk_col.is_null(i)) continue;
        int64_t pid = static_cast<int64_t>(dpk_col.get_numerical(i));
        std::string name = name_col.is_null(i) ? "?" : name_col.get_text(i);
        std::string cat  = cat_col.is_null(i)  ? "?" : cat_col.get_text(i);
        id_to_info[pid]  = {name, cat};
    }

    // Accumulate predicted demand per product across all weeks
    std::unordered_map<int64_t, float> pid_sum;
    std::unordered_map<int64_t, int>   pid_cnt;
    for (std::size_t i = 0; i < all_preds.size(); ++i) {
        if (pid_col.is_null(i)) continue;
        int64_t pid = static_cast<int64_t>(pid_col.get_numerical(i));
        pid_sum[pid] += all_preds[i];
        pid_cnt[pid]++;
    }

    // Sort by average predicted demand, descending
    std::vector<std::pair<int64_t, float>> ranked;
    ranked.reserve(pid_sum.size());
    for (const auto& [pid, total] : pid_sum)
        ranked.push_back({pid, total / static_cast<float>(pid_cnt.at(pid))});
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "\nTop 10 products by mean predicted next-week demand:\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::left
              << std::setw(28) << "Product"
              << std::setw(24) << "Category"
              << "Predicted demand\n";
    std::cout << std::string(72, '-') << "\n";

    std::size_t top_n = std::min<std::size_t>(10, ranked.size());
    for (std::size_t i = 0; i < top_n; ++i) {
        int64_t     pid  = ranked[i].first;
        float       pred = ranked[i].second;
        auto        it   = id_to_info.find(pid);
        std::string name = (it != id_to_info.end()) ? it->second.first  : "?";
        std::string cat  = (it != id_to_info.end()) ? it->second.second : "?";
        std::cout << std::left
                  << std::setw(28) << name.substr(0, 27)
                  << std::setw(24) << cat.substr(0, 23)
                  << std::fixed << std::setprecision(1)
                  << pred << " orders/week\n";
    }

    return 0;
}
