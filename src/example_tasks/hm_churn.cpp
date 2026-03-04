// example_tasks/hm_churn.cpp
//
// Task   : binary classification — will a customer churn?
// Dataset: rel-hm (H&M Personalized Fashion Recommendations)
//
// Definition
//   A customer "churns" if they have no transaction in the final
//   CHURN_WINDOW_DAYS days of the observation period.
//   Label 1 = churned, 0 = active.
//
// Why the label lives on 'customer', not 'transactions'
//   We want one prediction per customer, so the target table is
//   'customer'. The label is derived from transaction history and
//   injected as a synthetic column before training.
//
// Graph structure used by the GNN
//   transactions --[customer_id]--> customer
//   transactions --[article_id]---> article
//   (plus reverse edges for bidirectional message passing)
//
// The GNN propagates article purchase signals into customer embeddings,
// so the model can learn "customers who repeatedly buy discounted basics
// tend to churn when promotions end" style patterns implicitly.




/*
this file is equivalent to the following SQL:

ALTER TABLE customer ADD COLUMN will_churn INTEGER;

WITH last_purchase AS (
    SELECT customer_id, MAX(t_dat) AS last_ts
    FROM transactions
    GROUP BY customer_id
),
cutoff AS (
    SELECT MAX(t_dat) - INTERVAL 7 DAYS AS churn_cutoff
    FROM transactions
)
UPDATE customer
SET will_churn = CASE
    WHEN last_ts IS NULL OR last_ts <= churn_cutoff THEN 1
    ELSE 0
END
FROM last_purchase, cutoff
WHERE customer.customer_id = last_purchase.customer_id;


*/

#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/training/Trainer.h"
#include "relml/training/TaskSpec.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_map>

using namespace relml;

// ---------------------------------------------------------------------------
// How many days before the last observed date define the churn window.
// Customers with no purchase inside this window are labelled as churned.
// ---------------------------------------------------------------------------
static constexpr int CHURN_WINDOW_DAYS = 7;

// ---------------------------------------------------------------------------
// Schema
//
// Foreign keys on transactions are declared explicitly rather than relying
// on FKDetector because the article_id values in the transaction CSV may
// not have high enough coverage against the article table PK depending on
// how the dataset was preprocessed.
//
// transactions has no natural primary key and is never a FK target, so
// pkey_col = std::nullopt is correct — GraphBuilder only needs PKs for
// tables that other tables point to.
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, TableSchema> hm_schemas() {
    return {
        {"customer", {
            .pkey_col     = "customer_id",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                {.name = "club_member_status",     .type = ColumnType::CATEGORICAL},
                {.name = "fashion_news_frequency", .type = ColumnType::CATEGORICAL},
                {.name = "age",                    .type = ColumnType::NUMERICAL},
                // postal_code is a SHA-256 hash — very high cardinality, treat as TEXT
                // so the encoder skips it rather than building a giant one-hot vector
                {.name = "postal_code",            .type = ColumnType::TEXT},
            }
        }},
        {"article", {
            .pkey_col     = "article_id",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                {.name = "product_type_name",         .type = ColumnType::CATEGORICAL},
                {.name = "product_group_name",        .type = ColumnType::CATEGORICAL},
                {.name = "graphical_appearance_name", .type = ColumnType::CATEGORICAL},
                {.name = "colour_group_name",         .type = ColumnType::CATEGORICAL},
                {.name = "index_name",                .type = ColumnType::CATEGORICAL},
                {.name = "section_name",              .type = ColumnType::CATEGORICAL},
                {.name = "garment_group_name",        .type = ColumnType::CATEGORICAL},
                {.name = "detail_desc",               .type = ColumnType::TEXT},
            }
        }},
        {"transactions", {
            .pkey_col     = std::nullopt,   // no natural PK; never a FK target
            .time_col     = "t_dat",
            // Declare FKs explicitly: the GNN needs these edges
            .foreign_keys = {
                {.column = "customer_id", .target_table = "customer"},
                {.column = "article_id",  .target_table = "article"},
            },
            .columns      = {
                // sales_channel_id is 1 or 2 (online vs store) — treat as categorical
                {.name = "sales_channel_id", .type = ColumnType::CATEGORICAL},
                {.name = "price",            .type = ColumnType::NUMERICAL},
            }
        }},
    };
}

// ---------------------------------------------------------------------------
// inject_churn_labels
//
// Computes a "will_churn" column for the customer table based on transaction
// recency. The column is added directly to the in-memory Database so the
// rest of the pipeline sees it as a normal NUMERICAL column.
//
// Algorithm
//   1. Find the maximum t_dat across all transactions (the "present" moment).
//   2. churn_cutoff = max_date - CHURN_WINDOW_DAYS * 86400 seconds.
//   3. For each customer, find their most recent transaction timestamp.
//   4. If last_ts <= churn_cutoff (or no transactions at all) -> churned (1).
//      Otherwise -> active (0).
// ---------------------------------------------------------------------------
static void inject_churn_labels(Database& db) {
    const Table& txn      = db.get_table("transactions");
    Table&       customer = db.get_table("customer");

    const Column& t_dat_col   = txn.get_column("t_dat");
    const Column& cust_id_col = txn.get_column("customer_id");

    // Step 1: find the latest timestamp in the dataset
    int64_t max_ts = std::numeric_limits<int64_t>::min();
    for (std::size_t i = 0; i < txn.num_rows(); ++i) {
        if (t_dat_col.is_null(i)) continue;
        int64_t ts = t_dat_col.get_timestamp(i);
        if (ts > max_ts) max_ts = ts;
    }

    int64_t churn_cutoff = max_ts - static_cast<int64_t>(CHURN_WINDOW_DAYS) * 86400LL;

    // Step 2: for each customer_id, record their most recent transaction
    std::unordered_map<int64_t, int64_t> last_ts_by_customer;
    last_ts_by_customer.reserve(customer.num_rows());

    for (std::size_t i = 0; i < txn.num_rows(); ++i) {
        if (t_dat_col.is_null(i) || cust_id_col.is_null(i)) continue;
        int64_t cid = static_cast<int64_t>(cust_id_col.get_numerical(i));
        int64_t ts  = t_dat_col.get_timestamp(i);
        auto [it, inserted] = last_ts_by_customer.emplace(cid, ts);
        if (!inserted && ts > it->second)
            it->second = ts;
    }

    // Step 3: build the churn label column aligned to the customer table
    const Column& pk_col = customer.get_column("customer_id");
    Column churn_col("will_churn", ColumnType::NUMERICAL);
    churn_col.data.reserve(customer.num_rows());

    std::size_t n_churned    = 0;
    std::size_t n_no_history = 0;

    for (std::size_t i = 0; i < customer.num_rows(); ++i) {
        if (pk_col.is_null(i)) {
            churn_col.data.push_back(std::monostate{});
            continue;
        }
        int64_t cid = static_cast<int64_t>(pk_col.get_numerical(i));
        auto it = last_ts_by_customer.find(cid);

        double label;
        if (it == last_ts_by_customer.end()) {
            // Customer appears in customer table but has no transactions at all.
            // Treat as churned since they have never been active.
            label = 1.0;
            ++n_churned;
            ++n_no_history;
        } else if (it->second <= churn_cutoff) {
            label = 1.0;  // last purchase was before the churn window
            ++n_churned;
        } else {
            label = 0.0;  // bought something within the last CHURN_WINDOW_DAYS days
        }
        churn_col.data.push_back(label);
    }

    customer.add_column(std::move(churn_col));

    std::cout << "  Observation window ends  : " << max_ts << " (unix)\n"
              << "  Churn cutoff             : " << churn_cutoff << " (unix)\n"
              << "  Churn window             : " << CHURN_WINDOW_DAYS << " days\n"
              << "  Customers with no history: " << n_no_history << "\n"
              << "  Churned  (label=1)       : " << n_churned << "\n"
              << "  Active   (label=0)       : " << (customer.num_rows() - n_churned) << "\n"
              << "  Churn rate               : " << std::fixed << std::setprecision(3)
              << static_cast<float>(n_churned) / customer.num_rows() << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./rel-hm-data";

    std::cout << "Loading H&M database from: " << data_dir << "\n";
    Database db = CSVLoader::load_database(data_dir, "rel-hm", hm_schemas());

    // FK auto-detection as a sanity check — our explicit declarations take
    // priority, so this is non-destructive even if it finds the same edges.
    std::cout << "\nRunning FK detector (sanity check)...\n";
    auto detected = FKDetector::detect(db);
    for (const auto& fk : detected)
        std::cout << "  auto-detected: " << fk.src_table << "." << fk.src_column
                  << " -> " << fk.dst_table
                  << "  (coverage " << fk.coverage * 100.f << "%)\n";

    std::cout << "\nInjecting churn labels...\n";
    inject_churn_labels(db);

    std::cout << "\nBuilding heterogeneous graph...\n";
    HeteroGraph graph = GraphBuilder::build(db);
    graph.print_summary();

    // ---------------------------------------------------------------------------
    // TaskSpec — built manually, no LLM involved
    //
    // We predict customer.will_churn (the 0/1 column we just injected).
    // The threshold at 0.5 is a no-op since values are already 0 or 1,
    // but it keeps the pipeline consistent with how all binary tasks work.
    //
    // Random split is correct here because labels are per-customer and there
    // is no meaningful time ordering of customers themselves. A temporal split
    // would require a customer creation date column which this dataset does
    // not have.
    // ---------------------------------------------------------------------------
    TaskSpec spec;
    spec.target_table  = "customer";
    spec.target_column = "will_churn";
    spec.task_type     = TaskSpec::TaskType::BinaryClassification;

    spec.label_transform.kind      = LabelTransform::Kind::Threshold;
    spec.label_transform.threshold = 0.5f;
    spec.label_transform.inclusive = true;

    spec.split_strategy = TaskSpec::SplitStrategy::Random;
    spec.inference_mode = TaskSpec::InferenceMode::RowBased;
    spec.inference_agg  = TaskSpec::AggType::Fraction;

    std::cout << "\nBuilding train/val/test split...\n";
    TaskSplit split = spec.build_split(db);

    // ---------------------------------------------------------------------------
    // TrainConfig
    //
    // batch_size > 0 enables mini-batch training. Since the customer table
    // can have hundreds of thousands of rows, full-batch is expensive.
    // The encoder and GNN still run on the full graph each epoch (that is
    // unavoidable with message passing), but the head loss is computed in
    // chunks which reduces peak memory for the MLP head backward pass.
    // ---------------------------------------------------------------------------
    TrainConfig cfg;
    cfg.channels   = 32;
    cfg.gnn_layers = 2;
    cfg.hidden     = 64;
    cfg.dropout    = 0.3f;
    cfg.lr         = 3e-4f;
    cfg.pos_weight = 1.f;   // auto-rebalanced inside Trainer::fit based on class ratio
    cfg.epochs     = 30;
    cfg.batch_size = 4096;
    cfg.task       = spec;

    std::cout << "\nBuilding Trainer...\n";
    Trainer trainer(cfg, db, graph);

    std::cout << "\nTraining...\n";
    trainer.fit(split, db, graph);

    // ---------------------------------------------------------------------------
    // Inference: score all customers and report the predicted churn rate.
    // ---------------------------------------------------------------------------
    std::cout << "\nScoring all customers...\n";
    std::vector<float> all_preds = trainer.predict_all(db, graph);
    TaskSpec::InferenceResult result = spec.apply_inference(db, all_preds);

    if (result.aggregate.has_value())
        std::cout << "Predicted churn rate across all customers: "
                  << std::fixed << std::setprecision(2)
                  << (*result.aggregate * 100.f) << "%\n";

    // Optional: write per-customer scores to CSV for downstream analysis.
    // The row index in all_preds maps directly to the customer table row,
    // so joining on customer_id is straightforward.
    //
    // std::ofstream out("churn_scores.csv");
    // out << "customer_id,churn_prob\n";
    // const Column& pk = db.get_table("customer").get_column("customer_id");
    // for (std::size_t i = 0; i < all_preds.size(); ++i)
    //     if (!pk.is_null(i))
    //         out << static_cast<int64_t>(pk.get_numerical(i))
    //             << "," << all_preds[i] << "\n";

    return 0;
}