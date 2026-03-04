// src/example_tasks/ml1m_rating_classification.cpp
//
// Task   : binary classification — did a user rate a movie >= 4 stars?
// Dataset: ml-1m (MovieLens 1 Million)
//
// Definition
//   A rating is positive (label=1) if the user gave the movie 4 or 5 stars.
//   A rating is negative (label=0) if the user gave 1, 2, or 3 stars.
//   The label lives directly on the 'ratings' table so no label injection
//   is needed — unlike the churn task, the signal is already a column.
//
// Graph structure used by the GNN
//   ratings --[userId]---> users
//   ratings --[movieId]--> movies
//   (plus reverse edges for bidirectional message passing)
//
// The GNN propagates user preference signals into rating embeddings and
// movie quality signals back into user embeddings across two hops, so
// the model can learn collaborative filtering style patterns implicitly
// without any explicit matrix factorization.
//
// Split strategy
//   Temporal: ratings are sorted by timestamp and split 70/15/15.
//   This mimics real deployment — the model trains on older ratings
//   and is evaluated on the most recent ones.
//
// Inference
//   Entity synthesis: given a specific userId and movieId, predict
//   whether that user would rate that movie >= 4 stars.
//   This covers the case where the (user, movie) pair has no existing
//   rating row — exactly the cold-start recommendation scenario.

#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/training/Trainer.h"
#include "relml/training/TaskSpec.h"

#include <iomanip>
#include <iostream>
#include <unordered_map>

using namespace relml;

// ---------------------------------------------------------------------------
// Rating threshold: ratings >= this value are treated as positive (liked).
// ---------------------------------------------------------------------------
static constexpr float LIKE_THRESHOLD = 4.f;

// ---------------------------------------------------------------------------
// Schema
//
// users   : demographic table, one row per user
// movies  : catalog table, one row per movie
// ratings : observation table connecting users to movies
//           has a natural PK (ratingId) and a timestamp column
//
// FKs on ratings are declared explicitly because the dataset was preprocessed
// and userId/movieId values are contiguous integers starting from 1, so
// FKDetector's name-match + coverage check would find them anyway, but
// explicit is clearer and more robust.
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, TableSchema> ml1m_schemas() {
    return {
        {"users", {
            .pkey_col     = "userId",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                // gender is M/F — two categories
                {.name = "gender",     .type = ColumnType::CATEGORICAL},
                // age is stored as a bucket code (1, 18, 25, 35, 45, 50, 56)
                // treat as categorical so the encoder one-hot encodes it
                // rather than treating the bucket codes as real numbers
                {.name = "age",        .type = ColumnType::CATEGORICAL},
                // occupation is a code 0-20
                {.name = "occupation", .type = ColumnType::CATEGORICAL},
                // zip code: high cardinality string, skip in encoder
                {.name = "zip",        .type = ColumnType::TEXT},
            }
        }},
        {"movies", {
            .pkey_col     = "movieId",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                // title includes the year e.g. "Toy Story (1995)" — skip
                {.name = "title",  .type = ColumnType::TEXT},
                // genres is a pipe-separated string e.g. "Action|Comedy"
                // treat as categorical: the encoder will one-hot on the
                // full string. not ideal but avoids a multi-hot encoder
                // which the current pipeline does not support yet
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
            .columns      = {
                // rating is 1-5 stars, stored as a float
                {.name = "rating",    .type = ColumnType::NUMERICAL},
                // timestamp is a Unix integer in the original file
                // TypeInferrer will see it as NUMERICAL, which is fine —
                // TaskSpec::build_split handles both NUMERICAL and TIMESTAMP
                // time columns correctly
                {.name = "timestamp", .type = ColumnType::NUMERICAL},
            }
        }},
    };
}

// ---------------------------------------------------------------------------
// print_label_stats
//
// Shows the positive/negative split across train/val/test so you can
// verify the threshold and temporal split behaved as expected.
// ---------------------------------------------------------------------------
static void print_label_stats(const TaskSplit& split) {
    auto count = [](const std::vector<TaskSample>& s) {
        std::size_t pos = 0;
        for (const auto& x : s) if (x.label > 0.5f) ++pos;
        return pos;
    };

    std::size_t tr_pos = count(split.train);
    std::size_t va_pos = count(split.val);
    std::size_t te_pos = count(split.test);

    std::cout << "  Split statistics:\n"
              << "    train : " << split.train.size()
              << "  pos=" << tr_pos
              << "  neg=" << (split.train.size() - tr_pos)
              << "  rate=" << std::fixed << std::setprecision(3)
              << static_cast<float>(tr_pos) / split.train.size() << "\n"
              << "    val   : " << split.val.size()
              << "  pos=" << va_pos
              << "  neg=" << (split.val.size() - va_pos)
              << "  rate=" << static_cast<float>(va_pos) / split.val.size() << "\n"
              << "    test  : " << split.test.size()
              << "  pos=" << te_pos
              << "  neg=" << (split.test.size() - te_pos)
              << "  rate=" << static_cast<float>(te_pos) / split.test.size() << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./ml-1m-data";

    std::cout << "Loading MovieLens-1M database from: " << data_dir << "\n";
    Database db = CSVLoader::load_database(data_dir, "ml-1m", ml1m_schemas());

    // FK auto-detection as a sanity check — our explicit declarations on
    // ratings take priority so this is non-destructive
    std::cout << "\nRunning FK detector (sanity check)...\n";
    auto detected = FKDetector::detect(db);
    for (const auto& fk : detected)
        std::cout << "  auto-detected: " << fk.src_table << "." << fk.src_column
                  << " -> " << fk.dst_table
                  << "  (coverage " << fk.coverage * 100.f << "%)\n";

    std::cout << "\nBuilding heterogeneous graph...\n";
    HeteroGraph graph = GraphBuilder::build(db);
    graph.print_summary();

    // ---------------------------------------------------------------------------
    // TaskSpec
    //
    // target_table  = "ratings" because that is where the rating column lives
    //                 and we want one prediction per (user, movie) interaction
    // target_column = "rating"
    // task_type     = BinaryClassification
    // label         = threshold at LIKE_THRESHOLD (4.0), inclusive (>= 4)
    // split         = Temporal using the timestamp column
    //
    // inference_mode = EntitySynthesis: given a userId and movieId that may
    //   not have an existing rating row, predict whether the user would like
    //   the movie. This is the core recommendation scenario.
    //   Change to RowBased + AggType::Fraction if you want to ask
    //   "what fraction of all ratings are positive?" instead.
    // ---------------------------------------------------------------------------
    TaskSpec spec;
    spec.target_table  = "ratings";
    spec.target_column = "rating";
    spec.task_type     = TaskSpec::TaskType::BinaryClassification;

    spec.label_transform.kind      = LabelTransform::Kind::Threshold;
    spec.label_transform.threshold = LIKE_THRESHOLD;
    spec.label_transform.inclusive = true;  // rating >= 4.0 is positive

    spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
    spec.split_time_col = "timestamp";

    // Predict whether user 1 would like movie 1193 (One Flew Over the Cuckoo's Nest)
    // Change these IDs to any userId and movieId that exist in your dataset
    spec.inference_mode = TaskSpec::InferenceMode::EntitySynthesis;
    spec.entity_refs    = {{"userId", "1"}, {"movieId", "1193"}};
    spec.inference_agg  = TaskSpec::AggType::None;

    std::cout << "\nBuilding train/val/test split (temporal, threshold >= "
              << LIKE_THRESHOLD << ")...\n";
    TaskSplit split = spec.build_split(db);
    print_label_stats(split);

    // ---------------------------------------------------------------------------
    // TrainConfig
    //
    // channels=64 is a good starting point for ml-1m: the dataset has
    // ~1M ratings, ~6K users, ~4K movies so the largest embedding matrix is
    // 1M x 64 x 4 bytes = 256 MB, well within normal RAM limits.
    //
    // batch_size=0 means full-batch training. ml-1m is small enough that
    // this is fine and converges faster than mini-batches.
    // ---------------------------------------------------------------------------
    TrainConfig cfg;
    cfg.channels   = 64;
    cfg.gnn_layers = 2;
    cfg.hidden     = 64;
    cfg.dropout    = 0.3f;
    cfg.lr         = 3e-4f;
    cfg.pos_weight = 1.f;   // auto-rebalanced inside Trainer::fit
    cfg.epochs     = 30;
    cfg.batch_size = 0;     // 0 = full batch
    cfg.task       = spec;

    std::cout << "\nBuilding Trainer...\n";
    Trainer trainer(cfg, db, graph);

    std::cout << "\nTraining...\n";
    trainer.fit(split, db, graph);

    // ---------------------------------------------------------------------------
    // Inference example 1: entity synthesis
    //   Predict whether user 1 would like movie 1193
    // ---------------------------------------------------------------------------
    std::cout << "\nEntity synthesis inference...\n";
    float prob = trainer.synthesize_prediction(spec.entity_refs, db, graph);
    std::cout << "  P(user 1 likes movie 1193) = "
              << std::fixed << std::setprecision(4) << prob << "\n";

    // ---------------------------------------------------------------------------
    // Inference example 2: row-based fraction
    //   What fraction of all ratings in the dataset does the model
    //   predict as positive?
    // ---------------------------------------------------------------------------
    std::cout << "\nRow-based inference (predicted positive rate)...\n";
    TaskSpec row_spec       = spec;
    row_spec.inference_mode = TaskSpec::InferenceMode::RowBased;
    row_spec.inference_filters.clear();
    row_spec.inference_agg  = TaskSpec::AggType::Fraction;
    row_spec.entity_refs.clear();

    std::vector<float> all_preds = trainer.predict_all(db, graph);
    TaskSpec::InferenceResult result = row_spec.apply_inference(db, all_preds);

    if (result.aggregate.has_value())
        std::cout << "  Predicted positive rate across all ratings: "
                  << std::fixed << std::setprecision(2)
                  << (*result.aggregate * 100.f) << "%\n";

    // ---------------------------------------------------------------------------
    // Inference example 3: per-user mean predicted rating probability
    //   How likely is user 42 to give positive ratings in general?
    // ---------------------------------------------------------------------------
    std::cout << "\nRow-based inference (user 42 positive rate)...\n";
    TaskSpec user_spec       = spec;
    user_spec.inference_mode = TaskSpec::InferenceMode::RowBased;
    user_spec.entity_refs.clear();
    user_spec.inference_agg  = TaskSpec::AggType::Mean;

    InferenceFilter f;
    f.column = "userId";
    f.op     = "=";
    f.value  = "42";
    user_spec.inference_filters.push_back(f);

    TaskSpec::InferenceResult user_result = user_spec.apply_inference(db, all_preds);
    if (user_result.aggregate.has_value())
        std::cout << "  Mean predicted like-probability for user 42: "
                  << std::fixed << std::setprecision(4)
                  << *user_result.aggregate << "\n"
                  << "  (across " << user_result.row_indices.size()
                  << " rated movies)\n";

    return 0;
}