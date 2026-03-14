// src/example_tasks/avito_user_ad_visit.cpp
//
// Task   : binary classification — will user U revisit ad A in the next
//          PREDICTION_WINDOW_DAYS days?
// Dataset: rel-avito (Avito Context Ad Clicks, RelBench)
//
// The target table is materialised from:
//
//   SELECT DISTINCT v.UserID, v.AdID,
//       COALESCE(MAX(
//           CASE WHEN f.ViewDate >= :cutoff THEN 1 ELSE 0 END
//       ), 0) AS will_visit
//   FROM  VisitStream v
//   LEFT JOIN VisitStream f
//       ON  f.UserID   = v.UserID
//       AND f.AdID     = v.AdID
//       AND f.ViewDate >= :cutoff
//   WHERE v.ViewDate < :cutoff
//   GROUP BY v.UserID, v.AdID
//
// where  cutoff = max(ViewDate) - PREDICTION_WINDOW_DAYS * 86400
//
// One row per (user, ad) pair the user visited BEFORE the cutoff.
// Label = 1 if they visit the same ad again WITHIN the prediction window.
// Label = 0 otherwise.
//
// This formulation is tractable: the number of rows equals the number of
// distinct (user, ad) pairs in VisitStream before the cutoff — typically
// in the hundreds of thousands, not billions. No negative sampling is needed
// because the negatives are already defined by the query: every pair the user
// visited but did not revisit is a natural negative.
//
// Usage:
//   ./avito_user_ad_visit <data_dir>
//   ./avito_user_ad_visit <data_dir> <UserID> <AdID>

#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/training/Trainer.h"
#include "relml/training/TaskSpec.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace relml;

static constexpr int PREDICTION_WINDOW_DAYS = 4;

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, TableSchema> avito_schemas() {
    return {
        {"UserInfo", {
            .pkey_col     = "UserID",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {}
        }},
        {"AdsInfo", {
            .pkey_col     = "AdID",
            .time_col     = std::nullopt,
            .foreign_keys = {
                {.column = "LocationID", .target_table = "Location"},
                {.column = "CategoryID", .target_table = "Category"},
            },
            .columns = {
                {.name = "Title",     .type = ColumnType::TEXT},
                {.name = "IsContext", .type = ColumnType::CATEGORICAL},
            }
        }},
        {"Category", {
            .pkey_col     = std::optional<std::string>{"CategoryID"},
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {{.name = "Level", .type = ColumnType::CATEGORICAL}}
        }},
        {"Location", {
            .pkey_col     = std::optional<std::string>{"LocationID"},
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {{.name = "Level", .type = ColumnType::CATEGORICAL}}
        }},
        {"SearchInfo", {
            .pkey_col     = "SearchID",
            .time_col     = "SearchDate",
            .foreign_keys = {
                {.column = "UserID",     .target_table = "UserInfo"},
                {.column = "LocationID", .target_table = "Location"},
                {.column = "CategoryID", .target_table = "Category"},
            },
            .columns = {
                {.name = "IsUserLoggedOn", .type = ColumnType::CATEGORICAL},
                {.name = "SearchQuery",    .type = ColumnType::TEXT},
            }
        }},
        {"SearchStream", {
            .pkey_col     = std::nullopt,
            .time_col     = "SearchDate",
            .foreign_keys = {
                {.column = "SearchID", .target_table = "SearchInfo"},
                {.column = "AdID",     .target_table = "AdsInfo"},
            },
            .columns = {
                {.name = "ObjectType", .type = ColumnType::CATEGORICAL},
                {.name = "IsClick",    .type = ColumnType::CATEGORICAL},
            }
        }},
        {"VisitStream", {
            .pkey_col     = std::nullopt,
            .time_col     = "ViewDate",
            .foreign_keys = {
                {.column = "UserID", .target_table = "UserInfo"},
                {.column = "AdID",   .target_table = "AdsInfo"},
            },
            .columns = {}
        }},
        {"PhoneRequestsStream", {
            .pkey_col     = std::nullopt,
            .time_col     = "PhoneRequestDate",
            .foreign_keys = {
                {.column = "UserID", .target_table = "UserInfo"},
                {.column = "AdID",   .target_table = "AdsInfo"},
            },
            .columns = {}
        }},
    };
}

// ---------------------------------------------------------------------------
// build_user_ad_candidates
//
// Executes the SPJA query and adds the resulting table to the database.
//
// UserID and AdID are stored as NUMERICAL so GraphBuilder can resolve them
// against the NUMERICAL PKs of UserInfo and AdsInfo. After GraphBuilder runs,
// call flip_candidate_ids_to_text() so HeteroEncoder skips them.
// ---------------------------------------------------------------------------
static void build_user_ad_candidates(Database& db) {

    const Table&  visits   = db.get_table("VisitStream");
    const Column& user_col = visits.get_column("UserID");
    const Column& ad_col   = visits.get_column("AdID");
    const Column& date_col = visits.get_column("ViewDate");
    std::size_t   V        = visits.num_rows();

    // Find cutoff
    int64_t max_ts = std::numeric_limits<int64_t>::min();
    for (std::size_t i = 0; i < V; ++i)
        if (!date_col.is_null(i))
            max_ts = std::max(max_ts, date_col.get_timestamp(i));

    int64_t cutoff = max_ts
                   - static_cast<int64_t>(PREDICTION_WINDOW_DAYS) * 86400LL;

    std::cout << "  Observation window end  : " << max_ts  << " (unix)\n"
              << "  Cutoff (-" << PREDICTION_WINDOW_DAYS
              << " days)      : " << cutoff << " (unix)\n";

    // Collect future visits: (UserID, AdID) pairs with ViewDate >= cutoff.
    // These are the positive labels.
    struct PairHash {
        std::size_t operator()(const std::pair<int64_t,int64_t>& p) const {
            std::size_t h = std::hash<int64_t>{}(p.first);
            h ^= std::hash<int64_t>{}(p.second)
                 + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_set<std::pair<int64_t,int64_t>, PairHash> future_visits;
    for (std::size_t i = 0; i < V; ++i) {
        if (date_col.is_null(i) || user_col.is_null(i) || ad_col.is_null(i))
            continue;
        if (date_col.get_timestamp(i) < cutoff) continue;
        future_visits.insert({
            static_cast<int64_t>(user_col.get_numerical(i)),
            static_cast<int64_t>(ad_col.get_numerical(i))
        });
    }

    // Collect distinct (UserID, AdID) pairs from VisitStream before the cutoff.
    // These are the candidate rows — one per unique pair the user has visited.
    // No sampling: every historical (user, ad) pair is a candidate.
    // Negatives arise naturally: pairs not in future_visits get label 0.
    std::unordered_set<std::pair<int64_t,int64_t>, PairHash> historical_pairs;
    for (std::size_t i = 0; i < V; ++i) {
        if (date_col.is_null(i) || user_col.is_null(i) || ad_col.is_null(i))
            continue;
        if (date_col.get_timestamp(i) >= cutoff) continue;
        historical_pairs.insert({
            static_cast<int64_t>(user_col.get_numerical(i)),
            static_cast<int64_t>(ad_col.get_numerical(i))
        });
    }

    std::cout << "  Historical (user,ad) pairs: " << historical_pairs.size() << "\n";

    // Assign labels and build column vectors
    std::vector<double> rows_user, rows_ad, rows_label;
    rows_user.reserve(historical_pairs.size());
    rows_ad.reserve(historical_pairs.size());
    rows_label.reserve(historical_pairs.size());

    std::size_t n_pos = 0;
    for (const auto& [uid, aid] : historical_pairs) {
        rows_user.push_back(static_cast<double>(uid));
        rows_ad.push_back(static_cast<double>(aid));
        double label = future_visits.count({uid, aid}) ? 1.0 : 0.0;
        rows_label.push_back(label);
        if (label > 0.5) ++n_pos;
    }

    std::size_t total = rows_user.size();
    std::cout << "  Positives               : " << n_pos
              << "  (" << std::fixed << std::setprecision(2)
              << (total > 0 ? 100.f * n_pos / total : 0.f) << "%)\n"
              << "  Negatives               : " << (total - n_pos) << "\n";

    // Build the Table with NUMERICAL columns for GraphBuilder FK resolution
    Table cand("UserAdCandidates");
    cand.foreign_keys = {
        {.column = "UserID", .target_table = "UserInfo"},
        {.column = "AdID",   .target_table = "AdsInfo"},
    };

    Column col_user("UserID",     ColumnType::NUMERICAL);
    Column col_ad  ("AdID",       ColumnType::NUMERICAL);
    Column col_lbl ("will_visit", ColumnType::NUMERICAL);

    for (std::size_t i = 0; i < total; ++i) {
        col_user.data.push_back(rows_user[i]);
        col_ad.data.push_back(rows_ad[i]);
        col_lbl.data.push_back(rows_label[i]);
    }

    cand.add_column(std::move(col_user));
    cand.add_column(std::move(col_ad));
    cand.add_column(std::move(col_lbl));
    db.add_table(std::move(cand));
}

// ---------------------------------------------------------------------------
// flip_candidate_ids_to_text
//
// Called after GraphBuilder::build. Flips UserID and AdID to TEXT so
// HeteroEncoder skips them. All signal flows through the FK edges.
// ---------------------------------------------------------------------------
static void flip_candidate_ids_to_text(Database& db) {
    Table& cand = db.get_table("UserAdCandidates");
    cand.get_column("UserID").type = ColumnType::TEXT;
    cand.get_column("AdID").type   = ColumnType::TEXT;
}

// ---------------------------------------------------------------------------
// print_split_stats
// ---------------------------------------------------------------------------
static void print_split_stats(const TaskSplit& split) {
    auto print_row = [](const std::string& name,
                        const std::vector<TaskSample>& s) {
        std::size_t pos = 0;
        for (const auto& x : s) if (x.label > 0.5f) ++pos;
        std::cout << "    " << std::setw(6) << std::left << name
                  << ": " << s.size()
                  << "  pos=" << pos
                  << "  neg=" << (s.size() - pos)
                  << "  rate=" << std::fixed << std::setprecision(4)
                  << (s.size() > 0 ? static_cast<float>(pos) / s.size() : 0.f)
                  << "\n";
    };
    std::cout << "  Split statistics:\n";
    print_row("train", split.train);
    print_row("val",   split.val);
    print_row("test",  split.test);
}

// ---------------------------------------------------------------------------
// print_top_k_per_user
// ---------------------------------------------------------------------------
static void print_top_k_per_user(
    const Database&           db,
    const std::vector<float>& all_preds,
    std::size_t               top_k     = 5,
    std::size_t               max_users = 10)
{
    const Table&  cand     = db.get_table("UserAdCandidates");
    const Column& user_col = cand.get_column("UserID");
    const Column& ad_col   = cand.get_column("AdID");

    std::map<int64_t, std::vector<std::pair<float, int64_t>>> user_preds;
    for (std::size_t i = 0; i < cand.num_rows() && i < all_preds.size(); ++i) {
        if (user_col.is_null(i) || ad_col.is_null(i)) continue;
        int64_t uid = std::stoll(user_col.get_categorical(i));
        int64_t aid = std::stoll(ad_col.get_categorical(i));
        user_preds[uid].push_back({all_preds[i], aid});
    }

    std::cout << "\n  Top-" << top_k << " predicted revisit ads per user"
              << " (first " << max_users << " users):\n"
              << "  " << std::string(50, '-') << "\n";

    std::size_t shown = 0;
    for (auto& [uid, pairs] : user_preds) {
        if (shown >= max_users) break;
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        std::cout << "  User " << uid << ":\n";
        for (std::size_t k = 0; k < std::min(top_k, pairs.size()); ++k)
            std::cout << "    " << (k+1) << ". AdID=" << pairs[k].second
                      << "  p=" << std::fixed << std::setprecision(4)
                      << pairs[k].first << "\n";
        ++shown;
    }
    std::cout << "  " << std::string(50, '-') << "\n";
}

// ---------------------------------------------------------------------------
// score_pair
//
// Exact lookup in UserAdCandidates first. Falls back to synthesize_prediction
// for pairs not in the candidate set (e.g. a brand new ad the user has never
// seen — the model scores it from category/location/popularity signal alone).
// ---------------------------------------------------------------------------
static float score_pair(
    int64_t                   query_user,
    int64_t                   query_ad,
    const Database&           db,
    const HeteroGraph&        graph,
    Trainer&                  trainer,
    const std::vector<float>& all_preds)
{
    const Table&  cand     = db.get_table("UserAdCandidates");
    const Column& user_col = cand.get_column("UserID");
    const Column& ad_col   = cand.get_column("AdID");

    for (std::size_t i = 0; i < cand.num_rows() && i < all_preds.size(); ++i) {
        if (user_col.is_null(i) || ad_col.is_null(i)) continue;
        if (std::stoll(user_col.get_categorical(i)) == query_user &&
            std::stoll(ad_col.get_categorical(i))   == query_ad)
            return all_preds[i];
    }

    std::cout << "  (not in candidate set — using synthesize_prediction)\n";
    return trainer.synthesize_prediction(
        {{"UserID", std::to_string(query_user)},
         {"AdID",   std::to_string(query_ad)}},
        db, graph);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./rel-avito-data";

    std::cout << "Loading Avito database from: " << data_dir << "\n";
    Database db = CSVLoader::load_database(data_dir, "rel-avito", avito_schemas());

    std::cout << "\nRunning FK detector...\n";
    auto detected = FKDetector::detect(db);
    for (const auto& fk : detected)
        std::cout << "  " << fk.src_table << "." << fk.src_column
                  << " -> " << fk.dst_table
                  << "  (" << std::fixed << std::setprecision(1)
                  << fk.coverage * 100.f << "%)\n";
    if (detected.empty()) std::cout << "  (none beyond declared FKs)\n";

    std::cout << "\nMaterialising UserAdCandidates "
              << "(window = " << PREDICTION_WINDOW_DAYS << " days)...\n";
    build_user_ad_candidates(db);

    std::cout << "\nBuilding graph...\n";
    HeteroGraph graph = GraphBuilder::build(db);
    graph.print_summary();

    flip_candidate_ids_to_text(db);

    TaskSpec spec;
    spec.target_table  = "UserAdCandidates";
    spec.target_column = "will_visit";
    spec.task_type     = TaskSpec::TaskType::BinaryClassification;

    spec.label_transform.kind      = LabelTransform::Kind::Threshold;
    spec.label_transform.threshold = 0.5f;
    spec.label_transform.inclusive = true;

    spec.split_strategy = TaskSpec::SplitStrategy::Random;
    spec.inference_mode = TaskSpec::InferenceMode::RowBased;
    spec.inference_agg  = TaskSpec::AggType::Fraction;

    std::cout << "\nBuilding split...\n";
    TaskSplit split = spec.build_split(db);
    print_split_stats(split);

    db.get_table("UserAdCandidates").get_column("will_visit").type =
        ColumnType::TEXT;

    TrainConfig cfg;
    cfg.channels   = 32;
    cfg.gnn_layers = 2;
    cfg.hidden     = 64;
    cfg.dropout    = 0.3f;
    cfg.lr         = 3e-4f;
    cfg.pos_weight = 1.f;
    cfg.epochs     = 20;
    cfg.batch_size = 4096;
    cfg.task       = spec;

    std::cout << "\nTraining...\n";
    Trainer trainer(cfg, db, graph);
    trainer.fit(split, db, graph);

    std::cout << "\nScoring all candidate pairs...\n";
    std::vector<float> all_preds = trainer.predict_all(db, graph);

    auto result = spec.apply_inference(db, all_preds);
    if (result.aggregate.has_value())
        std::cout << "  Predicted revisit rate: " << std::fixed
                  << std::setprecision(2)
                  << (*result.aggregate * 100.f) << "%\n";

    print_top_k_per_user(db, all_preds, /*top_k=*/5, /*max_users=*/10);

    // Point query: ./avito_user_ad_visit <data_dir> <UserID> <AdID>
    if (argc >= 4) {
        int64_t query_user = std::stoll(argv[2]);
        int64_t query_ad   = std::stoll(argv[3]);
        float p = score_pair(query_user, query_ad, db, graph, trainer, all_preds);
        std::cout << "\nP(user " << query_user
                  << " revisits ad " << query_ad
                  << " in next " << PREDICTION_WINDOW_DAYS << " days) = "
                  << std::fixed << std::setprecision(4) << p << "\n";
    }

    return 0;
}