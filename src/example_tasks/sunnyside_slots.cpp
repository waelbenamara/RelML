// src/example_tasks/sunnyside_slots.cpp
//
// Task   : regression — predict next week's transaction count for each
//          (day-of-week × time-slot) at The Sunny Side café.
//
//          Answer: "How busy will Saturday afternoon be next week?"
//          Use:    weekly staffing — print the forecast on Monday morning
//                  and schedule shifts accordingly.
//
// Dataset: The Sunny Side café, Paris.
//          Built by processing.py (section 10).
//
// Time slots
//   Slot 0 — Morning  : 09:00–12:00
//   Slot 1 — Lunch    : 12:00–15:00
//   Slot 2 — Afternoon: 15:00–18:00
//
// Active days: Tuesday–Sunday (Monday excluded — café effectively closed).
// Entities: 6 days × 3 slots = 18 (day, slot) combinations.
// Observations: ~19 weeks × 18 entities = 342 rows (324 labeled).
//
// Graph structure
//   slot_week_learning  --[slot_entity_id]--> dim_slot
//   (plus reverse edges for bidirectional message passing)
//
//   dim_slot has 18 rows — one per (day_of_week, time_slot) entity.
//   The GNN lets Saturday/Afternoon share signal with Sunday/Afternoon:
//   dim_slot nodes aggregate over all historical observations for that
//   (day, slot) combination, learning day-of-week and slot-of-day effects.
//   slot_week_learning nodes then receive the enriched slot embedding
//   alongside their own lag/weather features.
//
// Features on slot_week_learning
//   Slot identity : day_of_week_num (CATEGORICAL), time_slot (CATEGORICAL)
//   Calendar      : iso_year (CATEGORICAL), iso_week (CATEGORICAL)
//   Weather       : temp_max/min_week_avg, precip_week_sum, wind_week_max
//   Today's count : transaction_count
//   Lags          : txn_lag1 … txn_lag4 (same entity, prior weeks)
//   Rolling mean  : txn_rolling4_mean
//
// Target: txn_count_next_week (declared TEXT; injected as NUMERICAL at runtime)
//
// Split: Temporal by week_start (70/15/15). Predicts future weeks from past.
//
// Inference
//   Scores every row in slot_week_learning and prints a staffing forecast
//   table for the most recent available week — one row per (day, slot).

#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/training/Trainer.h"
#include "relml/training/TaskSpec.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace relml;

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, TableSchema> slots_schemas() {
    return {

        // ── dim_slot ──────────────────────────────────────────────────────
        // 18 rows: 6 active days × 3 time slots.
        // The GNN aggregates all weekly observations for each (day, slot)
        // into a per-entity embedding.
        {"dim_slot", {
            .pkey_col     = "slot_entity_id",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns = {
                {.name = "day_of_week_num", .type = ColumnType::CATEGORICAL},
                {.name = "day_of_week",     .type = ColumnType::CATEGORICAL},
                {.name = "is_weekend",      .type = ColumnType::CATEGORICAL},
                {.name = "time_slot",       .type = ColumnType::CATEGORICAL},
                {.name = "slot_label",      .type = ColumnType::TEXT},
            }
        }},

        // ── slot_week_learning (TARGET TABLE) ─────────────────────────────
        // One row per (slot_entity × ISO week).
        // FK to dim_slot so the GNN can propagate cross-entity signals.
        {"slot_week_learning", {
            .pkey_col     = "row_id",
            .time_col     = "week_start",
            .foreign_keys = {
                {.column = "slot_entity_id", .target_table = "dim_slot"},
            },
            .columns = {
                {.name = "day_of_week_num",  .type = ColumnType::CATEGORICAL},
                {.name = "time_slot",        .type = ColumnType::CATEGORICAL},
                {.name = "iso_year",         .type = ColumnType::CATEGORICAL},
                {.name = "iso_week",         .type = ColumnType::CATEGORICAL},
                // ── TARGET — encoder must never see this ──────────────────
                {.name = "txn_count_next_week", .type = ColumnType::TEXT},
                // transaction_count, lags, rolling mean, weather →
                // auto-inferred NUMERICAL
            }
        }},
    };
}

// ---------------------------------------------------------------------------
// inject_slot_labels
// ---------------------------------------------------------------------------
static void inject_slot_labels(Database& db) {
    Table&        tbl     = db.get_table("slot_week_learning");
    const Column& raw_col = tbl.get_column("txn_count_next_week");

    Column label_col("visits_next_week", ColumnType::NUMERICAL);
    label_col.data.reserve(tbl.num_rows());

    std::size_t n_valid = 0, n_null = 0;
    for (std::size_t i = 0; i < tbl.num_rows(); ++i) {
        if (raw_col.is_null(i)) { label_col.data.push_back(std::monostate{}); ++n_null; continue; }
        const std::string& s = raw_col.get_text(i);
        if (s.empty())         { label_col.data.push_back(std::monostate{}); ++n_null; continue; }
        try   { label_col.data.push_back(std::stod(s)); ++n_valid; }
        catch (...) { label_col.data.push_back(std::monostate{}); ++n_null; }
    }
    tbl.add_column(std::move(label_col));
    std::cout << "  Labeled rows  : " << n_valid << "\n"
              << "  Null (no future week): " << n_null  << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1)
        ? argv[1]
        : "./data/sunny-side-data/database_sunnyside";

    std::cout << "Loading Sunny Side slot database from: " << data_dir << "\n";
    Database db("sunny-side-slots");

    const auto schemas = slots_schemas();
    auto load_one = [&](const std::string& csv, const std::string& tbl_name) {
        std::string path = data_dir + "/" + csv;
        std::cout << "  " << csv << " ... "; std::cout.flush();
        Table t = CSVLoader::load_table(path, tbl_name, schemas.at(tbl_name));
        std::cout << t.num_rows() << " rows\n";
        db.add_table(std::move(t));
    };

    load_one("dim_slot.csv",           "dim_slot");
    load_one("slot_week_learning.csv", "slot_week_learning");

    std::cout << "\nInjecting slot demand labels...\n";
    inject_slot_labels(db);

    std::cout << "\nBuilding heterogeneous graph...\n";
    HeteroGraph graph = GraphBuilder::build(db);
    graph.print_summary();

    // -----------------------------------------------------------------------
    // TaskSpec: regression on visits_next_week, temporal split by week_start
    // -----------------------------------------------------------------------
    TaskSpec spec;
    spec.target_table  = "slot_week_learning";
    spec.target_column = "visits_next_week";
    spec.task_type     = TaskSpec::TaskType::Regression;

    spec.label_transform.kind = LabelTransform::Kind::Normalize;

    spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
    spec.split_time_col = "week_start";

    spec.inference_mode = TaskSpec::InferenceMode::RowBased;
    spec.inference_agg  = TaskSpec::AggType::Mean;

    std::cout << "\nBuilding temporal split (70/15/15)...\n";
    TaskSplit split = spec.build_split(db);
    std::cout << "  train: " << split.train.size() << " rows\n"
              << "  val  : " << split.val.size()   << " rows\n"
              << "  test : " << split.test.size()  << " rows\n";

    // -----------------------------------------------------------------------
    // TrainConfig
    //
    // channels=32 : small dataset (342 rows, 18 entity nodes).
    // gnn_layers=2 : slot_week_learning → dim_slot → slot_week_learning.
    //                Layer 1 aggregates all weekly obs into each slot entity.
    //                Layer 2 lets each obs absorb the enriched slot embedding.
    // hidden=64    : MLP head capacity.
    // epochs=300   : full-batch on a tiny dataset benefits from many epochs.
    // -----------------------------------------------------------------------
    TrainConfig cfg;
    cfg.channels   = 32;
    cfg.gnn_layers = 2;
    cfg.hidden     = 64;
    cfg.dropout    = 0.2f;
    cfg.lr         = 3e-4f;
    cfg.pos_weight = 1.f;
    cfg.epochs     = 3000;
    cfg.batch_size = 0;
    cfg.task       = spec;

    std::cout << "\nBuilding Trainer...\n";
    Trainer trainer(cfg, db, graph);

    std::cout << "\nTraining...\n";
    trainer.fit(split, db, graph);

    // -----------------------------------------------------------------------
    // Inference: print a staffing forecast table for the most recent week.
    // Shows predicted transactions per (day, slot) so the owner can plan
    // shifts for the coming week.
    // -----------------------------------------------------------------------
    std::cout << "\nScoring all slot × week rows...\n";
    std::vector<float> all_preds = trainer.predict_all(db, graph);

    const Table&  swl        = db.get_table("slot_week_learning");
    const Column& entity_col = swl.get_column("slot_entity_id");
    const Column& week_col   = swl.get_column("iso_week");
    const Column& year_col   = swl.get_column("iso_year");
    const Column& lbl_col    = swl.get_column("visits_next_week");

    const Table&  ds        = db.get_table("dim_slot");
    const Column& ds_eid    = ds.get_column("slot_entity_id");
    const Column& ds_day    = ds.get_column("day_of_week");
    const Column& ds_lbl    = ds.get_column("slot_label");

    // Map slot_entity_id -> (day_name, slot_label)
    std::unordered_map<int, std::pair<std::string, std::string>> entity_info;
    for (std::size_t i = 0; i < ds.num_rows(); ++i) {
        if (ds_eid.is_null(i)) continue;
        int eid = static_cast<int>(ds_eid.get_numerical(i));
        entity_info[eid] = {
            ds_day.is_null(i) ? "?" : ds_day.get_text(i),
            ds_lbl.is_null(i) ? "?" : ds_lbl.get_text(i),
        };
    }

    // Find the most recent iso_year/week with labeled rows
    // iso_year/iso_week are CATEGORICAL → stored as strings, use get_text()
    int max_year = 0, max_week = 0;
    for (std::size_t i = 0; i < swl.num_rows(); ++i) {
        if (lbl_col.is_null(i) || year_col.is_null(i) || week_col.is_null(i)) continue;
        int y = std::stoi(year_col.get_text(i));
        int w = std::stoi(week_col.get_text(i));
        if (y > max_year || (y == max_year && w > max_week)) { max_year = y; max_week = w; }
    }

    // Day order for display
    std::map<std::string, int> day_order = {
        {"Tuesday",1},{"Wednesday",2},{"Thursday",3},
        {"Friday",4},{"Saturday",5},{"Sunday",6}
    };

    struct ForecastRow {
        std::string day, slot;
        float pred, actual;
        int day_ord;
    };
    std::vector<ForecastRow> forecast;

    for (std::size_t i = 0; i < swl.num_rows(); ++i) {
        if (lbl_col.is_null(i) || year_col.is_null(i) || week_col.is_null(i)) continue;
        int y = std::stoi(year_col.get_text(i));
        int w = std::stoi(week_col.get_text(i));
        if (y != max_year || w != max_week) continue;
        int eid = entity_col.is_null(i) ? -1 : static_cast<int>(entity_col.get_numerical(i));
        auto it = entity_info.find(eid);
        std::string day_name = (it != entity_info.end()) ? it->second.first  : "?";
        std::string slot_lbl = (it != entity_info.end()) ? it->second.second : "?";
        forecast.push_back({day_name, slot_lbl, all_preds[i],
                            static_cast<float>(lbl_col.get_numerical(i)),
                            day_order.count(day_name) ? day_order[day_name] : 99});
    }
    std::sort(forecast.begin(), forecast.end(),
              [](const auto& a, const auto& b){
                  return a.day_ord < b.day_ord ||
                         (a.day_ord == b.day_ord && a.slot < b.slot);
              });

    std::cout << "\nStaffing forecast — week " << max_week << " / " << max_year << ":\n";
    std::cout << std::string(62, '-') << "\n";
    std::cout << std::left
              << std::setw(12) << "Day"
              << std::setw(22) << "Slot"
              << std::setw(14) << "Predicted"
              << "Actual\n";
    std::cout << std::string(62, '-') << "\n";
    for (const auto& r : forecast) {
        std::cout << std::left
                  << std::setw(12) << r.day
                  << std::setw(22) << r.slot
                  << std::fixed << std::setprecision(1)
                  << std::setw(14) << r.pred
                  << r.actual << "\n";
    }

    return 0;
}
