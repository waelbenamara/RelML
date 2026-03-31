#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/training/Trainer.h"
#include "relml/training/TaskBuilder.h"
#include <iostream>

using namespace relml;

int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./data/ml-1m-data";

    std::unordered_map<std::string, TableSchema> schemas = {
        {"users",   {.pkey_col = "userId",
                     .time_col = std::nullopt,
                     .foreign_keys = {},
                     .columns  = {{.name = "gender",     .type = ColumnType::CATEGORICAL},
                                  {.name = "occupation", .type = ColumnType::CATEGORICAL}}}},
        {"movies",  {.pkey_col = "movieId",
                     .time_col = std::nullopt,
                     .foreign_keys = {},
                     .columns  = {{.name = "title",  .type = ColumnType::TEXT},
                                  {.name = "genres", .type = ColumnType::CATEGORICAL}}}},
        {"ratings", {.pkey_col = "ratingId",
                     .time_col = "timestamp",
                     .foreign_keys = {},
                     .columns = {}}},
    };

    Database db = CSVLoader::load_database(data_dir, "ml-1m", schemas);
    FKDetector::detect(db);
    HeteroGraph graph = GraphBuilder::build(db);

    std::cout << "\nBuilding rating task (>= 4 stars)...\n";
    TaskSplit task = build_rating_task(db, 4.f);

    TrainConfig cfg;
    cfg.channels    = 128;
    cfg.gnn_layers  = 2;
    cfg.hidden      = 64;
    cfg.dropout     = 0.3f;
    cfg.lr          = 3e-4f;
    cfg.pos_weight  = 1.f;
    cfg.epochs      = 20;
    cfg.target_node = "ratings";
    cfg.batch_size  = 0;  // mini-batch (0 = full batch); encoder+GNN once/epoch, faster on large data

    std::cout << "\nBuilding Trainer...\n";
    Trainer trainer(cfg, db, graph);
    trainer.fit(task, db, graph);

    return 0;
}