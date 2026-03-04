#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/encoding/HeteroEncoder.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/gnn/HeteroGraphSAGE.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace relml;

int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./rel-f1-data";

    std::unordered_map<std::string, TableSchema> schemas = {
        {"drivers",               {.pkey_col = "driverId",           .time_col = std::nullopt, .foreign_keys = {}, .columns = {}}},
        {"circuits",              {.pkey_col = "circuitId",          .time_col = std::nullopt, .foreign_keys = {}, .columns = {{.name = "country", .type = ColumnType::CATEGORICAL}}}},
        {"constructors",          {.pkey_col = "constructorId",      .time_col = std::nullopt, .foreign_keys = {}, .columns = {{.name = "nationality", .type = ColumnType::CATEGORICAL}}}},
        {"races",                 {.pkey_col = "raceId",             .time_col = "date",       .foreign_keys = {}, .columns = {{.name = "year", .type = ColumnType::CATEGORICAL}, {.name = "name", .type = ColumnType::TEXT}}}},
        {"results",               {.pkey_col = "resultId",           .time_col = "date",       .foreign_keys = {}, .columns = {}}},
        {"qualifying",            {.pkey_col = "qualifyId",          .time_col = "date",       .foreign_keys = {}, .columns = {}}},
        {"standings",             {.pkey_col = "driverStandingsId",   .time_col = "date",       .foreign_keys = {}, .columns = {}}},
        {"constructor_standings", {.pkey_col = "constructorStandingsId", .time_col = "date",   .foreign_keys = {}, .columns = {}}},
        {"constructor_results",   {.pkey_col = "constructorResultsId",   .time_col = "date",   .foreign_keys = {}, .columns = {}}},
    };

    Database db = CSVLoader::load_database(data_dir, "rel-f1", schemas);
    FKDetector::detect(db);

    // Phase 2: graph
    HeteroGraph graph = GraphBuilder::build(db);

    // Phase 3: initial node features
    HeteroEncoder encoder(128);
    encoder.fit(db);
    auto x_dict = encoder.transform(db);

    // Phase 4: GNN
    std::cout << "\nBuilding HeteroGraphSAGE (2 layers, 128 channels)...\n";
    HeteroGraphSAGE gnn(128, 2, graph.node_types(), graph.edge_types());

    std::cout << "Forward pass...\n";
    auto h_dict = gnn.forward(x_dict, graph);

    // Verify shapes are preserved
    for (const auto& [nt, nf] : h_dict) {
        assert(nf.num_nodes == db.get_table(nt).num_rows());
        assert(nf.channels  == 128);
        assert(nf.data.size() == nf.num_nodes * 128);
        std::cout << "  " << std::setw(28) << std::left << nt
                  << nf.num_nodes << " x " << nf.channels << "\n";
    }

    // GNN should change the embeddings (message passing happened)
    const NodeFeatures& drv_in  = x_dict.at("drivers");
    const NodeFeatures& drv_out = h_dict.at("drivers");
    bool changed = false;
    for (std::size_t i = 0; i < drv_in.data.size(); ++i) {
        if (std::abs(drv_in.data[i] - drv_out.data[i]) > 1e-6f) {
            changed = true; break;
        }
    }
    assert(changed);

    // Two different drivers should have different final embeddings
    bool different = false;
    for (std::size_t c = 0; c < 128; ++c) {
        if (drv_out(0, c) != drv_out(1, c)) { different = true; break; }
    }
    assert(different);

    // Print a summary of driver 0's embedding (first 8 dims)
    std::cout << "\nDriver 0 embedding (first 8 dims after 2 SAGE layers):\n  ";
    for (std::size_t c = 0; c < 8; ++c)
        std::cout << std::fixed << std::setprecision(4) << drv_out(0, c) << " ";
    std::cout << "\n";

    std::cout << "\nPhase 4 OK\n";
    return 0;
}