#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/encoding/HeteroEncoder.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>

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

    std::cout << "\nFitting HeteroEncoder (channels=128)...\n";
    HeteroEncoder encoder(128);
    encoder.fit(db);

    std::cout << "Transforming...\n";
    auto features = encoder.transform(db);

    // Verify shapes
    for (const auto& [node_type, nf] : features) {
        assert(nf.num_nodes  == db.get_table(node_type).num_rows());
        assert(nf.channels   == 128);
        assert(nf.data.size() == nf.num_nodes * nf.channels);
        std::cout << "  " << std::setw(28) << std::left << node_type
                  << nf.num_nodes << " x " << nf.channels << "\n";
    }

    // Verify drivers features are not all zero (encoding produced something)
    const NodeFeatures& drv = features.at("drivers");
    float sum = 0.f;
    for (float v : drv.data) sum += std::abs(v);
    assert(sum > 0.f);

    // Verify results features have correct row count
    assert(features.at("results").num_nodes == 20323);

    // Spot-check: two different driver rows should have different embeddings
    bool different = false;
    for (std::size_t c = 0; c < 128; ++c) {
        if (drv(0, c) != drv(1, c)) { different = true; break; }
    }
    assert(different);

    std::cout << "\nPhase 3 OK\n";
    return 0;
}
