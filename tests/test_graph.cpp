#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/graph/HeteroGraph.h"
#include <cassert>
#include <iostream>

using namespace relml;

int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./data/rel-f1-data";

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

    std::cout << "\nBuilding heterogeneous graph...\n";
    HeteroGraph g = GraphBuilder::build(db);

    g.print_summary();

    // Verify node counts match table row counts
    assert(g.num_nodes.at("drivers")      == 857);
    assert(g.num_nodes.at("races")        == 820);
    assert(g.num_nodes.at("results")      == 20323);
    assert(g.num_nodes.at("circuits")     == 77);
    assert(g.num_nodes.at("constructors") == 211);

    // Forward + reverse edges must be symmetric
    for (const auto& et : g.edge_types()) {
        if (et.fk_col.substr(0, 4) == "rev_") continue;
        EdgeType rev { et.dst, "rev_" + et.fk_col, et.src };
        assert(g.edge_index.count(rev) > 0);
        assert(g.edge_index.at(et).num_edges() == g.edge_index.at(rev).num_edges());
    }

    // results -> drivers: should have 20323 edges (one per result row)
    EdgeType res_drv { "results", "driverId", "drivers" };
    assert(g.edge_index.at(res_drv).num_edges() == 20323);

    // races -> circuits: should have 820 edges (one per race)
    EdgeType rac_cir { "races", "circuitId", "circuits" };
    assert(g.edge_index.at(rac_cir).num_edges() == 820);

    std::cout << "\nPhase 2 OK\n";
    return 0;
}