#include "relml/CSVLoader.h"
#include "relml/Database.h"
#include "relml/FKDetector.h"
#include <cassert>
#include <iostream>

using namespace relml;

int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./data/rel-f1-data";

    // Only pkey and time_col need to be declared.
    // Column types and foreign keys are fully automatic.
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

    // Detect foreign keys automatically
    std::cout << "\nDetecting foreign keys...\n";
    auto detected = FKDetector::detect(db);
    for (const auto& fk : detected)
        std::cout << "  " << fk.src_table << "." << fk.src_column
                  << " -> " << fk.dst_table
                  << "  (coverage " << fk.coverage * 100 << "%)\n";

    std::cout << "\n";
    db.print_schema();

    // Verify inferred types
    const Table& drivers = db.get_table("drivers");
    assert(drivers.get_column("driverId").type    == ColumnType::NUMERICAL);
    assert(drivers.get_column("forename").type    == ColumnType::TEXT);
    assert(drivers.get_column("nationality").type == ColumnType::CATEGORICAL);
    assert(drivers.get_column("dob").type         == ColumnType::TIMESTAMP);

    // Verify detected FKs
    const Table& results = db.get_table("results");
    assert(results.foreign_keys.size() >= 3);

    std::cout << "drivers[0].forename    = " << drivers.get_column("forename").get_text(0)           << "\n";
    std::cout << "drivers[0].nationality = " << drivers.get_column("nationality").get_categorical(0) << "\n";

    std::cout << "\nPhase 1 OK\n";
    return 0;
}