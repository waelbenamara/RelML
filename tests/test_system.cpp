#include "relml/agent/RelMLSystem.h"
#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace relml;

// ---------------------------------------------------------------------------
// Schema definitions
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, TableSchema> ml1m_schemas() {
    return {
        {"users", {
            .pkey_col     = "userId",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                {.name = "gender",     .type = ColumnType::CATEGORICAL},
                {.name = "occupation", .type = ColumnType::CATEGORICAL},
            }
        }},
        {"movies", {
            .pkey_col     = "movieId",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                {.name = "title",  .type = ColumnType::TEXT},
                {.name = "genres", .type = ColumnType::CATEGORICAL},
            }
        }},
        {"ratings", {
            .pkey_col     = "ratingId",
            .time_col     = "timestamp",
            .foreign_keys = {},
            .columns      = {},
        }},
    };
}

static std::unordered_map<std::string, TableSchema> f1_schemas() {
    return {
        {"drivers", {
            .pkey_col     = "driverId",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {},
        }},
        {"circuits", {
            .pkey_col     = "circuitId",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                {.name = "country", .type = ColumnType::CATEGORICAL},
            }
        }},
        {"constructors", {
            .pkey_col     = "constructorId",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns      = {
                {.name = "nationality", .type = ColumnType::CATEGORICAL},
            }
        }},
        {"races", {
            .pkey_col     = "raceId",
            .time_col     = "date",
            .foreign_keys = {},
            .columns      = {
                {.name = "year", .type = ColumnType::CATEGORICAL},
                {.name = "name", .type = ColumnType::TEXT},
            }
        }},
        {"results", {
            .pkey_col     = "resultId",
            .time_col     = "date",
            .foreign_keys = {},
            .columns      = {},
        }},
        {"qualifying", {
            .pkey_col     = "qualifyId",
            .time_col     = "date",
            .foreign_keys = {},
            .columns      = {},
        }},
        {"standings", {
            .pkey_col     = "driverStandingsId",
            .time_col     = "date",
            .foreign_keys = {},
            .columns      = {},
        }},
        {"constructor_standings", {
            .pkey_col     = "constructorStandingsId",
            .time_col     = "date",
            .foreign_keys = {},
            .columns      = {},
        }},
        {"constructor_results", {
            .pkey_col     = "constructorResultsId",
            .time_col     = "date",
            .foreign_keys = {},
            .columns      = {},
        }},
    };
}

// ---------------------------------------------------------------------------
// Dataset detection
// ---------------------------------------------------------------------------

enum class Dataset { ML1M, F1, Unknown };

static Dataset detect_dataset(const std::string& dir) {
    namespace fs = std::filesystem;

    bool has_ratings     = fs::exists(dir + "/ratings.csv");
    bool has_movies      = fs::exists(dir + "/movies.csv");
    bool has_drivers     = fs::exists(dir + "/drivers.csv");
    bool has_results     = fs::exists(dir + "/results.csv");

    if (has_ratings && has_movies) return Dataset::ML1M;
    if (has_drivers && has_results) return Dataset::F1;
    return Dataset::Unknown;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./data";

    // Strip trailing slash for consistent path joins
    while (data_dir.size() > 1 && data_dir.back() == '/')
        data_dir.pop_back();

    Dataset ds = detect_dataset(data_dir);

    std::unordered_map<std::string, TableSchema> schemas;
    std::string db_name;

    switch (ds) {
        case Dataset::ML1M:
            schemas = ml1m_schemas();
            db_name = "ml-1m";
            break;
        case Dataset::F1:
            schemas = f1_schemas();
            db_name = "rel-f1";
            break;
        case Dataset::Unknown:
            std::cerr << "Could not detect dataset in '" << data_dir << "'.\n"
                      << "Expected either ml-1m (ratings.csv + movies.csv) "
                      << "or rel-f1 (drivers.csv + results.csv).\n";
            return 1;
    }

    std::cout << "Dataset : " << db_name << "\n"
              << "Loading database from: " << data_dir << "\n";

    Database    db    = CSVLoader::load_database(data_dir, db_name, schemas);
    FKDetector::detect(db);
    HeteroGraph graph = GraphBuilder::build(db);
    graph.print_summary();

    RelMLSystem system(db, graph);

    std::cout << "\nEnter natural language queries (empty line or Ctrl-D to quit).\n"
              << "Registry: " << system.registry().root() << "\n\n";

    if (argc > 2) {
        for (int i = 2; i < argc; ++i) {
            std::cout << std::string(72, '=') << "\n"
                      << "Query: " << argv[i] << "\n"
                      << std::string(72, '=') << "\n";
            try {
                std::cout << system.query(argv[i]) << "\n\n";
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n\n";
            }
        }
    } else {
        std::string line;
        while (true) {
            std::cout << "Query> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line == "quit" || line == "exit") break;
            try {
                std::cout << "\n" << system.query(line) << "\n\n";
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n\n";
            }
        }
    }

    std::cout << "Done.\n";
    return 0;
}