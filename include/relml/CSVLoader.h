#pragma once

#include "Database.h"
#include "Typeinferrer.h"
#include <optional>
#include <string>
#include <unordered_map>

namespace relml {

struct ColumnOverride {
    std::string               name;
    std::optional<ColumnType> type; // nullopt = infer automatically
};

struct TableSchema {
    std::optional<std::string> pkey_col;
    std::optional<std::string> time_col;
    std::vector<ForeignKey>    foreign_keys;
    std::vector<ColumnOverride> columns; // only needed when inference is wrong
};

class CSVLoader {
public:
    static Table load_table(
        const std::string& filepath,
        const std::string& table_name,
        const TableSchema& schema = {}
    );

    static Database load_database(
        const std::string& dir,
        const std::string& db_name,
        const std::unordered_map<std::string, TableSchema>& schemas = {}
    );

private:
    static std::vector<std::string> split_line(const std::string& line);
};

} // namespace relml