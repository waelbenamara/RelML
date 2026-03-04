#include "relml/CSVLoader.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace relml {

static bool is_null_token(const std::string& s) {
    return s.empty() || s == "\\N" || s == "NA" || s == "nan"
        || s == "None" || s == "NULL";
}

static CellValue parse_cell(const std::string& token, ColumnType type) {
    if (is_null_token(token)) return std::monostate{};

    switch (type) {
        case ColumnType::NUMERICAL: {
            try   { return std::stod(token); }
            catch (...) { return std::monostate{}; }
        }
        case ColumnType::CATEGORICAL:
        case ColumnType::TEXT:
            return token;

        case ColumnType::TIMESTAMP: {
            std::tm tm{};
            std::istringstream ss(token);
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
            if (ss.fail()) {
                std::istringstream ss2(token);
                ss2 >> std::get_time(&tm, "%Y-%m-%d");
                if (ss2.fail()) return std::monostate{};
            }
            tm.tm_isdst = -1;
            return static_cast<int64_t>(std::mktime(&tm));
        }
    }
    return std::monostate{};
}

std::vector<std::string> CSVLoader::split_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"'; ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

Table CSVLoader::load_table(
    const std::string& filepath,
    const std::string& table_name,
    const TableSchema& schema)
{
    std::ifstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error("CSVLoader: cannot open: " + filepath);

    std::string header_line;
    std::getline(file, header_line);
    auto headers = split_line(header_line);

    // Read all raw rows for type inference
    std::vector<std::vector<std::string>> raw_rows;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty())
            raw_rows.push_back(split_line(line));
    }

    // Build override map: column_name -> explicit type
    std::unordered_map<std::string, ColumnType> overrides;
    for (const auto& co : schema.columns)
        if (co.type.has_value())
            overrides[co.name] = *co.type;

    // Infer types for all columns, then apply overrides where provided
    auto inferred = TypeInferrer::infer_all(headers, raw_rows);
    for (auto& ic : inferred) {
        auto it = overrides.find(ic.name);
        if (it != overrides.end())
            ic.type = it->second;
    }

    // Construct table
    Table table(table_name);
    table.pkey_col    = schema.pkey_col;
    table.time_col    = schema.time_col;
    table.foreign_keys = schema.foreign_keys;

    for (const auto& ic : inferred)
        table.columns.emplace_back(ic.name, ic.type);

    for (const auto& row : raw_rows) {
        for (std::size_t i = 0; i < inferred.size(); ++i) {
            const std::string& token = (i < row.size()) ? row[i] : "";
            table.columns[i].data.push_back(parse_cell(token, inferred[i].type));
        }
    }

    return table;
}

Database CSVLoader::load_database(
    const std::string& dir,
    const std::string& db_name,
    const std::unordered_map<std::string, TableSchema>& schemas)
{
    Database db(db_name);

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".csv") continue;

        std::string table_name = entry.path().stem().string();
        std::string filepath   = entry.path().string();

        TableSchema schema;
        auto it = schemas.find(table_name);
        if (it != schemas.end()) schema = it->second;

        std::cout << "Loading " << filepath << " ... ";
        Table t = load_table(filepath, table_name, schema);
        std::cout << t.num_rows() << " rows\n";
        db.add_table(std::move(t));
    }

    return db;
}

} // namespace relml