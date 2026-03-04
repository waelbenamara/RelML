#pragma once

#include "Column.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace relml {

struct ForeignKey {
    std::string column;       // column in this table
    std::string target_table; // primary key table it points to
};

struct Table {
    std::string name;
    std::vector<Column> columns;

    std::optional<std::string> pkey_col;
    std::optional<std::string> time_col;
    std::vector<ForeignKey>    foreign_keys;

    Table() = default;
    explicit Table(std::string name) : name(std::move(name)) {}

    std::size_t num_rows() const;
    std::size_t num_cols() const { return columns.size(); }

    bool        has_column(const std::string& col_name) const;
    Column&       get_column(const std::string& col_name);
    const Column& get_column(const std::string& col_name) const;

    void add_column(Column col);
    void print_schema() const;
    void print_head(std::size_t n = 5) const;
};

} // namespace relml
