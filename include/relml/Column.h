#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace relml {

enum class ColumnType {
    NUMERICAL,
    CATEGORICAL,
    TIMESTAMP,
    TEXT
};

std::string column_type_to_string(ColumnType t);

// Null is represented as std::monostate
using CellValue = std::variant<
    std::monostate,  // null
    double,          // NUMERICAL
    std::string,     // CATEGORICAL or TEXT
    int64_t          // TIMESTAMP (unix seconds)
>;

struct Column {
    std::string           name;
    ColumnType            type;
    std::vector<CellValue> data;

    Column() = default;
    Column(std::string name, ColumnType type)
        : name(std::move(name)), type(type) {}

    std::size_t size()                 const { return data.size(); }
    bool        is_null(std::size_t i) const;

    double      get_numerical  (std::size_t i) const;
    std::string get_categorical(std::size_t i) const;
    int64_t     get_timestamp  (std::size_t i) const;
    std::string get_text       (std::size_t i) const;
};

} // namespace relml
