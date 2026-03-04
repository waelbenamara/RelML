#include "relml/Column.h"
#include <stdexcept>

namespace relml {

std::string column_type_to_string(ColumnType t) {
    switch (t) {
        case ColumnType::NUMERICAL:   return "NUMERICAL";
        case ColumnType::CATEGORICAL: return "CATEGORICAL";
        case ColumnType::TIMESTAMP:   return "TIMESTAMP";
        case ColumnType::TEXT:        return "TEXT";
    }
    return "UNKNOWN";
}

bool Column::is_null(std::size_t i) const {
    return std::holds_alternative<std::monostate>(data[i]);
}

double Column::get_numerical(std::size_t i) const {
    if (is_null(i))
        throw std::runtime_error("Column::get_numerical: null at row " + std::to_string(i));
    return std::get<double>(data[i]);
}

std::string Column::get_categorical(std::size_t i) const {
    if (is_null(i))
        throw std::runtime_error("Column::get_categorical: null at row " + std::to_string(i));
    return std::get<std::string>(data[i]);
}

int64_t Column::get_timestamp(std::size_t i) const {
    if (is_null(i))
        throw std::runtime_error("Column::get_timestamp: null at row " + std::to_string(i));
    return std::get<int64_t>(data[i]);
}

std::string Column::get_text(std::size_t i) const {
    if (is_null(i))
        throw std::runtime_error("Column::get_text: null at row " + std::to_string(i));
    return std::get<std::string>(data[i]);
}

} // namespace relml
