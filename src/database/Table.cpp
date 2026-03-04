#include "relml/Table.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace relml {

std::size_t Table::num_rows() const {
    if (columns.empty()) return 0;
    return columns[0].size();
}

bool Table::has_column(const std::string& col_name) const {
    for (const auto& c : columns)
        if (c.name == col_name) return true;
    return false;
}

Column& Table::get_column(const std::string& col_name) {
    for (auto& c : columns)
        if (c.name == col_name) return c;
    throw std::runtime_error("Table '" + name + "': column '" + col_name + "' not found");
}

const Column& Table::get_column(const std::string& col_name) const {
    for (const auto& c : columns)
        if (c.name == col_name) return c;
    throw std::runtime_error("Table '" + name + "': column '" + col_name + "' not found");
}

void Table::add_column(Column col) {
    if (!columns.empty() && col.size() != num_rows())
        throw std::runtime_error("Column size mismatch in table '" + name + "'");
    columns.push_back(std::move(col));
}

void Table::print_schema() const {
    std::cout << "Table: " << name
              << " (" << num_rows() << " rows, "
              << num_cols() << " cols)\n";
    for (const auto& c : columns) {
        std::cout << "  " << std::setw(30) << std::left << c.name
                  << column_type_to_string(c.type);
        if (pkey_col && *pkey_col == c.name)  std::cout << "  [PK]";
        if (time_col  && *time_col  == c.name) std::cout << "  [TIME]";
        for (const auto& fk : foreign_keys)
            if (fk.column == c.name)
                std::cout << "  [FK -> " << fk.target_table << "]";
        std::cout << "\n";
    }
}

void Table::print_head(std::size_t n) const {
    n = std::min(n, num_rows());
    // header
    for (const auto& c : columns)
        std::cout << std::setw(15) << std::left << c.name << " ";
    std::cout << "\n" << std::string(columns.size() * 16, '-') << "\n";

    for (std::size_t row = 0; row < n; ++row) {
        for (const auto& c : columns) {
            if (c.is_null(row)) {
                std::cout << std::setw(15) << std::left << "NULL" << " ";
                continue;
            }
            switch (c.type) {
                case ColumnType::NUMERICAL:
                    std::cout << std::setw(15) << std::left << c.get_numerical(row); break;
                case ColumnType::CATEGORICAL:
                case ColumnType::TEXT:
                    std::cout << std::setw(15) << std::left << c.get_categorical(row); break;
                case ColumnType::TIMESTAMP:
                    std::cout << std::setw(15) << std::left << c.get_timestamp(row); break;
            }
            std::cout << " ";
        }
        std::cout << "\n";
    }
}

} // namespace relml
