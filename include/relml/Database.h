#pragma once

#include "Table.h"
#include <string>
#include <unordered_map>

namespace relml {

struct Database {
    std::string name;
    std::unordered_map<std::string, Table> tables;

    Database() = default;
    explicit Database(std::string name) : name(std::move(name)) {}

    void        add_table(Table t);
    bool        has_table(const std::string& name) const;
    Table&       get_table(const std::string& name);
    const Table& get_table(const std::string& name) const;

    std::vector<std::string> table_names() const;
    void print_schema() const;
};

} // namespace relml
