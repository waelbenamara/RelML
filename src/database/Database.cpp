#include "relml/Database.h"
#include <iostream>
#include <stdexcept>

namespace relml {

void Database::add_table(Table t) {
    std::string n = t.name;
    tables.emplace(n, std::move(t));
}

bool Database::has_table(const std::string& n) const {
    return tables.count(n) > 0;
}

Table& Database::get_table(const std::string& n) {
    auto it = tables.find(n);
    if (it == tables.end())
        throw std::runtime_error("Database: table '" + n + "' not found");
    return it->second;
}

const Table& Database::get_table(const std::string& n) const {
    auto it = tables.find(n);
    if (it == tables.end())
        throw std::runtime_error("Database: table '" + n + "' not found");
    return it->second;
}

std::vector<std::string> Database::table_names() const {
    std::vector<std::string> names;
    names.reserve(tables.size());
    for (const auto& [k, _] : tables)
        names.push_back(k);
    return names;
}

void Database::print_schema() const {
    std::cout << "Database: " << name
              << " (" << tables.size() << " tables)\n"
              << std::string(60, '=') << "\n";
    for (const auto& [_, t] : tables) {
        t.print_schema();
        std::cout << "\n";
    }
}

} // namespace relml
