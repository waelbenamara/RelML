#pragma once

#include "HeteroGraph.h"
#include "relml/Database.h"
#include <unordered_map>

namespace relml {

class GraphBuilder {
public:
    // Build a HeteroGraph from a fully-loaded Database.
    // Requires that FK detection has already been run (Table::foreign_keys populated).
    // For each FK relationship, both forward and reverse edges are added.
    static HeteroGraph build(const Database& db);

private:
    // Map from PK value -> row index for a given table's PK column.
    // PK values are stored as doubles (NUMERICAL type) in our Column.
    static std::unordered_map<int64_t, int64_t> build_pk_index(
        const Table& table
    );
};

} // namespace relml