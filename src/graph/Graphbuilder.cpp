#include "relml/graph/GraphBuilder.h"
#include <iostream>
#include <stdexcept>

namespace relml {

std::unordered_map<int64_t, int64_t> GraphBuilder::build_pk_index(const Table& table) {
    if (!table.pkey_col)
        throw std::runtime_error("GraphBuilder: table '" + table.name + "' has no pkey_col");

    const Column& pk = table.get_column(*table.pkey_col);
    std::unordered_map<int64_t, int64_t> index;
    index.reserve(pk.size());

    for (std::size_t row = 0; row < pk.size(); ++row) {
        if (pk.is_null(row)) continue;
        int64_t pk_val = static_cast<int64_t>(pk.get_numerical(row));
        index[pk_val]  = static_cast<int64_t>(row);
    }
    return index;
}

HeteroGraph GraphBuilder::build(const Database& db) {
    HeteroGraph g;

    // Register all node types with their row counts
    for (const auto& [name, table] : db.tables)
        g.num_nodes[name] = static_cast<int64_t>(table.num_rows());

    // Pre-build PK index for every table that has a PK
    std::unordered_map<std::string, std::unordered_map<int64_t,int64_t>> pk_indices;
    for (const auto& [name, table] : db.tables) {
        if (table.pkey_col)
            pk_indices[name] = build_pk_index(table);
    }

    // Walk every FK relationship and emit forward + reverse edges
    for (const auto& [src_name, src_table] : db.tables) {
        for (const auto& fk : src_table.foreign_keys) {
            const std::string& dst_name = fk.target_table;

            if (!db.has_table(dst_name)) {
                std::cerr << "Warning: FK target '" << dst_name << "' not found, skipping\n";
                continue;
            }
            const Table& dst_table = db.get_table(dst_name);
            if (!dst_table.pkey_col) {
                std::cerr << "Warning: table '" << dst_name << "' has no PK, skipping FK\n";
                continue;
            }

            const Column&  fk_col  = src_table.get_column(fk.column);
            const auto&    pk_idx  = pk_indices.at(dst_name);

            EdgeType fwd { src_name, fk.column, dst_name };
            EdgeType rev { dst_name, "rev_" + fk.column, src_name };

            EdgeIndex& fwd_ei = g.edge_index[fwd];
            EdgeIndex& rev_ei = g.edge_index[rev];

            std::size_t skipped = 0;
            for (std::size_t row = 0; row < fk_col.size(); ++row) {
                if (fk_col.is_null(row)) { ++skipped; continue; }

                int64_t fk_val = static_cast<int64_t>(fk_col.get_numerical(row));
                auto    it     = pk_idx.find(fk_val);
                if (it == pk_idx.end()) { ++skipped; continue; }

                int64_t src_row = static_cast<int64_t>(row);
                int64_t dst_row = it->second;

                fwd_ei.add(src_row, dst_row);
                rev_ei.add(dst_row, src_row);
            }

            if (skipped > 0)
                std::cerr << "  Note: " << skipped << " unresolved FK values in "
                          << src_name << "." << fk.column << "\n";
        }
    }

    return g;
}

} // namespace relml