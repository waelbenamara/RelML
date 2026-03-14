#include "relml/graph/GraphBuilder.h"
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace relml {

// ---------------------------------------------------------------------------
// PK index builders — one for numerical PKs, one for string PKs
// ---------------------------------------------------------------------------

static std::unordered_map<double, int64_t>
build_numerical_pk_index(const Table& table) {
    const Column& pk = table.get_column(*table.pkey_col);
    std::unordered_map<double, int64_t> index;
    index.reserve(pk.size());
    for (std::size_t row = 0; row < pk.size(); ++row) {
        if (pk.is_null(row)) continue;
        index[pk.get_numerical(row)] = static_cast<int64_t>(row);
    }
    return index;
}

static std::unordered_map<std::string, int64_t>
build_string_pk_index(const Table& table) {
    const Column& pk = table.get_column(*table.pkey_col);
    std::unordered_map<std::string, int64_t> index;
    index.reserve(pk.size());
    for (std::size_t row = 0; row < pk.size(); ++row) {
        if (pk.is_null(row)) continue;
        index[pk.get_categorical(row)] = static_cast<int64_t>(row);
    }
    return index;
}

// ---------------------------------------------------------------------------
// GraphBuilder::build
// ---------------------------------------------------------------------------

HeteroGraph GraphBuilder::build(const Database& db) {
    HeteroGraph g;

    for (const auto& [name, table] : db.tables)
        g.num_nodes[name] = static_cast<int64_t>(table.num_rows());

    // Pre-build PK indices for every table that has a PK.
    // Each table gets either a numerical or a string index depending on
    // its PK column type.
    std::unordered_map<std::string,
        std::unordered_map<double, int64_t>>      num_pk_indices;
    std::unordered_map<std::string,
        std::unordered_map<std::string, int64_t>> str_pk_indices;

    for (const auto& [name, table] : db.tables) {
        if (!table.pkey_col) continue;
        const Column& pk = table.get_column(*table.pkey_col);
        if (pk.type == ColumnType::NUMERICAL)
            num_pk_indices[name] = build_numerical_pk_index(table);
        else if (pk.type == ColumnType::CATEGORICAL ||
                 pk.type == ColumnType::TEXT)
            str_pk_indices[name] = build_string_pk_index(table);
    }

    // Walk every FK and emit forward + reverse edges
    for (const auto& [src_name, src_table] : db.tables) {
        for (const auto& fk : src_table.foreign_keys) {
            const std::string& dst_name = fk.target_table;

            if (!db.has_table(dst_name)) {
                std::cerr << "Warning: FK target '" << dst_name
                          << "' not found, skipping\n";
                continue;
            }
            const Table& dst_table = db.get_table(dst_name);
            if (!dst_table.pkey_col) {
                std::cerr << "Warning: table '" << dst_name
                          << "' has no PK, skipping FK\n";
                continue;
            }

            const Column& fk_col = src_table.get_column(fk.column);
            const Column& dst_pk = dst_table.get_column(*dst_table.pkey_col);

            // Determine which index to use
            bool use_numerical = (fk_col.type == ColumnType::NUMERICAL &&
                                  dst_pk.type  == ColumnType::NUMERICAL);
            bool use_string    = ((fk_col.type == ColumnType::CATEGORICAL ||
                                   fk_col.type == ColumnType::TEXT) &&
                                  (dst_pk.type == ColumnType::CATEGORICAL ||
                                   dst_pk.type == ColumnType::TEXT));

            if (!use_numerical && !use_string) {
                std::cerr << "Warning: type mismatch on FK "
                          << src_name << "." << fk.column
                          << " -> " << dst_name << ", skipping\n";
                continue;
            }

            EdgeType fwd { src_name, fk.column, dst_name };
            EdgeType rev { dst_name, "rev_" + fk.column, src_name };

            EdgeIndex& fwd_ei = g.edge_index[fwd];
            EdgeIndex& rev_ei = g.edge_index[rev];

            std::size_t skipped = 0;

            if (use_numerical) {
                const auto& pk_idx = num_pk_indices.at(dst_name);
                for (std::size_t row = 0; row < fk_col.size(); ++row) {
                    if (fk_col.is_null(row)) { ++skipped; continue; }
                    auto it = pk_idx.find(fk_col.get_numerical(row));
                    if (it == pk_idx.end()) { ++skipped; continue; }
                    fwd_ei.add(static_cast<int64_t>(row), it->second);
                    rev_ei.add(it->second, static_cast<int64_t>(row));
                }
            } else {
                const auto& pk_idx = str_pk_indices.at(dst_name);
                for (std::size_t row = 0; row < fk_col.size(); ++row) {
                    if (fk_col.is_null(row)) { ++skipped; continue; }
                    auto it = pk_idx.find(fk_col.get_categorical(row));
                    if (it == pk_idx.end()) { ++skipped; continue; }
                    fwd_ei.add(static_cast<int64_t>(row), it->second);
                    rev_ei.add(it->second, static_cast<int64_t>(row));
                }
            }

            if (skipped > 0)
                std::cerr << "  Note: " << skipped
                          << " unresolved FK values in "
                          << src_name << "." << fk.column << "\n";
        }
    }

    return g;
}

} // namespace relml