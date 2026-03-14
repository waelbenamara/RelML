#include "relml/FKDetector.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <unordered_set>

namespace relml {

bool FKDetector::name_matches_pk(
    const std::string& col_name,
    const std::string& table_name,
    const std::string& pk_col)
{
    // Direct name match with the PK column
    if (col_name == pk_col)
        return true;

    // Strip trailing 's' from table name to get singular form
    // e.g. "races" -> "race", "drivers" -> "driver"
    std::string singular = table_name;
    if (!singular.empty() && singular.back() == 's')
        singular.pop_back();

    // Check col_name == singular + "Id" (case-insensitive)
    std::string candidate = singular + "Id";
    if (col_name.size() == candidate.size()) {
        bool match = true;
        for (std::size_t i = 0; i < col_name.size(); ++i)
            if (std::tolower((unsigned char)col_name[i]) !=
                std::tolower((unsigned char)candidate[i])) {
                match = false;
                break;
            }
        if (match) return true;
    }

    return false;
}

double FKDetector::compute_coverage_numerical(
    const Column& src_col,
    const Column& dst_pk_col)
{
    std::unordered_set<double> pk_values;
    pk_values.reserve(dst_pk_col.size());
    for (std::size_t i = 0; i < dst_pk_col.size(); ++i)
        if (!dst_pk_col.is_null(i))
            pk_values.insert(dst_pk_col.get_numerical(i));

    std::size_t total   = 0;
    std::size_t matched = 0;
    for (std::size_t i = 0; i < src_col.size(); ++i) {
        if (src_col.is_null(i)) continue;
        ++total;
        if (pk_values.count(src_col.get_numerical(i)))
            ++matched;
    }
    return total == 0 ? 0.0 : static_cast<double>(matched) / total;
}

double FKDetector::compute_coverage_string(
    const Column& src_col,
    const Column& dst_pk_col)
{
    std::unordered_set<std::string> pk_values;
    pk_values.reserve(dst_pk_col.size());
    for (std::size_t i = 0; i < dst_pk_col.size(); ++i) {
        if (dst_pk_col.is_null(i)) continue;
        // PK may be CATEGORICAL or TEXT
        if (dst_pk_col.type == ColumnType::CATEGORICAL ||
            dst_pk_col.type == ColumnType::TEXT)
            pk_values.insert(dst_pk_col.get_categorical(i));
    }

    std::size_t total   = 0;
    std::size_t matched = 0;
    for (std::size_t i = 0; i < src_col.size(); ++i) {
        if (src_col.is_null(i)) continue;
        ++total;
        std::string val;
        if (src_col.type == ColumnType::CATEGORICAL ||
            src_col.type == ColumnType::TEXT)
            val = src_col.get_categorical(i);
        if (pk_values.count(val))
            ++matched;
    }
    return total == 0 ? 0.0 : static_cast<double>(matched) / total;
}

std::vector<DetectedFK> FKDetector::detect(Database& db) {
    std::vector<DetectedFK> detected;

    for (auto& [src_name, src_table] : db.tables) {
        for (auto& src_col : src_table.columns) {

            // PK columns are never FKs
            if (src_table.pkey_col && *src_table.pkey_col == src_col.name)
                continue;

            // Skip TIMESTAMP columns — they are never FK references
            if (src_col.type == ColumnType::TIMESTAMP)
                continue;

            // Skip columns already declared as FKs
            bool already_declared = false;
            for (const auto& fk : src_table.foreign_keys)
                if (fk.column == src_col.name) { already_declared = true; break; }
            if (already_declared) continue;

            // Try every other table's PK
            for (auto& [dst_name, dst_table] : db.tables) {
                if (dst_name == src_name)  continue;
                if (!dst_table.pkey_col)   continue;

                const std::string& pk_col_name = *dst_table.pkey_col;

                if (!name_matches_pk(src_col.name, dst_name, pk_col_name))
                    continue;

                const Column& dst_pk = dst_table.get_column(pk_col_name);

                // Types must be compatible for coverage to make sense
                bool src_numeric = (src_col.type == ColumnType::NUMERICAL);
                bool dst_numeric = (dst_pk.type  == ColumnType::NUMERICAL);
                bool src_string  = (src_col.type == ColumnType::CATEGORICAL ||
                                    src_col.type == ColumnType::TEXT);
                bool dst_string  = (dst_pk.type  == ColumnType::CATEGORICAL ||
                                    dst_pk.type  == ColumnType::TEXT);

                double cov = 0.0;
                if (src_numeric && dst_numeric)
                    cov = compute_coverage_numerical(src_col, dst_pk);
                else if (src_string && dst_string)
                    cov = compute_coverage_string(src_col, dst_pk);
                else
                    continue;  // type mismatch — not a valid FK

                if (cov >= COVERAGE_THRESHOLD) {
                    src_table.foreign_keys.push_back({src_col.name, dst_name});
                    detected.push_back({src_name, src_col.name, dst_name, pk_col_name, cov});
                }
            }
        }
    }

    return detected;
}

} // namespace relml