#include "relml/FKDetector.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <unordered_set>

namespace relml {

// Check if col_name is a plausible FK reference to dst_table with pk_col.
// Accepts:
//   exact match:          col_name == pk_col         (e.g. raceId == raceId)
//   singular table name:  col_name == singular + "Id" (e.g. raceId matches table "races")
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
            if (std::tolower(col_name[i]) != std::tolower(candidate[i])) {
                match = false; break;
            }
        if (match) return true;
    }

    return false;
}

double FKDetector::compute_coverage(
    const Column& src_col,
    const Column& dst_pk_col)
{
    // Build a set of all PK values
    std::unordered_set<double> pk_values;
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

    if (total == 0) return 0.0;
    return static_cast<double>(matched) / total;
}

std::vector<DetectedFK> FKDetector::detect(Database& db) {
    std::vector<DetectedFK> detected;

    for (auto& [src_name, src_table] : db.tables) {
        for (auto& src_col : src_table.columns) {

            // Only NUMERICAL non-PK columns can be FKs
            if (src_col.type != ColumnType::NUMERICAL) continue;
            if (src_table.pkey_col && *src_table.pkey_col == src_col.name) continue;

            // Already declared — skip
            bool already_declared = false;
            for (const auto& fk : src_table.foreign_keys)
                if (fk.column == src_col.name) { already_declared = true; break; }
            if (already_declared) continue;

            // Try every other table's PK
            for (auto& [dst_name, dst_table] : db.tables) {
                if (dst_name == src_name)     continue;
                if (!dst_table.pkey_col)      continue;

                const std::string& pk_col = *dst_table.pkey_col;

                if (!name_matches_pk(src_col.name, dst_name, pk_col))
                    continue;

                // Name matched — verify with value coverage
                const Column& dst_pk = dst_table.get_column(pk_col);
                double cov = compute_coverage(src_col, dst_pk);

                if (cov >= COVERAGE_THRESHOLD) {
                    src_table.foreign_keys.push_back({src_col.name, dst_name});
                    detected.push_back({src_name, src_col.name, dst_name, pk_col, cov});
                }
            }
        }
    }

    return detected;
}

} // namespace relml