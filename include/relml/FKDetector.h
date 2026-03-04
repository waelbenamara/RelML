#pragma once

#include "Database.h"
#include <vector>

namespace relml {

struct DetectedFK {
    std::string src_table;
    std::string src_column;
    std::string dst_table;
    std::string dst_column; // the PK column in dst_table
    double      coverage;   // fraction of non-null src values found in dst PK
};

class FKDetector {
public:
    // Minimum fraction of non-null values that must match the target PK.
    // 1.0 = strict (all values must exist in PK), lower = more permissive.
    static constexpr double COVERAGE_THRESHOLD = 0.99;

    // Detect all foreign keys across the database using:
    //   1. Name match: src column name == dst table PK column name
    //   2. Value coverage: >= COVERAGE_THRESHOLD of src values exist in dst PK
    // Detected FKs are written directly into each Table's foreign_keys field.
    static std::vector<DetectedFK> detect(Database& db);

private:
    static bool name_matches_pk(
        const std::string& col_name,
        const std::string& table_name,
        const std::string& pk_col
    );

    static double compute_coverage(
        const Column& src_col,
        const Column& dst_pk_col
    );
};

} // namespace relml