#pragma once

#include "Column.h"
#include <string>
#include <vector>

namespace relml {

struct InferredColumn {
    std::string name;
    ColumnType  type;
};

class TypeInferrer {
public:
    // Ratio of unique values to total rows below which a string column
    // is considered CATEGORICAL rather than TEXT.
    static constexpr double CATEGORICAL_THRESHOLD = 0.05;

    // Infer the type of a single column from a sample of raw string values.
    static ColumnType infer(
        const std::string& col_name,
        const std::vector<std::string>& samples
    );

    // Infer types for all columns given header + raw rows.
    static std::vector<InferredColumn> infer_all(
        const std::vector<std::string>& headers,
        const std::vector<std::vector<std::string>>& rows
    );

private:
    static bool is_numeric   (const std::string& s);
    static bool is_timestamp (const std::string& s);
};

} // namespace relml