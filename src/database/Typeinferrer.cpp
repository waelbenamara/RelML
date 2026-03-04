#include "relml/Typeinferrer.h"
#include <regex>
#include <set>
#include <unordered_set>

namespace relml {

static const std::regex RE_INTEGER(R"(^-?\d+$)");
static const std::regex RE_FLOAT  (R"(^-?\d*\.?\d+([eE][+-]?\d+)?$)");
static const std::regex RE_DATE   (R"(^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$)");

static bool is_null_token(const std::string& s) {
    return s.empty() || s == "\\N" || s == "NA" || s == "nan" || s == "None" || s == "NULL";
}

bool TypeInferrer::is_numeric(const std::string& s) {
    return std::regex_match(s, RE_INTEGER) || std::regex_match(s, RE_FLOAT);
}

bool TypeInferrer::is_timestamp(const std::string& s) {
    return std::regex_match(s, RE_DATE);
}

ColumnType TypeInferrer::infer(
    const std::string& col_name,
    const std::vector<std::string>& samples)
{
    std::size_t n_numeric   = 0;
    std::size_t n_timestamp = 0;
    std::size_t n_valid     = 0;
    std::unordered_set<std::string> unique_vals;

    for (const auto& s : samples) {
        if (is_null_token(s)) continue;
        ++n_valid;
        if (is_numeric(s))   ++n_numeric;
        if (is_timestamp(s)) ++n_timestamp;
        if (unique_vals.size() <= 1000)
            unique_vals.insert(s);
    }

    if (n_valid == 0)
        return ColumnType::TEXT;

    if (n_numeric == n_valid)
        return ColumnType::NUMERICAL;

    if (n_timestamp == n_valid)
        return ColumnType::TIMESTAMP;

    // String column — decide CATEGORICAL vs TEXT by cardinality
    double cardinality_ratio = static_cast<double>(unique_vals.size()) / n_valid;
    if (cardinality_ratio <= CATEGORICAL_THRESHOLD)
        return ColumnType::CATEGORICAL;

    return ColumnType::TEXT;
}

std::vector<InferredColumn> TypeInferrer::infer_all(
    const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows)
{
    std::vector<InferredColumn> result;
    result.reserve(headers.size());

    for (std::size_t col = 0; col < headers.size(); ++col) {
        std::vector<std::string> samples;
        samples.reserve(rows.size());
        for (const auto& row : rows)
            if (col < row.size())
                samples.push_back(row[col]);

        ColumnType type = infer(headers[col], samples);
        result.push_back({headers[col], type});
    }

    return result;
}

} // namespace relml