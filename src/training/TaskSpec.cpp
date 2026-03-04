#include "relml/training/TaskSpec.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace relml {

// ---------------------------------------------------------------------------
// LabelTransform
// ---------------------------------------------------------------------------

float LabelTransform::apply(float v) const {
    switch (kind) {
        case Kind::Threshold:
            return (inclusive ? v >= threshold : v > threshold) ? 1.f : 0.f;
        case Kind::Identity:
            return v;
        case Kind::Normalize:
            return (v - norm_mean) / norm_std;
        case Kind::Buckets: {
            std::size_t cls = 0;
            for (float b : buckets) {
                if (v >= b) ++cls;
                else break;
            }
            return static_cast<float>(cls);
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
// TaskSpec helpers
// ---------------------------------------------------------------------------

std::size_t TaskSpec::output_dim() const {
    switch (task_type) {
        case TaskType::BinaryClassification: return 1;
        case TaskType::Regression:           return 1;
        case TaskType::MulticlassClassification:
            return label_transform.buckets.size() + 1;
    }
    return 1;
}

// Stable canonical string — no inference-time fields.
std::string TaskSpec::fingerprint() const {
    std::ostringstream ss;
    ss << target_table << "|" << target_column << "|";

    switch (task_type) {
        case TaskType::BinaryClassification:      ss << "bin"; break;
        case TaskType::Regression:                ss << "reg"; break;
        case TaskType::MulticlassClassification:  ss << "cls"; break;
    }
    ss << "|";

    switch (label_transform.kind) {
        case LabelTransform::Kind::Threshold:
            ss << "thr:" << label_transform.threshold
               << (label_transform.inclusive ? ">=" : ">");
            break;
        case LabelTransform::Kind::Identity:  ss << "id";  break;
        case LabelTransform::Kind::Normalize: ss << "nrm"; break;
        case LabelTransform::Kind::Buckets:
            ss << "bkt:";
            for (float b : label_transform.buckets) ss << b << ",";
            break;
    }
    ss << "|";

    switch (split_strategy) {
        case SplitStrategy::Temporal: ss << "temp"; break;
        case SplitStrategy::Random:   ss << "rand"; break;
    }

    // Simple djb2 hash for a compact, filesystem-safe identifier.
    std::string canon = ss.str();
    uint64_t h = 5381;
    for (unsigned char c : canon) h = h * 33 ^ c;

    std::ostringstream hex;
    hex << std::hex << h;
    return hex.str();
}

// ---------------------------------------------------------------------------
// TaskSpec::build_split
// ---------------------------------------------------------------------------

TaskSplit TaskSpec::build_split(const Database& db) {
    const Table& table = db.get_table(target_table);
    if (!table.has_column(target_column))
        throw std::runtime_error("TaskSpec: column '" + target_column
                                 + "' not found in table '" + target_table + "'");

    const Column& label_col = table.get_column(target_column);

    // Temporal split requires a time column
    const Column* time_col_ptr = nullptr;
    if (split_strategy == SplitStrategy::Temporal) {
        std::string tcol = split_time_col
            ? *split_time_col
            : (table.time_col ? *table.time_col : "");
        if (tcol.empty())
            throw std::runtime_error(
                "TaskSpec: temporal split requested but no time_col for '"
                + target_table + "'");
        time_col_ptr = &table.get_column(tcol);
    }

    struct Row { int64_t idx; int64_t ts; float raw; };
    std::vector<Row> valid_rows;
    valid_rows.reserve(table.num_rows());

    for (std::size_t i = 0; i < table.num_rows(); ++i) {
        if (label_col.is_null(i)) continue;

        float raw = 0.f;
        if (label_col.type == ColumnType::NUMERICAL)
            raw = static_cast<float>(label_col.get_numerical(i));
        else
            throw std::runtime_error(
                "TaskSpec: target column '" + target_column + "' must be NUMERICAL");

        int64_t ts = 0;
        if (time_col_ptr) {
            if (time_col_ptr->is_null(i)) continue;
            // Timestamps may be stored as NUMERICAL (unix int) or TIMESTAMP
            if (time_col_ptr->type == ColumnType::NUMERICAL)
                ts = static_cast<int64_t>(time_col_ptr->get_numerical(i));
            else
                ts = time_col_ptr->get_timestamp(i);
        }

        valid_rows.push_back({static_cast<int64_t>(i), ts, raw});
    }

    if (split_strategy == SplitStrategy::Temporal)
        std::sort(valid_rows.begin(), valid_rows.end(),
                  [](const Row& a, const Row& b){ return a.ts < b.ts; });
    else {
        // Deterministic Fisher-Yates with a fixed seed.
        // valid_rows already holds the correct row indices from the collection phase.
        uint64_t state = 0x9e3779b97f4a7c15ULL;  // fixed seed
        for (std::size_t i = valid_rows.size() - 1; i > 0; --i) {
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
            uint64_t j = (state * 0x2545F4914F6CDD1DULL) % (i + 1);
            std::swap(valid_rows[i], valid_rows[j]);
        }
    }

    // Fit normalization stats from the training portion
    if (label_transform.kind == LabelTransform::Kind::Normalize) {
        std::size_t n_train = static_cast<std::size_t>(valid_rows.size() * train_frac);
        double sum = 0.0, sum_sq = 0.0;
        for (std::size_t i = 0; i < n_train; ++i) {
            sum    += valid_rows[i].raw;
            sum_sq += valid_rows[i].raw * valid_rows[i].raw;
        }
        float mean = static_cast<float>(sum / n_train);
        float var  = static_cast<float>(sum_sq / n_train) - mean * mean;
        label_transform.norm_mean = mean;
        label_transform.norm_std  = (var > 1e-8f) ? std::sqrt(var) : 1.f;
    }

    std::size_t N   = valid_rows.size();
    std::size_t t70 = static_cast<std::size_t>(N * train_frac);
    std::size_t t85 = static_cast<std::size_t>(N * (train_frac + val_frac));

    TaskSplit split;
    for (std::size_t i = 0; i < N; ++i) {
        TaskSample s{valid_rows[i].idx, label_transform.apply(valid_rows[i].raw)};
        if      (i < t70) split.train.push_back(s);
        else if (i < t85) split.val.push_back(s);
        else              split.test.push_back(s);
    }
    return split;
}

// ---------------------------------------------------------------------------
// TaskSpec::apply_inference
// ---------------------------------------------------------------------------

TaskSpec::InferenceResult TaskSpec::apply_inference(
    const Database& db,
    const std::vector<float>& all_preds) const
{
    const Table& table = db.get_table(target_table);
    InferenceResult result;

    // Evaluate a single filter against a row
    auto passes = [&](std::size_t row) -> bool {
        for (const auto& f : inference_filters) {
            if (!table.has_column(f.column)) continue;
            const Column& col = table.get_column(f.column);
            if (col.is_null(row)) return false;

            float row_val = 0.f;
            if (col.type == ColumnType::NUMERICAL)
                row_val = static_cast<float>(col.get_numerical(row));
            else if (col.type == ColumnType::CATEGORICAL)
                row_val = 0.f;  // string comparison handled below

            // String equality
            if (col.type == ColumnType::CATEGORICAL || col.type == ColumnType::TEXT) {
                std::string row_str = col.get_categorical(row);
                if (f.op == "="  && row_str != f.value) return false;
                if (f.op == "!=" && row_str == f.value) return false;
                continue;
            }

            float fv = std::stof(f.value);
            if (f.op == "="  && row_val != fv) return false;
            if (f.op == "!=" && row_val == fv) return false;
            if (f.op == ">=" && row_val <  fv) return false;
            if (f.op == ">"  && row_val <= fv) return false;
            if (f.op == "<=" && row_val >  fv) return false;
            if (f.op == "<"  && row_val >= fv) return false;
        }
        return true;
    };

    for (std::size_t i = 0; i < table.num_rows(); ++i) {
        if (i >= all_preds.size()) break;
        if (!passes(i)) continue;
        result.row_indices.push_back(static_cast<int64_t>(i));
        result.predictions.push_back(all_preds[i]);
    }

    if (inference_agg != AggType::None && !result.predictions.empty()) {
        float val = 0.f;
        switch (inference_agg) {
            case AggType::Mean:
                for (float p : result.predictions) val += p;
                val /= result.predictions.size();
                break;
            case AggType::Fraction:
                for (float p : result.predictions) val += (p >= 0.5f ? 1.f : 0.f);
                val /= result.predictions.size();
                break;
            case AggType::Count:
                for (float p : result.predictions) val += (p >= 0.5f ? 1.f : 0.f);
                break;
            case AggType::None: break;
        }
        result.aggregate = val;
    }

    return result;
}

} // namespace relml
