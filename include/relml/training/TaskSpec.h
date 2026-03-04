#pragma once

#include "relml/Database.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace relml {

// ---------------------------------------------------------------------------
// Shared types (previously in TaskBuilder.h)
// ---------------------------------------------------------------------------

struct TaskSample {
    int64_t node_idx;
    float   label;
};

struct TaskSplit {
    std::vector<TaskSample> train;
    std::vector<TaskSample> val;
    std::vector<TaskSample> test;
};

// ---------------------------------------------------------------------------
// LabelTransform
// ---------------------------------------------------------------------------

struct LabelTransform {
    enum class Kind { Threshold, Identity, Normalize, Buckets };

    Kind  kind      = Kind::Identity;
    float threshold = 0.f;
    bool  inclusive = true;              // >= vs >
    std::vector<float> buckets;          // sorted, for multiclass
    float norm_mean = 0.f;              // fitted from training rows
    float norm_std  = 1.f;

    float apply(float v) const;
};

// ---------------------------------------------------------------------------
// InferenceFilter — used for row-based inference
// ---------------------------------------------------------------------------

struct InferenceFilter {
    std::string column;
    std::string op;     // "=", "!=", ">=", ">", "<=", "<"
    std::string value;
};

// ---------------------------------------------------------------------------
// TaskSpec
// ---------------------------------------------------------------------------

struct TaskSpec {
    // Training fields (all included in fingerprint)
    std::string   target_table;
    std::string   target_column;

    enum class TaskType {
        BinaryClassification,
        Regression,
        MulticlassClassification
    };
    TaskType task_type = TaskType::BinaryClassification;

    LabelTransform label_transform;

    enum class SplitStrategy { Temporal, Random };
    SplitStrategy               split_strategy = SplitStrategy::Random;
    std::optional<std::string>  split_time_col;

    float train_frac = 0.70f;
    float val_frac   = 0.15f;

    // -----------------------------------------------------------------------
    // Inference fields (NOT in fingerprint — do not affect which model is trained)
    // -----------------------------------------------------------------------

    // How to perform inference:
    //   RowBased       — score existing rows, then filter with inference_filters
    //   EntitySynthesis — look up named entities by FK, mean-pool their GNN
    //                     embeddings, synthesize a single prediction
    enum class InferenceMode { RowBased, EntitySynthesis };
    InferenceMode inference_mode = InferenceMode::RowBased;

    // RowBased: applied as a conjunction over existing rows
    std::vector<InferenceFilter> inference_filters;

    enum class AggType { None, Mean, Fraction, Count };
    AggType inference_agg = AggType::None;

    // EntitySynthesis: fk_column_in_target_table → entity_id (as string)
    // e.g. {"userId": "5", "movieId": "56"}
    std::unordered_map<std::string, std::string> entity_refs;

    // -----------------------------------------------------------------------
    // Methods
    // -----------------------------------------------------------------------

    // Stable hash of training-relevant fields only.
    std::string fingerprint() const;

    // Output dimension for the MLP head.
    std::size_t output_dim() const;

    // Build train/val/test split from the database.
    TaskSplit build_split(const Database& db);

    // RowBased inference result
    struct InferenceResult {
        std::vector<int64_t>   row_indices;
        std::vector<float>     predictions;
        std::optional<float>   aggregate;
    };

    // Apply row-based filters and aggregation to a flat prediction vector.
    InferenceResult apply_inference(
        const Database&           db,
        const std::vector<float>& all_preds) const;
};

} // namespace relml