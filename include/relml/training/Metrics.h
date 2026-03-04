#pragma once

#include <string>
#include <vector>

namespace relml {

// ---------------------------------------------------------------------------
// EvalMetrics — single struct covering classification AND regression.
// Classification fields are zero for regression tasks and vice versa.
// ---------------------------------------------------------------------------

struct EvalMetrics {
    // Classification
    float average_precision = 0.f;
    float roc_auc           = 0.f;
    float accuracy          = 0.f;
    float f1                = 0.f;

    // Regression
    float rmse = 0.f;
    float mae  = 0.f;
    float r2   = 0.f;

    // Common
    float loss = 0.f;

    void print(const std::string& prefix) const;
};

// Binary classification metrics (AP, ROC-AUC, F1, accuracy).
EvalMetrics compute_metrics(
    const std::vector<float>& probs,
    const std::vector<float>& targets,
    float threshold = 0.5f);

// Regression metrics (RMSE, MAE, R²).
EvalMetrics compute_regression_metrics(
    const std::vector<float>& preds,
    const std::vector<float>& targets);

// Multiclass metrics (accuracy and macro-averaged F1).
// logits is N*K (raw scores), targets contains class indices as floats.
EvalMetrics compute_multiclass_metrics(
    const std::vector<float>& logits,
    const std::vector<float>& targets,
    std::size_t K);

} // namespace relml