#include "relml/training/Metrics.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace relml {

void EvalMetrics::print(const std::string& prefix) const {
    std::cout << prefix << "\n"
              << "  loss              : " << std::fixed << std::setprecision(4) << loss << "\n";

    // Binary classification fields
    if (average_precision > 0.f || roc_auc > 0.f) {
        std::cout << "  average_precision : " << average_precision << "\n"
                  << "  roc_auc           : " << roc_auc           << "\n"
                  << "  accuracy          : " << accuracy          << "\n"
                  << "  f1                : " << f1                << "\n";
    }
    // Multiclass fields (accuracy/f1 populated but AP/AUC are zero)
    else if (accuracy > 0.f || f1 > 0.f) {
        std::cout << "  accuracy          : " << accuracy          << "\n"
                  << "  f1                : " << f1                << "\n";
    }

    // Regression fields
    if (rmse > 0.f || mae > 0.f) {
        std::cout << "  rmse              : " << rmse     << "\n"
                  << "  mae               : " << mae      << "\n"
                  << "  r2                : " << r2       << "\n";
    }
}

// ---------------------------------------------------------------------------
// Binary classification
// ---------------------------------------------------------------------------

EvalMetrics compute_metrics(
    const std::vector<float>& probs,
    const std::vector<float>& targets,
    float threshold)
{
    std::size_t N = probs.size();

    std::vector<std::size_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b){
        return probs[a] > probs[b];
    });

    float n_pos = 0.f, n_neg = 0.f;
    for (float t : targets) (t > 0.5f ? n_pos : n_neg) += 1.f;

    // ROC-AUC (trapezoidal)
    float tp = 0.f, fp = 0.f, prev_fpr = 0.f, prev_tpr = 0.f, auc = 0.f;
    for (std::size_t i : order) {
        if (targets[i] > 0.5f) tp += 1.f;
        else                    fp += 1.f;
        float tpr = (n_pos > 0) ? tp / n_pos : 0.f;
        float fpr = (n_neg > 0) ? fp / n_neg : 0.f;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.f;
        prev_fpr = fpr; prev_tpr = tpr;
    }

    // Average Precision
    float ap = 0.f, cum_tp = 0.f;
    for (std::size_t k = 0; k < N; ++k) {
        if (targets[order[k]] > 0.5f) {
            cum_tp += 1.f;
            ap += cum_tp / static_cast<float>(k + 1);
        }
    }
    ap = (n_pos > 0) ? ap / n_pos : 0.f;

    // Accuracy / F1
    float tp2 = 0.f, fp2 = 0.f, fn2 = 0.f, tn2 = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        int pred = probs[i] >= threshold ? 1 : 0;
        int gt   = targets[i] > 0.5f    ? 1 : 0;
        if (pred == 1 && gt == 1) tp2 += 1.f;
        if (pred == 1 && gt == 0) fp2 += 1.f;
        if (pred == 0 && gt == 1) fn2 += 1.f;
        if (pred == 0 && gt == 0) tn2 += 1.f;
    }
    float accuracy = (tp2 + tn2) / static_cast<float>(N);
    float f1       = (2*tp2 + fp2 + fn2 > 0) ? 2*tp2 / (2*tp2 + fp2 + fn2) : 0.f;

    return EvalMetrics{ap, auc, accuracy, f1, 0.f, 0.f, 0.f, 0.f};
}

// ---------------------------------------------------------------------------
// Regression
// ---------------------------------------------------------------------------

EvalMetrics compute_regression_metrics(
    const std::vector<float>& preds,
    const std::vector<float>& targets)
{
    std::size_t N = preds.size();
    if (N == 0) return {};

    double sum_e  = 0.0, sum_ae = 0.0, sum_se = 0.0;
    double sum_y  = 0.0, sum_y2 = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        double e  = preds[i] - targets[i];
        sum_e    += e;
        sum_ae   += std::abs(e);
        sum_se   += e * e;
        sum_y    += targets[i];
        sum_y2   += targets[i] * targets[i];
    }

    float rmse = static_cast<float>(std::sqrt(sum_se / N));
    float mae  = static_cast<float>(sum_ae / N);

    double mean_y  = sum_y / N;
    double ss_tot  = sum_y2 - N * mean_y * mean_y;
    float  r2      = (ss_tot > 1e-12) ? static_cast<float>(1.0 - sum_se / ss_tot) : 0.f;

    EvalMetrics m;
    m.rmse = rmse;
    m.mae  = mae;
    m.r2   = r2;
    return m;
}

// ---------------------------------------------------------------------------
// Multiclass
// ---------------------------------------------------------------------------

EvalMetrics compute_multiclass_metrics(
    const std::vector<float>& logits,
    const std::vector<float>& targets,
    std::size_t K)
{
    std::size_t N = targets.size();
    if (N == 0 || K == 0) return {};

    // Argmax predictions
    std::vector<std::size_t> preds(N);
    for (std::size_t i = 0; i < N; ++i) {
        float best = logits[i * K];
        preds[i] = 0;
        for (std::size_t k = 1; k < K; ++k) {
            if (logits[i * K + k] > best) {
                best = logits[i * K + k];
                preds[i] = k;
            }
        }
    }

    // Per-class TP, FP, FN for macro-F1
    std::vector<float> tp(K, 0.f), fp(K, 0.f), fn_vec(K, 0.f);
    float correct = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t gt = static_cast<std::size_t>(targets[i]);
        if (preds[i] == gt) { correct += 1.f; tp[gt] += 1.f; }
        else                { fp[preds[i]] += 1.f; fn_vec[gt] += 1.f; }
    }

    float macro_f1 = 0.f;
    for (std::size_t k = 0; k < K; ++k) {
        float denom = 2*tp[k] + fp[k] + fn_vec[k];
        macro_f1 += (denom > 0) ? 2*tp[k] / denom : 0.f;
    }
    macro_f1 /= static_cast<float>(K);

    EvalMetrics m;
    m.accuracy = correct / static_cast<float>(N);
    m.f1       = macro_f1;
    return m;
}

} // namespace relml