#pragma once

#include "relml/Database.h"
#include "relml/encoding/HeteroEncoder.h"
#include "relml/gnn/HeteroGraphSAGE.h"
#include "relml/gnn/MLPHead.h"
#include "relml/graph/HeteroGraph.h"
#include "relml/training/Adam.h"
#include "relml/training/Metrics.h"
#include "relml/training/TaskSpec.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace relml {

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

struct BCELoss {
    float pos_weight = 1.f;
    std::pair<float, std::vector<float>> forward_backward(
        const std::vector<float>& logits,
        const std::vector<float>& targets) const;
};

struct MSELoss {
    std::pair<float, std::vector<float>> forward_backward(
        const std::vector<float>& preds,
        const std::vector<float>& targets) const;
};

struct CrossEntropyLoss {
    // Per-class weights. If empty, all classes are weighted equally.
    // Set automatically by Trainer::fit() from training label frequencies
    // (inverse-frequency weighting, normalized so weights sum to K).
    // Example for 3-class football (43% home / 23% draw / 33% away):
    //   w[0] ≈ 0.78  w[1] ≈ 1.45  w[2] ≈ 1.01
    // This forces the model to pay more attention to the minority draw class.
    std::vector<float> class_weights;

    std::pair<float, std::vector<float>> forward_backward(
        const std::vector<float>& logits,
        const std::vector<float>& targets,
        std::size_t K) const;
};

// ---------------------------------------------------------------------------
// TrainConfig
// ---------------------------------------------------------------------------

struct TrainConfig {
    std::size_t channels   = 128;
    std::size_t gnn_layers = 2;
    std::size_t hidden     = 64;
    float       dropout    = 0.3f;
    float       lr         = 3e-4f;
    float       pos_weight = 1.f;
    std::size_t epochs     = 20;
    std::size_t batch_size = 0;   // 0 = full batch

    TaskSpec    task;

    // Backward-compat alias — maps to task.target_table
    std::string target_node;
};

// ---------------------------------------------------------------------------
// CSV-based task loader (legacy path used by test_training.cpp)
// ---------------------------------------------------------------------------

TaskSplit load_task(const std::string& csv_path, const Database& db,
                    const std::string& target_node);

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

class Trainer {
public:
    Trainer(const TrainConfig& cfg, const Database& db, const HeteroGraph& graph);

    // Train and return best validation metrics.
    EvalMetrics fit(const TaskSplit& task, const Database& db, const HeteroGraph& graph);

    // Score every row in the target table. Returns predictions in original scale.
    std::vector<float> predict_all(const Database& db, const HeteroGraph& graph);

    // Synthesize a prediction for a hypothetical entity combination.
    float synthesize_prediction(
        const std::unordered_map<std::string, std::string>& entity_refs,
        const Database&    db,
        const HeteroGraph& graph);

    EvalMetrics evaluate(const std::vector<TaskSample>& samples,
                         const Database& db, const HeteroGraph& graph);

    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

    const TrainConfig& config() const { return cfg_; }

private:
    HeteroEncoder   encoder;
    HeteroGraphSAGE gnn;
    MLPHead         head;
    BCELoss         bce_loss_;
    MSELoss         mse_loss_;
    CrossEntropyLoss ce_loss_;
    Adam            optimizer;
    TrainConfig     cfg_;

    std::vector<Parameter*> all_params_;
    std::size_t             num_encoder_params_ = 0;

    std::vector<float> gather(const NodeFeatures& nf,
                              const std::vector<TaskSample>& samples) const;
    std::vector<float> labels(const std::vector<TaskSample>& samples) const;

    std::pair<float, std::vector<float>> compute_loss(
        const std::vector<float>& logits,
        const std::vector<float>& targets) const;

    float forward_pass_batch(
        const std::unordered_map<std::string, NodeFeatures>& h_dict,
        const std::vector<TaskSample>& batch,
        std::unordered_map<std::string, std::vector<float>>& d_h_full_accum,
        bool train);

    float apply_output_transform(float raw_logit) const;
};

} // namespace relml