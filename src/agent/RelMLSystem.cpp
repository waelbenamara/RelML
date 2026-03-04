#include "relml/agent/RelMLSystem.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace relml {

static const char* RESPONDER_PROMPT = R"(
You are an assistant that translates machine learning inference results into
clear, concise natural language answers. The user asked a question about a
relational database. You are given the original question and the numerical
result(s) from a trained model. Answer the question directly and precisely,
citing the numbers. Keep the answer to 2-3 sentences.
)";

static AgentConfig responder_config() {
    AgentConfig cfg;
    cfg.system_prompt = RESPONDER_PROMPT;
    return cfg;
}

RelMLSystem::RelMLSystem(
    const Database&     db,
    const HeteroGraph&  graph,
    const SystemConfig& sys_cfg)
    : db_(db),
      graph_(graph),
      sys_cfg_(sys_cfg),
      registry_(sys_cfg.registry_root),
      parser_(),
      responder_(responder_config())
{}

RelMLSystem::Intent RelMLSystem::resolve_intent(const TaskSpec& spec) const {
    std::string fp = spec.fingerprint();
    if (active_task_ && active_task_->fingerprint() == fp)
        return Intent::InferOnly;
    if (registry_.has(fp))
        return Intent::LoadAndInfer;
    return Intent::TrainAndInfer;
}

std::unique_ptr<Trainer> RelMLSystem::make_trainer(const TaskSpec& spec) const {
    TrainConfig cfg;
    cfg.channels   = sys_cfg_.channels;
    cfg.gnn_layers = sys_cfg_.gnn_layers;
    cfg.hidden     = sys_cfg_.hidden;
    cfg.dropout    = sys_cfg_.dropout;
    cfg.lr         = sys_cfg_.lr;
    cfg.epochs     = sys_cfg_.epochs;
    cfg.batch_size = sys_cfg_.batch_size;
    cfg.task       = spec;
    return std::make_unique<Trainer>(cfg, db_, graph_);
}

// ---------------------------------------------------------------------------
// Response formatters
// ---------------------------------------------------------------------------

std::string RelMLSystem::format_synthesis_response(
    const std::string& nl,
    const TaskSpec&    spec,
    float              prediction) const
{
    std::ostringstream ctx;
    ctx << "Question: " << nl << "\n\n"
        << "Task: predict '" << spec.target_column
        << "' in table '"    << spec.target_table << "'\n"
        << "Inference method: entity synthesis (hypothetical combination)\n"
        << "Entities used: ";
    for (const auto& [k, v] : spec.entity_refs)
        ctx << k << "=" << v << " ";
    ctx << "\n";

    switch (spec.task_type) {
        case TaskSpec::TaskType::BinaryClassification:
            ctx << "Result: predicted probability = "
                << std::fixed << std::setprecision(4) << prediction << "\n";
            break;
        case TaskSpec::TaskType::Regression:
            ctx << "Result: predicted value = "
                << std::fixed << std::setprecision(4) << prediction << "\n";
            break;
        case TaskSpec::TaskType::MulticlassClassification:
            ctx << "Result: predicted class = "
                << static_cast<int>(prediction) << "\n";
            break;
    }

    responder_.reset();
    return responder_.send(ctx.str()).text;
}

std::string RelMLSystem::format_row_response(
    const std::string&               nl,
    const TaskSpec&                  spec,
    const TaskSpec::InferenceResult& result) const
{
    std::ostringstream ctx;
    ctx << "Question: " << nl << "\n\n"
        << "Task: predict '" << spec.target_column
        << "' in table '"    << spec.target_table << "'\n";

    if (result.predictions.empty()) {
        ctx << "Result: no matching rows after filtering.\n";
    } else if (result.aggregate.has_value()) {
        float agg = *result.aggregate;
        switch (spec.inference_agg) {
            case TaskSpec::AggType::Fraction:
                ctx << std::fixed << std::setprecision(2)
                    << "Result: " << (agg * 100.f) << "% of "
                    << result.row_indices.size() << " matching rows predicted positive.\n";
                break;
            case TaskSpec::AggType::Mean:
                ctx << "Result: mean predicted value = "
                    << std::fixed << std::setprecision(4) << agg << "\n";
                break;
            case TaskSpec::AggType::Count:
                ctx << "Result: " << static_cast<int>(agg)
                    << " rows predicted positive (out of "
                    << result.row_indices.size() << " matching).\n";
                break;
            default: break;
        }
    } else {
        std::size_t cap = std::min(result.predictions.size(), std::size_t{10});
        ctx << "Predictions (" << result.predictions.size() << " rows):\n";
        for (std::size_t i = 0; i < cap; ++i)
            ctx << "  row " << result.row_indices[i] << ": "
                << std::fixed << std::setprecision(4) << result.predictions[i] << "\n";
        if (result.predictions.size() > cap)
            ctx << "  ... and " << (result.predictions.size() - cap) << " more.\n";
    }

    responder_.reset();
    return responder_.send(ctx.str()).text;
}

// ---------------------------------------------------------------------------
// Task summary for logging
// ---------------------------------------------------------------------------

static std::string task_type_str(TaskSpec::TaskType t) {
    switch (t) {
        case TaskSpec::TaskType::BinaryClassification: return "binary_classification";
        case TaskSpec::TaskType::Regression:            return "regression";
        case TaskSpec::TaskType::MulticlassClassification: return "multiclass_classification";
    }
    return "?";
}

static std::string label_transform_str(const LabelTransform& lt) {
    switch (lt.kind) {
        case LabelTransform::Kind::Threshold:
            return "threshold " + std::to_string(lt.threshold) + (lt.inclusive ? " (>=)" : " (>)");
        case LabelTransform::Kind::Identity:   return "identity";
        case LabelTransform::Kind::Normalize:  return "normalize";
        case LabelTransform::Kind::Buckets:    return "buckets";
    }
    return "?";
}

static void log_task_spec(const TaskSpec& spec, const std::string& nl) {
    std::cout << "  [RelMLSystem] Query: \"" << nl << "\"\n";
    std::cout << "  [RelMLSystem] Task:\n";
    std::cout << "    target        : " << spec.target_table << "." << spec.target_column << "\n";
    std::cout << "    task_type     : " << task_type_str(spec.task_type) << "\n";
    std::cout << "    label         : " << label_transform_str(spec.label_transform) << "\n";
    std::cout << "    split         : "
              << (spec.split_strategy == TaskSpec::SplitStrategy::Temporal ? "temporal" : "random")
              << (spec.split_time_col ? std::string(" (") + *spec.split_time_col + ")" : "") << "\n";
    std::cout << "    inference_mode: "
              << (spec.inference_mode == TaskSpec::InferenceMode::EntitySynthesis ? "entity_synthesis" : "row_based")
              << "\n";
    if (spec.inference_mode == TaskSpec::InferenceMode::EntitySynthesis && !spec.entity_refs.empty()) {
        std::cout << "    entity_refs   : ";
        for (const auto& [k, v] : spec.entity_refs)
            std::cout << k << "=" << v << " ";
        std::cout << "\n";
    }
    if (!spec.inference_filters.empty()) {
        std::cout << "    filters       : ";
        for (std::size_t i = 0; i < spec.inference_filters.size(); ++i) {
            const auto& f = spec.inference_filters[i];
            if (i) std::cout << ", ";
            std::cout << f.column << " " << f.op << " " << f.value;
        }
        std::cout << "\n";
    }
    std::cout << "    inference_agg : ";
    switch (spec.inference_agg) {
        case TaskSpec::AggType::None:     std::cout << "none\n"; break;
        case TaskSpec::AggType::Mean:    std::cout << "mean\n"; break;
        case TaskSpec::AggType::Fraction: std::cout << "fraction\n"; break;
        case TaskSpec::AggType::Count:   std::cout << "count\n"; break;
    }
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

std::string RelMLSystem::query(const std::string& nl) {
    std::cout << "  [RelMLSystem] Parsing query...\n";
    TaskSpec spec = parser_.parse(nl, db_);
    std::string fp = spec.fingerprint();
    std::cout << "  [RelMLSystem] Fingerprint: " << fp
              << "  mode: " << (spec.inference_mode == TaskSpec::InferenceMode::EntitySynthesis
                                ? "entity_synthesis" : "row_based") << "\n";

    log_task_spec(spec, nl);

    Intent intent = resolve_intent(spec);

    if (intent == Intent::TrainAndInfer) {
        std::cout << "  [RelMLSystem] Intent: train_and_infer\n";
        std::cout << "  [RelMLSystem] Training: " << task_type_str(spec.task_type)
                  << " on " << spec.target_table << "." << spec.target_column
                  << " (" << label_transform_str(spec.label_transform) << "), "
                  << (spec.split_strategy == TaskSpec::SplitStrategy::Temporal ? "temporal" : "random")
                  << " split.\n";
        TaskSplit split = spec.build_split(db_);
        active_trainer_ = make_trainer(spec);
        active_trainer_->fit(split, db_, graph_);
        active_trainer_->save_weights(registry_.weight_path(fp));
        active_task_ = spec;
    } else if (intent == Intent::LoadAndInfer) {
        std::cout << "  [RelMLSystem] Intent: load_and_infer\n";
        active_trainer_ = make_trainer(spec);
        active_trainer_->load_weights(registry_.weight_path(fp));
        active_task_ = spec;
    } else {
        std::cout << "  [RelMLSystem] Intent: infer_only\n";
        // Update inference fields on active_task_ so the new query's
        // entity_refs / filters are used even though the model is cached.
        if (active_task_) {
            active_task_->inference_mode    = spec.inference_mode;
            active_task_->inference_filters = spec.inference_filters;
            active_task_->inference_agg     = spec.inference_agg;
            active_task_->entity_refs       = spec.entity_refs;
        }
    }

    const TaskSpec& active_spec = active_task_ ? *active_task_ : spec;

    if (active_spec.inference_mode == TaskSpec::InferenceMode::EntitySynthesis) {
        float pred = active_trainer_->synthesize_prediction(
            active_spec.entity_refs, db_, graph_);
        return format_synthesis_response(nl, active_spec, pred);
    } else {
        std::vector<float> all_preds = active_trainer_->predict_all(db_, graph_);
        TaskSpec::InferenceResult result = active_spec.apply_inference(db_, all_preds);
        return format_row_response(nl, active_spec, result);
    }
}

} // namespace relml