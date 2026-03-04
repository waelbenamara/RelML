#pragma once

#include "relml/Database.h"
#include "relml/graph/HeteroGraph.h"
#include "relml/training/TaskParser.h"
#include "relml/training/TaskSpec.h"
#include "relml/training/Trainer.h"
#include "relml/agent/Agent.h"
#include "relml/agent/ModelRegistry.h"

#include <memory>
#include <optional>
#include <string>

namespace relml {

struct SystemConfig {
    std::size_t channels      = 128;
    std::size_t gnn_layers    = 2;
    std::size_t hidden        = 64;
    float       dropout       = 0.3f;
    float       lr            = 3e-4f;
    std::size_t epochs        = 20;
    std::size_t batch_size    = 0;
    std::string registry_root;   // empty = use $RELML_HOME or ~/.relml
};

class RelMLSystem {
public:
    RelMLSystem(
        const Database&     db,
        const HeteroGraph&  graph,
        const SystemConfig& sys_cfg = SystemConfig{});

    std::string query(const std::string& nl);

    const ModelRegistry& registry() const { return registry_; }

private:
    const Database&    db_;
    const HeteroGraph& graph_;
    SystemConfig       sys_cfg_;
    ModelRegistry      registry_;
    TaskParser         parser_;
    mutable Agent      responder_;

    std::unique_ptr<Trainer>    active_trainer_;
    std::optional<TaskSpec>     active_task_;

    enum class Intent { TrainAndInfer, LoadAndInfer, InferOnly };
    Intent resolve_intent(const TaskSpec& spec) const;

    std::unique_ptr<Trainer> make_trainer(const TaskSpec& spec) const;

    std::string format_synthesis_response(
        const std::string& nl,
        const TaskSpec&    spec,
        float              prediction) const;

    std::string format_row_response(
        const std::string&               nl,
        const TaskSpec&                  spec,
        const TaskSpec::InferenceResult& result) const;
};

} // namespace relml