#pragma once

#include "relml/encoding/HeteroEncoder.h"
#include "relml/graph/HeteroGraph.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace relml {

struct SAGELayer {
    std::unordered_map<std::string, Linear>    W_self;
    std::unordered_map<std::string, Linear>    W_neigh;
    std::unordered_map<std::string, Parameter> bias;

    std::size_t channels = 0;

    SAGELayer() = default;
    explicit SAGELayer(std::size_t ch) : channels(ch) {}

    void init(const std::vector<std::string>& node_types,
              const std::vector<EdgeType>&    edge_types);
    void reset_parameters();

    // Non-const: caches forward state for backward
    std::unordered_map<std::string, NodeFeatures> forward(
        const std::unordered_map<std::string, NodeFeatures>& x_dict,
        const HeteroGraph& graph,
        bool apply_relu);

    // Uses cached state from last forward call
    std::unordered_map<std::string, std::vector<float>> backward(
        const std::unordered_map<std::string, std::vector<float>>& grad_h_dict,
        const HeteroGraph& graph);

    std::vector<Parameter*> parameters();

private:
    // Forward cache
    std::unordered_map<std::string, NodeFeatures>       x_cache_;
    std::unordered_map<std::string, std::vector<float>> agg_cache_;   // edge_name -> agg
    std::unordered_map<std::string, std::vector<float>> pre_relu_;    // node_type -> pre-act
    bool relu_applied_ = true;

    void mean_aggregate(
        const NodeFeatures&   src,
        const EdgeIndex&      ei,
        std::size_t           num_dst,
        std::vector<float>&   out_agg,
        std::vector<int32_t>& out_degree) const;

    std::vector<float> mean_aggregate_backward(
        const std::vector<float>& grad_agg,
        const EdgeIndex&          ei,
        std::size_t               num_src,
        std::size_t               num_dst) const;
};

struct HeteroGraphSAGE {
    std::vector<SAGELayer> layers;
    std::size_t channels   = 0;
    std::size_t num_layers = 0;

    HeteroGraphSAGE() = default;
    HeteroGraphSAGE(std::size_t channels, std::size_t num_layers,
                    const std::vector<std::string>& node_types,
                    const std::vector<EdgeType>&    edge_types);

    void reset_parameters();

    std::unordered_map<std::string, NodeFeatures> forward(
        const std::unordered_map<std::string, NodeFeatures>& x_dict,
        const HeteroGraph& graph);

    std::unordered_map<std::string, std::vector<float>> backward(
        const std::unordered_map<std::string, std::vector<float>>& grad_h,
        const HeteroGraph& graph);

    std::vector<Parameter*> parameters();
};

} // namespace relml