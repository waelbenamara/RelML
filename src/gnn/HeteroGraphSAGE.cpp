#include "relml/gnn/HeteroGraphSAGE.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace relml {

// ---------------------------------------------------------------------------
// SAGELayer
// ---------------------------------------------------------------------------

void SAGELayer::init(const std::vector<std::string>& node_types,
                     const std::vector<EdgeType>&    edge_types) {
    for (const auto& nt : node_types) {
        W_self.emplace(nt, Linear(channels, channels));
        bias[nt] = Parameter(channels, 0.f);
    }
    for (const auto& et : edge_types)
        W_neigh.emplace(et.name(), Linear(channels, channels));
}

void SAGELayer::reset_parameters() {
    for (auto& [_, w] : W_self)  w.reset_parameters();
    for (auto& [_, w] : W_neigh) w.reset_parameters();
    for (auto& [_, b] : bias)    std::fill(b.data.begin(), b.data.end(), 0.f);
}

void SAGELayer::mean_aggregate(
    const NodeFeatures&   src,
    const EdgeIndex&      ei,
    std::size_t           num_dst,
    std::vector<float>&   out_agg,
    std::vector<int32_t>& out_degree) const
{
    out_agg.assign(num_dst * channels, 0.f);
    out_degree.assign(num_dst, 0);

    // Sequential: multiple edges scatter to same dst, cannot parallelize
    for (std::size_t e = 0; e < ei.num_edges(); ++e) {
        int64_t s = ei.src[e], d = ei.dst[e];
        ++out_degree[d];
        const float* sr = src.data.data() + s * channels;
        float*       dr = out_agg.data()  + d * channels;
        for (std::size_t c = 0; c < channels; ++c) dr[c] += sr[c];
    }

#pragma omp parallel for schedule(static)
    for (std::size_t d = 0; d < num_dst; ++d) {
        if (!out_degree[d]) continue;
        float inv = 1.f / out_degree[d];
        float* r  = out_agg.data() + d * channels;
        for (std::size_t c = 0; c < channels; ++c) r[c] *= inv;
    }
}

std::vector<float> SAGELayer::mean_aggregate_backward(
    const std::vector<float>& grad_agg,
    const EdgeIndex&          ei,
    std::size_t               num_src,
    std::size_t               num_dst) const
{
    std::vector<int32_t> count(num_dst, 0);
    for (std::size_t e = 0; e < ei.num_edges(); ++e)
        ++count[ei.dst[e]];

    std::vector<float> grad_src(num_src * channels, 0.f);

    // Sequential: multiple edges scatter to same src node, cannot parallelize
    for (std::size_t e = 0; e < ei.num_edges(); ++e) {
        int64_t s = ei.src[e], d = ei.dst[e];
        if (!count[d]) continue;
        float inv          = 1.f / count[d];
        const float* gd    = grad_agg.data() + d * channels;
        float*       gs    = grad_src.data() + s * channels;
        for (std::size_t c = 0; c < channels; ++c)
            gs[c] += inv * gd[c];
    }
    return grad_src;
}

std::unordered_map<std::string, NodeFeatures> SAGELayer::forward(
    const std::unordered_map<std::string, NodeFeatures>& x_dict,
    const HeteroGraph& graph,
    bool apply_relu)
{
    relu_applied_ = apply_relu;
    x_cache_      = x_dict;
    agg_cache_.clear();
    pre_relu_.clear();

    std::unordered_map<std::string, std::vector<float>> neigh_sum;
    for (const auto& [nt, cnt] : graph.num_nodes)
        neigh_sum[nt].assign(cnt * channels, 0.f);

    for (const auto& et : graph.edge_types()) {
        auto w_it = W_neigh.find(et.name());
        auto x_it = x_dict.find(et.src);
        if (w_it == W_neigh.end() || x_it == x_dict.end()) continue;

        int64_t num_dst = graph.num_nodes.at(et.dst);

        std::vector<float>   agg;
        std::vector<int32_t> deg;
        mean_aggregate(x_it->second, graph.edge_index.at(et), num_dst, agg, deg);
        agg_cache_[et.name()] = agg;

        std::vector<float> transformed = w_it->second.forward(agg, num_dst);

        // Safe to parallelize: each edge type runs sequentially, so no two
        // threads write to the same neigh_sum vector simultaneously
        std::vector<float>& ns = neigh_sum.at(et.dst);
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < transformed.size(); ++i)
            ns[i] += transformed[i];
    }

    std::unordered_map<std::string, NodeFeatures> out;
    for (const auto& [nt, nf] : x_dict) {
        int64_t N = graph.num_nodes.at(nt);
        std::vector<float> self_out = W_self.at(nt).forward(nf.data, N);

        const std::vector<float>& ns = neigh_sum.at(nt);
        const std::vector<float>& b  = bias.at(nt).data;
        std::vector<float> pre(N * channels);

#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < (std::size_t)N; ++i)
            for (std::size_t c = 0; c < channels; ++c)
                pre[i*channels+c] = self_out[i*channels+c] + ns[i*channels+c] + b[c];

        pre_relu_[nt] = pre;

        std::vector<float> h(N * channels);
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < (std::size_t)N * channels; ++i)
            h[i] = apply_relu ? std::max(0.f, pre[i]) : pre[i];

        NodeFeatures nf_out;
        nf_out.node_type = nt;
        nf_out.num_nodes = N;
        nf_out.channels  = channels;
        nf_out.data      = std::move(h);
        out.emplace(nt, std::move(nf_out));
    }
    return out;
}

std::unordered_map<std::string, std::vector<float>> SAGELayer::backward(
    const std::unordered_map<std::string, std::vector<float>>& grad_h_dict,
    const HeteroGraph& graph)
{
    std::unordered_map<std::string, std::vector<float>> grad_pre;

    for (const auto& [nt, gh] : grad_h_dict) {
        const std::vector<float>& pre = pre_relu_.at(nt);
        std::size_t N = gh.size() / channels;
        std::vector<float> gp(N * channels);

#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N * channels; ++i)
            gp[i] = relu_applied_ ? (pre[i] > 0.f ? gh[i] : 0.f) : gh[i];

        grad_pre[nt] = std::move(gp);

        // Bias gradient — sequential over rows, then channels
        auto& bg = bias.at(nt).grad;
        const auto& gp_nt = grad_pre.at(nt);
        for (std::size_t o = 0; o < channels; ++o) {
            float acc = 0.f;
            for (std::size_t i = 0; i < N; ++i)
                acc += gp_nt[i*channels+o];
            bg[o] += acc;
        }
    }

    std::unordered_map<std::string, std::vector<float>> grad_x;
    for (const auto& [nt, nf] : x_cache_)
        grad_x[nt].assign(nf.num_nodes * channels, 0.f);

    // Self-transform backward
    for (auto& [nt, gp] : grad_pre) {
        std::vector<float> gx = W_self.at(nt).backward(gp);
        auto& dx = grad_x.at(nt);
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < gx.size(); ++i) dx[i] += gx[i];
    }

    // Neighbor path backward — edge types run sequentially so the
    // grad_x accumulation per edge type is safe to parallelize
    for (const auto& et : graph.edge_types()) {
        auto w_it  = W_neigh.find(et.name());
        auto gp_it = grad_pre.find(et.dst);
        auto ac_it = agg_cache_.find(et.name());
        if (w_it == W_neigh.end() || gp_it == grad_pre.end() || ac_it == agg_cache_.end()) continue;

        int64_t num_dst = graph.num_nodes.at(et.dst);
        int64_t num_src = graph.num_nodes.at(et.src);

        std::vector<float> grad_agg = w_it->second.backward(gp_it->second);
        std::vector<float> grad_src = mean_aggregate_backward(
            grad_agg, graph.edge_index.at(et), num_src, num_dst);

        auto gx_it = grad_x.find(et.src);
        if (gx_it == grad_x.end()) continue;
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < grad_src.size(); ++i)
            gx_it->second[i] += grad_src[i];
    }

    return grad_x;
}

std::vector<Parameter*> SAGELayer::parameters() {
    std::vector<Parameter*> params;
    for (auto& [_, w] : W_self) {
        auto p = w.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    for (auto& [_, w] : W_neigh) {
        auto p = w.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    for (auto& [_, b] : bias)
        params.push_back(&b);
    return params;
}

// ---------------------------------------------------------------------------
// HeteroGraphSAGE
// ---------------------------------------------------------------------------

HeteroGraphSAGE::HeteroGraphSAGE(
    std::size_t ch, std::size_t nl,
    const std::vector<std::string>& node_types,
    const std::vector<EdgeType>&    edge_types)
    : channels(ch), num_layers(nl)
{
    layers.resize(nl, SAGELayer(ch));
    for (auto& l : layers) l.init(node_types, edge_types);
}

void HeteroGraphSAGE::reset_parameters() {
    for (auto& l : layers) l.reset_parameters();
}

std::unordered_map<std::string, NodeFeatures> HeteroGraphSAGE::forward(
    const std::unordered_map<std::string, NodeFeatures>& x_dict,
    const HeteroGraph& graph)
{
    auto h = x_dict;
    for (std::size_t l = 0; l < layers.size(); ++l)
        h = layers[l].forward(h, graph, l < layers.size() - 1);
    return h;
}

std::unordered_map<std::string, std::vector<float>> HeteroGraphSAGE::backward(
    const std::unordered_map<std::string, std::vector<float>>& grad_h,
    const HeteroGraph& graph)
{
    auto g = grad_h;
    for (int l = (int)layers.size() - 1; l >= 0; --l)
        g = layers[l].backward(g, graph);
    return g;
}

std::vector<Parameter*> HeteroGraphSAGE::parameters() {
    std::vector<Parameter*> params;
    for (auto& l : layers) {
        auto p = l.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

} // namespace relml