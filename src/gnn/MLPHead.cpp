#include "relml/gnn/MLPHead.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace relml {

MLPHead::MLPHead(std::size_t in_dim, std::size_t hidden,
                 float dropout, std::size_t out_features)
    : layer1(in_dim, hidden),
      layer2(hidden, out_features),
      dropout_p(dropout)
{}

void MLPHead::reset_parameters() {
    layer1.reset_parameters();
    layer2.reset_parameters();
}

void MLPHead::zero_grad() {
    layer1.zero_grad();
    layer2.zero_grad();
}

std::vector<float> MLPHead::forward(const std::vector<float>& x, std::size_t num_rows) {
    cache_.num_rows = num_rows;

    cache_.h1 = layer1.forward(x, num_rows);

    cache_.h1_relu = cache_.h1;
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < cache_.h1_relu.size(); ++i)
        cache_.h1_relu[i] = std::max(0.f, cache_.h1[i]);

    std::size_t H = layer1.out_dim;
    cache_.h1_dropped = cache_.h1_relu;
    cache_.mask.assign(num_rows * H, 1.f);

    if (training_ && dropout_p > 0.f) {
        static std::mt19937 rng(42);
        std::bernoulli_distribution drop(dropout_p);
        // scale surviving units so expected sum is preserved
        float scale = 1.f / (1.f - dropout_p);
        for (std::size_t i = 0; i < num_rows * H; ++i) {
            if (drop(rng)) {
                cache_.mask[i]       = 0.f;
                cache_.h1_dropped[i] = 0.f;
            } else {
                // mask stays 1, value is already scaled
                cache_.h1_dropped[i] *= scale;
            }
        }
    }

    return layer2.forward(cache_.h1_dropped, num_rows);
}

std::vector<float> MLPHead::backward(const std::vector<float>& d_logits) {
    std::size_t N = cache_.num_rows;
    std::size_t H = layer1.out_dim;

    // gradient w.r.t. h1_dropped
    std::vector<float> d_dropped = layer2.backward(d_logits);

    // gradient w.r.t. h1_relu
    // In forward: surviving units were multiplied by scale = 1/(1-p).
    // The backward pass must apply the same linear operation: multiply by
    // scale where mask=1, zero where mask=0.
    // mask[i] is already 0 for dropped units, so d_relu[i] = d_dropped[i] * scale * 1
    // and 0 for dropped units = d_dropped[i] * 0.
    // This is simply: d_relu[i] = d_dropped[i] * (mask[i] > 0 ? scale : 0)
    // which equals d_dropped[i] * mask[i] * scale — but mask is already 0 or 1
    // so we write it as: d_dropped[i] * (mask[i] * scale).
    //
    // NOTE: the previous implementation incorrectly applied scale a second time
    // as a separate multiplication after the mask, resulting in scale^2 instead
    // of scale for surviving units. Fixed below.
    std::vector<float> d_relu(N * H);
    if (training_ && dropout_p > 0.f) {
        float scale = 1.f / (1.f - dropout_p);
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N * H; ++i)
            d_relu[i] = d_dropped[i] * cache_.mask[i] * scale;
    } else {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N * H; ++i)
            d_relu[i] = d_dropped[i];
    }

    // gradient w.r.t. h1 (pre-ReLU): zero out where pre-ReLU value was <= 0
    std::vector<float> d_h1(N * H);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N * H; ++i)
        d_h1[i] = cache_.h1[i] > 0.f ? d_relu[i] : 0.f;

    return layer1.backward(d_h1);
}

std::vector<Parameter*> MLPHead::parameters() {
    auto p1 = layer1.parameters();
    auto p2 = layer2.parameters();
    p1.insert(p1.end(), p2.begin(), p2.end());
    return p1;
}

} // namespace relml