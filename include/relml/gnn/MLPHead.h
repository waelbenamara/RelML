#pragma once

#include "relml/encoding/HeteroEncoder.h"  // for Linear, Parameter
#include <vector>

namespace relml {

class MLPHead {
public:
    // out_features: 1 for binary classification or regression,
    //               K for K-class classification.
    explicit MLPHead(std::size_t in_dim,
                     std::size_t hidden,
                     float       dropout     = 0.f,
                     std::size_t out_features = 1);

    void reset_parameters();
    void zero_grad();

    // Returns N * out_features raw logits (no activation applied).
    std::vector<float> forward(const std::vector<float>& x, std::size_t num_rows);

    // Returns gradient w.r.t. input (N * in_dim).
    std::vector<float> backward(const std::vector<float>& d_logits);

    std::vector<Parameter*> parameters();

    void train() { training_ = true;  }
    void eval()  { training_ = false; }

    std::size_t out_features() const { return layer2.out_dim; }

private:
    Linear layer1;
    Linear layer2;
    float  dropout_p;
    bool   training_ = true;

    struct Cache {
        std::size_t        num_rows = 0;
        std::vector<float> h1;
        std::vector<float> h1_relu;
        std::vector<float> h1_dropped;
        std::vector<float> mask;
    } cache_;
};

} // namespace relml