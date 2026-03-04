#pragma once

#include "relml/Parameter.h"
#include "relml/encoding/NumericalEncoder.h"
#include "relml/encoding/CategoricalEncoder.h"
#include "relml/encoding/TimestampEncoder.h"
#include "relml/Database.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace relml {

struct ColumnEncoder {
    ColumnType type;
    std::unique_ptr<NumericalEncoder>   numerical;
    std::unique_ptr<CategoricalEncoder> categorical;
    std::unique_ptr<TimestampEncoder>   timestamp;

    std::size_t        output_dim() const;
    std::vector<float> transform(const Column& col) const;
};

// Linear projection: y = x @ W^T + b
// Caches its input on forward for use in backward.
struct Linear {
    std::size_t in_dim  = 0;
    std::size_t out_dim = 0;
    Parameter   W;   // out_dim x in_dim
    Parameter   b;   // out_dim

    Linear() = default;
    Linear(std::size_t in, std::size_t out);

    void reset_parameters();
    void zero_grad();

    // Non-const: caches x for backward
    std::vector<float> forward(const std::vector<float>& x, std::size_t num_rows);

    // Uses cached input. Accumulates into W.grad, b.grad. Returns dL/dx.
    std::vector<float> backward(const std::vector<float>& grad_out);

    std::vector<Parameter*> parameters() { return {&W, &b}; }

private:
    std::vector<float> x_cache_;
    std::size_t        cached_rows_ = 0;
};

struct NodeFeatures {
    std::string        node_type;
    std::size_t        num_nodes = 0;
    std::size_t        channels  = 0;
    std::vector<float> data;

    float operator()(std::size_t row, std::size_t col) const {
        return data[row * channels + col];
    }
};

class HeteroEncoder {
public:
    explicit HeteroEncoder(std::size_t channels) : channels_(channels) {}

    void fit(const Database& db);

    // Non-const: Linear::forward caches inputs
    std::unordered_map<std::string, NodeFeatures> transform(const Database& db);

    // Backward through projection layers only. Raw encoders are fixed.
    void backward(const std::unordered_map<std::string, std::vector<float>>& grad_dict);

    std::vector<Parameter*> parameters();

    std::size_t channels() const { return channels_; }

private:
    std::size_t channels_;
    std::unordered_map<std::string,
        std::unordered_map<std::string, ColumnEncoder>> encoders_;
    std::unordered_map<std::string, Linear> projections_;
    bool fitted_ = false;

    void        fit_table(const Table& table);
    std::size_t raw_dim(const std::string& node_type) const;
};

} // namespace relml