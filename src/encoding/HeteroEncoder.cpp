#include "relml/encoding/HeteroEncoder.h"
#include <algorithm>
#include <cmath>
#include <random>

#ifdef RELML_USE_BLAS
// cblas.h may live under openblas/ depending on the distro
#if __has_include(<cblas.h>)
#  include <cblas.h>
#elif __has_include(<openblas/cblas.h>)
#  include <openblas/cblas.h>
#else
#  undef RELML_USE_BLAS
#  warning "cblas.h not found — disabling BLAS acceleration"
#endif
#endif

namespace relml {

// ---------------------------------------------------------------------------
// ColumnEncoder
// ---------------------------------------------------------------------------

std::size_t ColumnEncoder::output_dim() const {
    switch (type) {
        case ColumnType::NUMERICAL:   return 1;
        case ColumnType::CATEGORICAL: return categorical->vocab_size();
        case ColumnType::TIMESTAMP:   return TimestampEncoder::OUTPUT_DIM;
        case ColumnType::TEXT:        return 0;
    }
    return 0;
}

std::vector<float> ColumnEncoder::transform(const Column& col) const {
    switch (type) {
        case ColumnType::NUMERICAL:   return numerical->transform(col);
        case ColumnType::CATEGORICAL: return categorical->transform(col);
        case ColumnType::TIMESTAMP:   return timestamp->transform(col);
        case ColumnType::TEXT:        return {};
    }
    return {};
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

Linear::Linear(std::size_t in, std::size_t out)
    : in_dim(in), out_dim(out),
      W(out * in),
      b(out)
{
    reset_parameters();
}

void Linear::reset_parameters() {
    float bound = std::sqrt(6.f / (in_dim + out_dim));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (auto& w : W.data) w = dist(rng);
    std::fill(b.data.begin(), b.data.end(), 0.f);
    W.zero_grad();
    b.zero_grad();
}

void Linear::zero_grad() {
    W.zero_grad();
    b.zero_grad();
}

std::vector<float> Linear::forward(const std::vector<float>& x, std::size_t N) {
    x_cache_     = x;
    cached_rows_ = N;

    std::vector<float> out(N * out_dim, 0.f);

#ifdef RELML_USE_BLAS
    // out = x * W^T   (N×in_dim) * (in_dim×out_dim)^T → N×out_dim
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, out_dim, in_dim,
                1.f, x.data(), in_dim,
                W.data.data(), in_dim,
                0.f, out.data(), out_dim);
    // add bias row-wise
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t o = 0; o < out_dim; ++o)
            out[i * out_dim + o] += b.data[o];
#else
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t o = 0; o < out_dim; ++o) {
            float acc = b.data[o];
            for (std::size_t k = 0; k < in_dim; ++k)
                acc += W.data[o * in_dim + k] * x[i * in_dim + k];
            out[i * out_dim + o] = acc;
        }
#endif
    return out;
}

std::vector<float> Linear::backward(const std::vector<float>& grad_out) {
    std::size_t N = cached_rows_;

#ifdef RELML_USE_BLAS
    // W.grad += grad_out^T * x   (out_dim×N) * (N×in_dim) → out_dim×in_dim
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                out_dim, in_dim, N,
                1.f, grad_out.data(), out_dim,
                x_cache_.data(), in_dim,
                1.f, W.grad.data(), in_dim);

    // b.grad += sum over rows of grad_out
#pragma omp parallel for schedule(static)
    for (std::size_t o = 0; o < out_dim; ++o) {
        float acc = 0.f;
        for (std::size_t i = 0; i < N; ++i)
            acc += grad_out[i * out_dim + o];
        b.grad[o] += acc;
    }

    // grad_x = grad_out * W   (N×out_dim) * (out_dim×in_dim) → N×in_dim
    std::vector<float> grad_x(N * in_dim, 0.f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, in_dim, out_dim,
                1.f, grad_out.data(), out_dim,
                W.data.data(), in_dim,
                0.f, grad_x.data(), in_dim);
#else
#pragma omp parallel for schedule(static) collapse(2)
    for (std::size_t o = 0; o < out_dim; ++o)
        for (std::size_t k = 0; k < in_dim; ++k) {
            float acc = 0.f;
            for (std::size_t i = 0; i < N; ++i)
                acc += grad_out[i * out_dim + o] * x_cache_[i * in_dim + k];
            W.grad[o * in_dim + k] += acc;
        }

#pragma omp parallel for schedule(static)
    for (std::size_t o = 0; o < out_dim; ++o) {
        float acc = 0.f;
        for (std::size_t i = 0; i < N; ++i)
            acc += grad_out[i * out_dim + o];
        b.grad[o] += acc;
    }

    std::vector<float> grad_x(N * in_dim, 0.f);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t k = 0; k < in_dim; ++k) {
            float acc = 0.f;
            for (std::size_t o = 0; o < out_dim; ++o)
                acc += grad_out[i * out_dim + o] * W.data[o * in_dim + k];
            grad_x[i * in_dim + k] = acc;
        }
#endif
    return grad_x;
}

// ---------------------------------------------------------------------------
// HeteroEncoder
// ---------------------------------------------------------------------------

void HeteroEncoder::fit_table(const Table& table) {
    auto& col_encoders = encoders_[table.name];
    for (const auto& col : table.columns) {
        if (table.pkey_col && col.name == *table.pkey_col) continue;
        if (col.type == ColumnType::TEXT) continue;

        ColumnEncoder ce;
        ce.type = col.type;
        switch (col.type) {
            case ColumnType::NUMERICAL:
                ce.numerical = std::make_unique<NumericalEncoder>();
                ce.numerical->fit(col); break;
            case ColumnType::CATEGORICAL:
                ce.categorical = std::make_unique<CategoricalEncoder>();
                ce.categorical->fit(col); break;
            case ColumnType::TIMESTAMP:
                ce.timestamp = std::make_unique<TimestampEncoder>();
                ce.timestamp->fit(col); break;
            default: break;
        }
        col_encoders.emplace(col.name, std::move(ce));
    }
}

std::size_t HeteroEncoder::raw_dim(const std::string& node_type) const {
    auto it = encoders_.find(node_type);
    if (it == encoders_.end()) return 0;
    std::size_t dim = 0;
    for (const auto& [_, ce] : it->second) dim += ce.output_dim();
    return dim;
}

void HeteroEncoder::fit(const Database& db) {
    for (const auto& [name, table] : db.tables)
        fit_table(table);

    for (const auto& [name, _] : db.tables) {
        std::size_t in = raw_dim(name);
        if (in == 0) in = 1;
        projections_.emplace(name, Linear(in, channels_));
    }
    fitted_ = true;
}

std::unordered_map<std::string, NodeFeatures>
HeteroEncoder::transform(const Database& db) {
    std::unordered_map<std::string, NodeFeatures> result;

    for (const auto& [name, table] : db.tables) {
        std::size_t N   = table.num_rows();
        std::size_t raw = raw_dim(name);
        if (raw == 0) raw = 1;

        std::vector<float> raw_mat(N * raw, 0.f);

        auto enc_it = encoders_.find(name);
        if (enc_it != encoders_.end()) {
            std::size_t col_offset = 0;
            for (const auto& col : table.columns) {
                auto ce_it = enc_it->second.find(col.name);
                if (ce_it == enc_it->second.end()) continue;
                const ColumnEncoder& ce  = ce_it->second;
                std::size_t          dim = ce.output_dim();
                if (dim == 0) continue;
                std::vector<float> col_data = ce.transform(col);

                // Each row writes to a unique slice — safe to parallelize
#pragma omp parallel for schedule(static)
                for (std::size_t row = 0; row < N; ++row)
                    for (std::size_t d = 0; d < dim; ++d)
                        raw_mat[row * raw + col_offset + d] = col_data[row * dim + d];

                col_offset += dim;
            }
        }

        std::vector<float> projected = projections_.at(name).forward(raw_mat, N);

        NodeFeatures nf;
        nf.node_type = name;
        nf.num_nodes = N;
        nf.channels  = channels_;
        nf.data      = std::move(projected);
        result.emplace(name, std::move(nf));
    }
    return result;
}

void HeteroEncoder::backward(
    const std::unordered_map<std::string, std::vector<float>>& grad_dict)
{
    for (auto& [name, proj] : projections_) {
        auto it = grad_dict.find(name);
        if (it == grad_dict.end()) continue;
        proj.backward(it->second);
    }
}

std::vector<Parameter*> HeteroEncoder::parameters() {
    std::vector<Parameter*> params;
    for (auto& [_, proj] : projections_) {
        auto p = proj.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

} // namespace relml