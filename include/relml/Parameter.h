#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace relml {

struct Parameter {
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<float> m1;  // Adam first moment
    std::vector<float> m2;  // Adam second moment

    Parameter() = default;
    explicit Parameter(std::size_t n) : data(n), grad(n), m1(n), m2(n) {}
    Parameter(std::size_t n, float fill) : data(n, fill), grad(n), m1(n), m2(n) {}

    std::size_t size() const { return data.size(); }

    // Only zeroes the gradient — moments persist across steps
    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.f);
    }
};

} // namespace relml