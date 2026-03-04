#include "relml/encoding/NumericalEncoder.h"
#include <cmath>
#include <stdexcept>

namespace relml {

void NumericalEncoder::fit(const Column& col) {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < col.size(); ++i) {
        if (col.is_null(i)) continue;
        sum += col.get_numerical(i);
        ++count;
    }
    if (count == 0) { mean_ = 0.f; std_ = 1.f; fitted_ = true; return; }

    mean_ = static_cast<float>(sum / count);

    double var = 0.0;
    for (std::size_t i = 0; i < col.size(); ++i) {
        if (col.is_null(i)) continue;
        double diff = col.get_numerical(i) - mean_;
        var += diff * diff;
    }
    float s = static_cast<float>(std::sqrt(var / count));
    std_ = (s < 1e-8f) ? 1.f : s;
    fitted_ = true;
}

std::vector<float> NumericalEncoder::transform(const Column& col) const {
    std::vector<float> out(col.size());
    for (std::size_t i = 0; i < col.size(); ++i) {
        float val = col.is_null(i) ? mean_ : static_cast<float>(col.get_numerical(i));
        out[i] = (val - mean_) / std_;
    }
    return out;
}

} // namespace relml
