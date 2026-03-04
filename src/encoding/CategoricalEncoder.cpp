#include "relml/encoding/CategoricalEncoder.h"
#include <stdexcept>

namespace relml {

void CategoricalEncoder::fit(const Column& col) {
    for (std::size_t i = 0; i < col.size(); ++i) {
        if (col.is_null(i)) continue;
        const std::string& val = col.get_categorical(i);
        if (!vocab_.count(val))
            vocab_[val] = vocab_.size();
    }
    fitted_ = true;
}

std::vector<float> CategoricalEncoder::transform(const Column& col) const {
    std::size_t V   = vocab_.size();
    std::size_t N   = col.size();
    std::vector<float> out(N * V, 0.f);
    for (std::size_t i = 0; i < N; ++i) {
        if (col.is_null(i)) continue;
        auto it = vocab_.find(col.get_categorical(i));
        if (it == vocab_.end()) continue;  // unknown category -> all-zero
        out[i * V + it->second] = 1.f;
    }
    return out;
}

} // namespace relml
