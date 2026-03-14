#include "relml/encoding/CategoricalEncoder.h"
#include <map>
#include <stdexcept>

namespace relml {

void CategoricalEncoder::fit(const Column& col) {
    // Collect unique non-null values into a sorted container first so that
    // the index assigned to each category is deterministic across runs.
    // std::unordered_map insertion order depends on the hash function and
    // load factor — it varies across compilers, STL versions, and even
    // between runs when ASLR affects pointer-based hash seeds.
    std::map<std::string, int> sorted_unique;
    for (std::size_t i = 0; i < col.size(); ++i) {
        if (col.is_null(i)) continue;
        sorted_unique[col.get_categorical(i)];  // insert with default value 0
    }

    // Assign indices in sorted (alphabetical) order
    vocab_.clear();
    vocab_.reserve(sorted_unique.size());
    std::size_t idx = 0;
    for (const auto& [val, _] : sorted_unique)
        vocab_[val] = idx++;
}

std::vector<float> CategoricalEncoder::transform(const Column& col) const {
    std::size_t V = vocab_.size();
    std::size_t N = col.size();
    std::vector<float> out(N * V, 0.f);
    for (std::size_t i = 0; i < N; ++i) {
        if (col.is_null(i)) continue;
        auto it = vocab_.find(col.get_categorical(i));
        if (it == vocab_.end()) continue;  // unknown category -> all-zero row
        out[i * V + it->second] = 1.f;
    }
    return out;
}

} // namespace relml