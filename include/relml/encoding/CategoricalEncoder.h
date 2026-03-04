#pragma once

#include "relml/Column.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace relml {

// Maps a CATEGORICAL column to one-hot vectors.
// Unknown categories at transform time map to an all-zero vector.
class CategoricalEncoder {
public:
    void fit(const Column& col);

    // Returns a flat matrix: num_rows x vocab_size, row-major
    std::vector<float> transform(const Column& col) const;

    std::size_t vocab_size() const { return vocab_.size(); }

private:
    std::unordered_map<std::string, std::size_t> vocab_;
    bool fitted_ = false;
};

} // namespace relml
