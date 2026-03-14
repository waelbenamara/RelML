#pragma once

#include "relml/Column.h"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace relml {

// Maps a CATEGORICAL column to one-hot vectors.
// Unknown categories at transform time map to an all-zero vector.
//
// The vocabulary is built from a sorted std::map so that index assignments
// are deterministic across runs regardless of insertion order. This is
// required for saved weights to be portable: if vocab indices shift between
// a training run and an inference run (different hash table ordering),
// the encoder produces different raw vectors for the same input, invalidating
// every weight in the model.
class CategoricalEncoder {
public:
    void fit(const Column& col);

    // Returns a flat matrix: num_rows x vocab_size, row-major
    std::vector<float> transform(const Column& col) const;

    std::size_t vocab_size() const { return vocab_.size(); }

    // Read-only access for serialisation (e.g. saving vocab to checkpoint)
    const std::unordered_map<std::string, std::size_t>& vocab() const { return vocab_; }

private:
    // Lookup map built from sorted insertion — O(1) access at transform time
    std::unordered_map<std::string, std::size_t> vocab_;
};

} // namespace relml