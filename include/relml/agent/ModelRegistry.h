#pragma once

#include <string>

namespace relml {

// Manages the on-disk layout for persisted model weights.
// Keys are TaskSpec fingerprints (short hex strings).
// Binary I/O is delegated to Trainer::save_weights / load_weights,
// keeping serialisation logic in one place.
class ModelRegistry {
public:
    explicit ModelRegistry(const std::string& root = "");

    bool        has(const std::string& fingerprint) const;
    void        remove(const std::string& fingerprint) const;
    std::string weight_path(const std::string& fingerprint) const;

    const std::string& root() const { return root_; }

private:
    std::string root_;
};

} // namespace relml
