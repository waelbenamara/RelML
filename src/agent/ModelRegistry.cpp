#include "relml/agent/ModelRegistry.h"

#include <cstdlib>
#include <filesystem>

namespace relml {

namespace fs = std::filesystem;

static std::string resolve_root(const std::string& hint) {
    if (!hint.empty()) return hint;
    const char* env = std::getenv("RELML_HOME");
    if (env && *env) return env;
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.relml";
}

ModelRegistry::ModelRegistry(const std::string& root)
    : root_(resolve_root(root))
{
    fs::create_directories(root_);
}

std::string ModelRegistry::weight_path(const std::string& fp) const {
    return root_ + "/" + fp + ".bin";
}

bool ModelRegistry::has(const std::string& fp) const {
    return fs::exists(weight_path(fp));
}

void ModelRegistry::remove(const std::string& fp) const {
    std::string p = weight_path(fp);
    if (fs::exists(p)) fs::remove(p);
}

} // namespace relml
