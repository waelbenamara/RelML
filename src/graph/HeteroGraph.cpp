#include "relml/graph/HeteroGraph.h"
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace relml {

std::vector<std::string> HeteroGraph::node_types() const {
    std::vector<std::string> v;
    v.reserve(num_nodes.size());
    for (const auto& [k, _] : num_nodes) v.push_back(k);
    std::sort(v.begin(), v.end());
    return v;
}

std::vector<EdgeType> HeteroGraph::edge_types() const {
    std::vector<EdgeType> v;
    v.reserve(edge_index.size());
    for (const auto& [k, _] : edge_index) v.push_back(k);
    std::sort(v.begin(), v.end(), [](const EdgeType& a, const EdgeType& b){
        return a.name() < b.name();
    });
    return v;
}

int64_t HeteroGraph::total_edges() const {
    int64_t n = 0;
    for (const auto& [_, ei] : edge_index) n += ei.num_edges();
    return n;
}

void HeteroGraph::print_summary() const {
    std::cout << "HeteroGraph\n"
              << std::string(60, '=') << "\n";

    std::cout << "Node types (" << num_nodes.size() << "):\n";
    for (const auto& nt : node_types())
        std::cout << "  " << std::setw(28) << std::left << nt
                  << num_nodes.at(nt) << " nodes\n";

    std::cout << "\nEdge types (" << edge_index.size() << "):\n";
    for (const auto& et : edge_types())
        std::cout << "  " << std::setw(48) << std::left << et.name()
                  << edge_index.at(et).num_edges() << " edges\n";

    std::cout << "\nTotal edges: " << total_edges() << "\n";
}

} // namespace relml