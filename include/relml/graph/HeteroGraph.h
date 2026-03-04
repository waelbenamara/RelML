#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace relml {

// (src_table, fk_column, dst_table) — mirrors RelBench's edge naming
struct EdgeType {
    std::string src;
    std::string fk_col;
    std::string dst;

    std::string name() const { return src + "__" + fk_col + "__" + dst; }
    bool operator==(const EdgeType& o) const {
        return src == o.src && fk_col == o.fk_col && dst == o.dst;
    }
};

struct EdgeTypeHash {
    std::size_t operator()(const EdgeType& e) const {
        std::size_t h = std::hash<std::string>{}(e.src);
        h ^= std::hash<std::string>{}(e.fk_col) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<std::string>{}(e.dst)    + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// COO edge list: src_nodes[i] -> dst_nodes[i]
struct EdgeIndex {
    std::vector<int64_t> src;
    std::vector<int64_t> dst;

    std::size_t num_edges() const { return src.size(); }

    void add(int64_t s, int64_t d) {
        src.push_back(s);
        dst.push_back(d);
    }
};

struct HeteroGraph {
    // node_type -> number of nodes (= number of rows in that table)
    std::unordered_map<std::string, int64_t> num_nodes;

    // edge_type -> COO edge index
    std::unordered_map<EdgeType, EdgeIndex, EdgeTypeHash> edge_index;

    std::vector<std::string>  node_types() const;
    std::vector<EdgeType>     edge_types() const;
    int64_t                   total_edges() const;

    void print_summary() const;
};

} // namespace relml