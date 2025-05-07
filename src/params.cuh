#pragma once

#include "hashmap.cuh"

enum class Metric {
    Euclidean,
    Cosine,
};

struct PartitionParameter {
    uint32_t centroid_num = 10;
    float boundary_factor = 1.05f;
    float sample_factor = 0.01f;
    Metric metric = Metric::Euclidean;
};

struct NNDescentParameter {
    uint32_t graph_degree = 64;
    uint32_t intermediate_graph_degree = 128;
    uint32_t max_iterations = 20;
    float termination_threshold = 0.0001;

    NNDescentParameter(size_t graph_degree) : graph_degree(graph_degree),
                                              intermediate_graph_degree(1.5 * graph_degree) {}
};

struct IndexParameter {
    uint32_t graph_degree = 32;
    uint32_t knn_degree = 32;
    float relaxant_factor = 1.05;
    Metric metric = Metric::Euclidean;
    raft::host_vector_view<uint32_t> segment_start_view;
    raft::host_vector_view<uint32_t> segment_length_view;
    raft::host_vector_view<uint32_t> mapping_view;
};

struct SearchParameter {
    uint32_t top_k = 10;
    uint32_t beam = 100;
    uint32_t max_iterations = 0;
    uint32_t min_iterations = 0;
    uint32_t hash_bit;
    float hash_max_fill_rate = 0.5;
    uint32_t hashmap_min_bitlen = 0;
    uint32_t hash_reset_interval = 1;
    uint32_t search_block_size = 0;
};

inline uint32_t
calculate_hash_bitlen(uint32_t beam, uint32_t graph_degree, float hashmap_max_fill_rate, uint32_t hashmap_min_bitlen) {
    uint32_t hash_bitlen = 0;
    uint32_t max_visited_nodes = beam + graph_degree;
    uint32_t min_bitlen = 8;
    uint32_t max_bitlen = 13;
    if (min_bitlen < hashmap_min_bitlen) {
        min_bitlen = hashmap_min_bitlen;
    }
    hash_bitlen = min_bitlen;
    while (max_visited_nodes > hashmap::get_size(hash_bitlen) * hashmap_max_fill_rate) {
        hash_bitlen += 1;
    }
    if (hash_bitlen > max_bitlen) {
        hash_bitlen = max_bitlen;
    }
    return hash_bitlen;
}

inline uint32_t
calculate_hash_reset_interval(uint32_t beam, uint32_t graph_degree, float hashmap_max_fill_rate, uint32_t hash_bitlen) {
    uint32_t hash_reset_interval = 1024 * 1024;
    hash_reset_interval = 1;
    while (1) {
        const auto max_visited_nodes = beam + graph_degree * (hash_reset_interval + 1);
        if (max_visited_nodes > hashmap::get_size(hash_bitlen) * hashmap_max_fill_rate) {
            break;
        }
        hash_reset_interval += 1;
    }
    return hash_reset_interval;
}

void
adjust_search_params(uint32_t min_iterations, uint32_t &max_iterations, uint32_t &beam) {
    uint32_t _max_iterations = max_iterations;
    if (max_iterations == 0) {
        _max_iterations = 1 + std::min(beam * 1.1, beam + 10.0);
    }
    if (max_iterations < min_iterations) {
        _max_iterations = min_iterations;
    }
    if (max_iterations < _max_iterations) {
        max_iterations = _max_iterations;
    }
    if (beam % 32) {
        uint32_t itopk32 = beam;
        itopk32 += 32 - (beam % 32);
        beam = itopk32;
    }
}