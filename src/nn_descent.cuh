/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>

#include "utils.cuh"
#include "params.cuh"

#include <mma.h>
#include <omp.h>

#include <random>
#include <thread>

#include <iostream>
#include <cfloat>

#include <cuvs/neighbors/cagra.hpp>

using pinned_memory_resource = thrust::universal_host_pinned_memory_resource;
template<typename T>
using pinned_memory_allocator = thrust::mr::stateless_resource_allocator<T, pinned_memory_resource>;

using DistData_t = float;
constexpr uint32_t DEGREE_ON_DEVICE{32};
constexpr uint32_t SEGMENT_SIZE{32};
constexpr uint32_t counter_interval{100};

template<typename Index_t>
struct InternalID_t;

template<>
class InternalID_t<int> {
private:
    using Index_t = int;
    Index_t id_{std::numeric_limits<Index_t>::max()};

public:
    __host__ __device__ bool is_new() const { return id_ >= 0; }

    __host__ __device__ Index_t &id_with_flag() { return id_; }

    __host__ __device__ Index_t id() const {
        if (is_new()) return id_;
        return -id_ - 1;
    }

    __host__ __device__ void mark_old() {
        if (id_ >= 0) id_ = -id_ - 1;
    }

    __host__ __device__ bool operator==(const InternalID_t<int> &other) const {
        return id() == other.id();
    }
};

template<typename Index_t>
struct ResultItem;

template<>
class ResultItem<int> {
private:
    using Index_t = int;
    Index_t id_;
    DistData_t dist_;

public:
    __host__ __device__ ResultItem()
            : id_(INT_MAX), dist_(FLT_MAX) {};

    __host__ __device__ ResultItem(const Index_t id_with_flag, const DistData_t dist)
            : id_(id_with_flag), dist_(dist) {};

    __host__ __device__ bool is_new() const { return id_ >= 0; }

    __host__ __device__ Index_t &id_with_flag() { return id_; }

    __host__ __device__ Index_t id() const {
        if (is_new()) return id_;
        return -id_ - 1;
    }

    __host__ __device__ DistData_t &dist() { return dist_; }

    __host__ __device__ void mark_old() {
        if (id_ >= 0) id_ = -id_ - 1;
    }

    __host__ __device__ bool operator<(const ResultItem<Index_t> &other) const {
        if (dist_ == other.dist_) return id() < other.id();
        return dist_ < other.dist_;
    }

    __host__ __device__ bool operator==(const ResultItem<Index_t> &other) const {
        return id() == other.id();
    }

    __host__ __device__ bool operator>=(const ResultItem<Index_t> &other) const {
        return !(*this < other);
    }

    __host__ __device__ bool operator<=(const ResultItem<Index_t> &other) const {
        return (*this == other) || (*this < other);
    }

    __host__ __device__ bool operator>(const ResultItem<Index_t> &other) const {
        return !(*this <= other);
    }

    __host__ __device__ bool operator!=(const ResultItem<Index_t> &other) const {
        return !(*this == other);
    }
};

template<typename T>
int get_batch_size(const int it_now, const T nrow, const int batch_size) {
    int it_total = ceildiv(nrow, batch_size);
    return (it_now == it_total - 1) ? nrow - it_now * batch_size : batch_size;
}

template<typename T>
constexpr __host__ __device__ __forceinline__ int skew_dim(int ndim) {
    if constexpr (std::is_same<T, float>::value) {
        ndim = ceildiv(ndim, 4) * 4;
        return ndim + (ndim % 32 == 0) * 4;
    }
}

template<typename T>
__device__ __forceinline__ ResultItem<T> xor_swap(ResultItem<T> x, int mask, int dir) {
    ResultItem<T> y;
    y.dist() = __shfl_xor_sync(warp_full_mask(), x.dist(), mask, warp_size());
    y.id_with_flag() =
            __shfl_xor_sync(warp_full_mask(), x.id_with_flag(), mask, warp_size());
    return x < y == dir ? y : x;
}

__device__ __forceinline__ int xor_swap(int x, int mask, int dir) {
    int y = __shfl_xor_sync(warp_full_mask(), x, mask, warp_size());
    return x < y == dir ? y : x;
}

__device__ __forceinline__ uint32_t bfe(uint32_t lane_id, uint32_t pos) {
    uint32_t res;
    asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
    return res;
}

template<typename T>
__device__ __forceinline__ void warp_bitonic_sort(T *element_ptr, uint32_t lane_id) {
    static_assert(warp_size() == 32);
    auto &element = *element_ptr;
    element = xor_swap(element, 0x01, bfe(lane_id, 1) ^ bfe(lane_id, 0));
    element = xor_swap(element, 0x02, bfe(lane_id, 2) ^ bfe(lane_id, 1));
    element = xor_swap(element, 0x01, bfe(lane_id, 2) ^ bfe(lane_id, 0));
    element = xor_swap(element, 0x04, bfe(lane_id, 3) ^ bfe(lane_id, 2));
    element = xor_swap(element, 0x02, bfe(lane_id, 3) ^ bfe(lane_id, 1));
    element = xor_swap(element, 0x01, bfe(lane_id, 3) ^ bfe(lane_id, 0));
    element = xor_swap(element, 0x08, bfe(lane_id, 4) ^ bfe(lane_id, 3));
    element = xor_swap(element, 0x04, bfe(lane_id, 4) ^ bfe(lane_id, 2));
    element = xor_swap(element, 0x02, bfe(lane_id, 4) ^ bfe(lane_id, 1));
    element = xor_swap(element, 0x01, bfe(lane_id, 4) ^ bfe(lane_id, 0));
    element = xor_swap(element, 0x10, bfe(lane_id, 4));
    element = xor_swap(element, 0x08, bfe(lane_id, 3));
    element = xor_swap(element, 0x04, bfe(lane_id, 2));
    element = xor_swap(element, 0x02, bfe(lane_id, 1));
    element = xor_swap(element, 0x01, bfe(lane_id, 0));
    return;
}

struct BuildConfig {
    uint32_t max_dataset_size;
    uint32_t dataset_dim;
    uint32_t node_degree{64};
    uint32_t internal_node_degree{0};
    uint32_t max_iterations{50};
    float termination_threshold{0.0001};
    uint32_t segment_num;
    uint32_t *segment_length;
};

template<typename Index_t>
class BloomFilter {
public:
    BloomFilter(uint32_t nrow, uint32_t num_sets_per_list, uint32_t num_hashs)
            : nrow_(nrow),
              num_sets_per_list_(num_sets_per_list),
              num_hashs_(num_hashs),
              bitsets_(nrow * num_bits_per_set_ * num_sets_per_list) {
    }

    void add(uint32_t list_id, Index_t key) {
        if (is_cleared) { is_cleared = false; }
        uint32_t hash = hash_0(key);
        uint32_t global_set_idx = list_id * num_bits_per_set_ * num_sets_per_list_ +
                                  key % num_sets_per_list_ * num_bits_per_set_;
        bitsets_[global_set_idx + hash % num_bits_per_set_] = 1;
        for (uint32_t i = 1; i < num_hashs_; i++) {
            hash = hash + hash_1(key);
            bitsets_[global_set_idx + hash % num_bits_per_set_] = 1;
        }
    }

    bool check(uint32_t list_id, Index_t key) {
        bool is_present = true;
        uint32_t hash = hash_0(key);
        uint32_t global_set_idx = list_id * num_bits_per_set_ * num_sets_per_list_ +
                                  key % num_sets_per_list_ * num_bits_per_set_;

        is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];

        if (!is_present) return false;
        for (uint32_t i = 1; i < num_hashs_; i++) {
            hash = hash + hash_1(key);
            is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];
            if (!is_present) return false;
        }
        return true;
    }

    void clear() {
        if (is_cleared) return;
#pragma omp parallel for
        for (uint32_t i = 0; i < nrow_ * num_bits_per_set_ * num_sets_per_list_; i++) {
            bitsets_[i] = 0;
        }
        is_cleared = true;
    }

private:
    uint32_t hash_0(uint32_t value) {
        value *= 1103515245;
        value += 12345;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        return value;
    }

    uint32_t hash_1(uint32_t value) {
        value *= 1664525;
        value += 1013904223;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        return value;
    }

    static constexpr uint32_t num_bits_per_set_ = 512;
    bool is_cleared{true};
    std::vector<bool> bitsets_;
    uint32_t nrow_;
    uint32_t num_sets_per_list_;
    uint32_t num_hashs_;
};

template<typename Index_t>
struct GnndGraph {
    static constexpr int segment_size = 32;
    InternalID_t<Index_t> *h_graph;

    uint32_t nrow;
    uint32_t node_degree;
    uint32_t num_samples;
    uint32_t num_segments;

    uint32_t segment_num;

    raft::host_matrix<DistData_t, uint32_t, raft::row_major> h_dists;

    thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_graph_new;
    thrust::host_vector<int2, pinned_memory_allocator<int2>> h_list_sizes_new;

    thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_graph_old;
    thrust::host_vector<int2, pinned_memory_allocator<int2>> h_list_sizes_old;
    std::vector<BloomFilter<Index_t>> bloom_filter;

    GnndGraph(const GnndGraph &) = delete;

    GnndGraph &operator=(const GnndGraph &) = delete;

    GnndGraph(const uint32_t nrow, const uint32_t node_degree, const uint32_t internal_node_degree,
              const uint32_t num_samples, uint32_t segment_num, uint32_t *segment_length);

    void init_random_graph(raft::host_vector_view<uint32_t> segment_start_view,
                           raft::host_vector_view<uint32_t> segment_length_view);

    void sample_graph_new(InternalID_t<Index_t> *new_neighbors, const uint32_t width,
                          raft::host_vector_view<uint32_t> segment_start_view,
                          raft::host_vector_view<uint32_t> segment_length_view);

    void sample_graph(bool sample_new);

    void update_graph(const InternalID_t<Index_t> *new_neighbors,
                      const DistData_t *new_dists,
                      const uint32_t width,
                      std::atomic<int64_t> &update_counter);

    void sample_segment_graph(int segment_start, int segment_end, bool sample_new);

    void update_segment_graph(const InternalID_t<Index_t> *new_neighbors,
                              const DistData_t *new_dists,
                              const uint32_t width,
                              std::atomic<int64_t> &update_counter,
                              int segment_start, int segment_end);

    void clear();

    ~GnndGraph();
};

template<typename Data_t = float, typename Index_t = int>
class GNND {
public:
    GNND(raft::resources const &handle, const BuildConfig &build_config);

    GNND(const GNND &) = delete;

    GNND &operator=(const GNND &) = delete;

    ~GNND() = default;

    void build(std::optional<raft::device_matrix<Data_t>> &data, uint32_t nrow, Index_t *output_graph,
               raft::host_vector_view<uint32_t> segment_start_view,
               raft::host_vector_view<uint32_t> segment_length_view);

    using ID_t = InternalID_t<Index_t>;

private:
    void add_reverse_edges(Index_t *graph_ptr,
                           Index_t *h_rev_graph_ptr,
                           Index_t *d_rev_graph_ptr,
                           int2 *list_sizes,
                           int segment_start,
                           int segment_length,
                           cudaStream_t stream = 0);

    void local_join(int segment_start, int segment_length, cudaStream_t stream = 0);

    raft::resources const &handle;

    BuildConfig build_config_;
    GnndGraph<Index_t> graph_;
    std::vector<std::unique_ptr<std::atomic<int64_t>>> update_counter_;

    uint32_t nrow_;
    uint32_t ndim_;

    raft::device_matrix<__half, uint32_t, raft::row_major> d_data_;
    raft::device_vector<DistData_t, uint32_t> l2_norms_;

    raft::device_matrix<ID_t, uint32_t, raft::row_major> graph_buffer_;
    raft::device_matrix<DistData_t, uint32_t, raft::row_major> dists_buffer_;

    thrust::host_vector<ID_t, pinned_memory_allocator<ID_t>> graph_host_buffer_;
    thrust::host_vector<DistData_t, pinned_memory_allocator<DistData_t>> dists_host_buffer_;

    raft::device_vector<int, uint32_t> d_locks_;

    thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_rev_graph_new_;
    thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_graph_old_;
    thrust::host_vector<Index_t, pinned_memory_allocator<Index_t>> h_rev_graph_old_;

    raft::device_vector<int2, uint32_t> d_list_sizes_new_;
    raft::device_vector<int2, uint32_t> d_list_sizes_old_;
};

constexpr int TILE_ROW_WIDTH = 64;
constexpr int TILE_COL_WIDTH = 128;

constexpr int NUM_SAMPLES = 32;
constexpr int MAX_NUM_BI_SAMPLES = 64;
constexpr int SKEWED_MAX_NUM_BI_SAMPLES = skew_dim<float>(MAX_NUM_BI_SAMPLES);
constexpr int BLOCK_SIZE = 512;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template<typename Data_t>
__device__ __forceinline__ void load_vec(Data_t *vec_buffer,
                                         const Data_t *d_vec,
                                         const uint32_t load_dims,
                                         const uint32_t padding_dims,
                                         const uint32_t lane_id) {
    if constexpr (std::is_same_v<Data_t, float> or std::is_same_v<Data_t, uint8_t> or
                  std::is_same_v<Data_t, int8_t>) {
        constexpr uint32_t num_load_elems_per_warp = warp_size();
        for (uint32_t step = 0; step < ceildiv(padding_dims, num_load_elems_per_warp); step++) {
            uint32_t idx = step * num_load_elems_per_warp + lane_id;
            if (idx < load_dims) {
                vec_buffer[idx] = d_vec[idx];
            } else if (idx < padding_dims) {
                vec_buffer[idx] = 0.0f;
            }
        }
    }
    if constexpr (std::is_same_v<Data_t, __half>) {
        if ((size_t) d_vec % sizeof(float2) == 0 && (size_t) vec_buffer % sizeof(float2) == 0 &&
            load_dims % 4 == 0 && padding_dims % 4 == 0) {
            constexpr uint32_t num_load_elems_per_warp = warp_size() * 4;
#pragma unroll
            for (uint32_t step = 0; step < ceildiv(padding_dims, num_load_elems_per_warp); step++) {
                uint32_t idx_in_vec = step * num_load_elems_per_warp + lane_id * 4;
                if (idx_in_vec + 4 <= load_dims) {
                    *(float2 *) (vec_buffer + idx_in_vec) = *(float2 *) (d_vec + idx_in_vec);
                } else if (idx_in_vec + 4 <= padding_dims) {
                    *(float2 *) (vec_buffer + idx_in_vec) = float2({0.0f, 0.0f});
                }
            }
        } else {
            constexpr uint32_t num_load_elems_per_warp = warp_size();
            for (uint32_t step = 0; step < ceildiv(padding_dims, num_load_elems_per_warp); step++) {
                uint32_t idx = step * num_load_elems_per_warp + lane_id;
                if (idx < load_dims) {
                    vec_buffer[idx] = d_vec[idx];
                } else if (idx < padding_dims) {
                    vec_buffer[idx] = 0.0f;
                }
            }
        }
    }
}

template<typename Data_t>
__global__ void preprocess_data_kernel(const Data_t *input_data,
                                       __half *output_data,
                                       uint32_t dim,
                                       DistData_t *l2_norms) {
    extern __shared__ char buffer[];
    __shared__ float l2_norm;
    auto *s_vec = reinterpret_cast<Data_t *>(buffer);

    uint32_t list_id = blockIdx.x;
    uint32_t lane_id = threadIdx.x % warp_size();

    load_vec(s_vec, input_data + blockIdx.x * dim, dim, dim, lane_id);
    if (threadIdx.x == 0) { l2_norm = 0; }
    __syncthreads();

    uint32_t num_step = ceildiv(dim, static_cast<uint32_t>(warp_size()));
    for (uint32_t step = 0; step < num_step; step++) {
        uint32_t idx = step * warp_size() + lane_id;
        float part_dist = 0;
        if (idx < dim) {
            part_dist = s_vec[idx];
            part_dist = part_dist * part_dist;
        }
        __syncwarp();
        for (uint32_t offset = warp_size() >> 1; offset >= 1; offset >>= 1) {
            part_dist += __shfl_down_sync(warp_full_mask(), part_dist, offset);
        }
        if (lane_id == 0) { l2_norm += part_dist; }
        __syncwarp();
    }

    for (uint32_t step = 0; step < num_step; step++) {
        uint32_t idx = step * warp_size() + lane_id;
        if (idx < dim) {
            if (l2_norms == nullptr) {
                output_data[list_id * dim + idx] = (float) input_data[list_id * dim + idx] / sqrt(l2_norm);
            } else {
                output_data[list_id * dim + idx] = input_data[list_id * dim + idx];
                if (idx == 0) { l2_norms[list_id] = l2_norm; }
            }
        }
    }
}

template<typename Index_t>
__global__ void add_rev_edges_kernel(const Index_t *graph,
                                     Index_t *rev_graph,
                                     int num_samples,
                                     int2 *list_sizes) {
    uint32_t list_id = blockIdx.x;
    int2 list_size = list_sizes[list_id];

    for (uint32_t idx = threadIdx.x; idx < list_size.x; idx += blockDim.x) {
        uint32_t rev_list_id = graph[list_id * num_samples + idx];
        uint32_t idx_in_rev_list = atomicAdd(&list_sizes[rev_list_id].y, 1);
        if (idx_in_rev_list >= num_samples) {
            atomicExch(&list_sizes[rev_list_id].y, num_samples);
        } else {
            rev_graph[rev_list_id * num_samples + idx_in_rev_list] = list_id;
        }
    }
}

template<typename Index_t, typename ID_t = InternalID_t<Index_t>>
__device__ void insert_to_global_graph(ResultItem<Index_t> elem,
                                       uint32_t list_id,
                                       ID_t *graph,
                                       DistData_t *dists,
                                       uint32_t node_degree,
                                       int *locks) {
    uint32_t lane_id = threadIdx.x % warp_size();
    uint32_t global_idx_base = list_id * node_degree;
    if (elem.id() == list_id) return;

    const uint32_t num_segments = ceildiv(node_degree, static_cast<uint32_t>(warp_size()));

    uint32_t loop_flag = 0;
    do {
        int segment_id = elem.id() % num_segments;
        if (lane_id == 0) {
            loop_flag = atomicCAS(&locks[list_id * num_segments + segment_id], 0, 1) == 0;
        }

        loop_flag = __shfl_sync(warp_full_mask(), loop_flag, 0);

        if (loop_flag == 1) {
            ResultItem<Index_t> knn_list_frag;
            uint32_t local_idx = segment_id * warp_size() + lane_id;
            uint32_t global_idx = global_idx_base + local_idx;
            if (local_idx < node_degree) {
                knn_list_frag.id_with_flag() = graph[global_idx].id_with_flag();
                knn_list_frag.dist() = dists[global_idx];
            }

            int pos_to_insert = -1;
            ResultItem<Index_t> prev_elem;

            prev_elem.id_with_flag() =
                    __shfl_up_sync(warp_full_mask(), knn_list_frag.id_with_flag(), 1);
            prev_elem.dist() = __shfl_up_sync(warp_full_mask(), knn_list_frag.dist(), 1);

            if (lane_id == 0) {
                prev_elem = ResultItem<Index_t>{INT_MIN, FLT_MIN};
            }
            if (elem > prev_elem && elem < knn_list_frag) {
                pos_to_insert = segment_id * warp_size() + lane_id;
            } else if (elem == prev_elem || elem == knn_list_frag) {
                pos_to_insert = -2;
            }
            uint32_t mask = __ballot_sync(warp_full_mask(), pos_to_insert >= 0);
            if (mask) {
                uint32_t set_lane_id = __ffs(mask) - 1;// find first set bit
                pos_to_insert = __shfl_sync(warp_full_mask(), pos_to_insert, set_lane_id);
            }

            if (pos_to_insert >= 0) {
                if (local_idx > pos_to_insert) {
                    local_idx++;
                } else if (local_idx == pos_to_insert) {
                    graph[global_idx_base + local_idx].id_with_flag() = elem.id_with_flag();
                    dists[global_idx_base + local_idx] = elem.dist();
                    local_idx++;
                }
                uint32_t global_pos = global_idx_base + local_idx;
                if (local_idx < (segment_id + 1) * warp_size() && local_idx < node_degree) {
                    graph[global_pos].id_with_flag() = knn_list_frag.id_with_flag();
                    dists[global_pos] = knn_list_frag.dist();
                }
            }
            __threadfence();
            if (loop_flag && lane_id == 0) { atomicExch(&locks[list_id * num_segments + segment_id], 0); }
        }
    } while (!loop_flag);
}

template<typename Index_t>
__device__ ResultItem<Index_t> get_min_item(const Index_t id,
                                            const uint32_t idx_in_list,
                                            const Index_t *neighbors,
                                            const DistData_t *distances,
                                            const bool find_in_row = true) {
    uint32_t lane_id = threadIdx.x % warp_size();

    static_assert(MAX_NUM_BI_SAMPLES == 64);
    uint32_t idx[MAX_NUM_BI_SAMPLES / warp_size()];
    float dist[MAX_NUM_BI_SAMPLES / warp_size()] = {FLT_MAX, FLT_MAX};
    idx[0] = lane_id;
    idx[1] = warp_size() + lane_id;

    if (neighbors[idx[0]] != id) {
        dist[0] = find_in_row ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + lane_id]
                              : distances[idx_in_list + lane_id * SKEWED_MAX_NUM_BI_SAMPLES];
    }

    if (neighbors[idx[1]] != id) {
        dist[1] = find_in_row ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + warp_size() + lane_id]
                              : distances[idx_in_list + (warp_size() + lane_id) * SKEWED_MAX_NUM_BI_SAMPLES];
    }

    if (dist[1] < dist[0]) {
        dist[0] = dist[1];
        idx[0] = idx[1];
    }
    __syncwarp();
    for (uint32_t offset = warp_size() >> 1; offset >= 1; offset >>= 1) {
        uint32_t other_idx = __shfl_down_sync(warp_full_mask(), idx[0], offset);
        float other_dist = __shfl_down_sync(warp_full_mask(), dist[0], offset);
        if (other_dist < dist[0]) {
            dist[0] = other_dist;
            idx[0] = other_idx;
        }
    }

    ResultItem<Index_t> result;
    result.dist() = __shfl_sync(warp_full_mask(), dist[0], 0);
    result.id_with_flag() = neighbors[__shfl_sync(warp_full_mask(), idx[0], 0)];
    return result;
}

template<typename T>
__device__ __forceinline__ void remove_duplicates(
        T *list_a, uint32_t list_a_size, T *list_b, uint32_t list_b_size, uint32_t &unique_counter,
        uint32_t execute_warp_id) {
    static_assert(warp_size() == 32);
    if (!(threadIdx.x >= execute_warp_id * warp_size() &&
          threadIdx.x < execute_warp_id * warp_size() + warp_size())) {
        return;
    }
    uint32_t lane_id = threadIdx.x % warp_size();
    T elem = INT_MAX;
    if (lane_id < list_a_size) { elem = list_a[lane_id]; }
    warp_bitonic_sort(&elem, lane_id);

    if (elem != INT_MAX) { list_a[lane_id] = elem; }

    T elem_b = INT_MAX;

    if (lane_id < list_b_size) { elem_b = list_b[lane_id]; }
    __syncwarp();

    uint32_t idx_l = 0;
    uint32_t idx_r = list_a_size;
    bool existed = false;
    while (idx_l < idx_r) {
        uint32_t idx = (idx_l + idx_r) / 2;
        T elem_a = list_a[idx];
        if (elem_a == elem_b) {
            existed = true;
            break;
        }
        if (elem_b > elem_a) {
            idx_l = idx + 1;
        } else {
            idx_r = idx;
        }
    }
    if (!existed && elem_b != INT_MAX) {
        uint32_t idx = atomicAdd(&unique_counter, 1);
        list_a[list_a_size + idx] = elem_b;
    }
}

template<typename Index_t, typename ID_t = InternalID_t<Index_t>>
__global__ void
#ifdef __CUDA_ARCH__
#if (__CUDA_ARCH__) == 750 || ((__CUDA_ARCH__) >= 860 && (__CUDA_ARCH__) <= 890)
__launch_bounds__(BLOCK_SIZE)
#else
__launch_bounds__(BLOCK_SIZE, 4)
#endif
#endif
local_join_kernel(const Index_t *graph_new,
                  const Index_t *rev_graph_new,
                  const int2 *sizes_new,
                  const Index_t *graph_old,
                  const Index_t *rev_graph_old,
                  const int2 *sizes_old,
                  const uint32_t width,
                  const __half *data,
                  const uint32_t data_dim,
                  ID_t *graph,
                  DistData_t *dists,
                  uint32_t graph_width,
                  int *locks,
                  DistData_t *l2_norms) {
#if (__CUDA_ARCH__ >= 700)
    using namespace nvcuda;
    __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];

    constexpr int APAD = 8;
    constexpr int BPAD = 8;
    __shared__ __half s_nv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + APAD];  // New vectors
    __shared__ __half s_ov[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];  // Old vectors
    static_assert(sizeof(float) * MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES <=
                  sizeof(__half) * MAX_NUM_BI_SAMPLES * (TILE_COL_WIDTH + BPAD));
    // s_distances: MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES, reuse the space of s_ov
    float *s_distances = (float *) &s_ov[0][0];
    uint32_t *s_unique_counter = (uint32_t *) &s_ov[0][0];

    if (threadIdx.x == 0) {
        s_unique_counter[0] = 0;
        s_unique_counter[1] = 0;
    }

    Index_t *new_neighbors = s_list;
    Index_t *old_neighbors = s_list + MAX_NUM_BI_SAMPLES;

    size_t list_id = blockIdx.x;
    int2 list_new_size2 = sizes_new[list_id];
    uint32_t list_new_size = list_new_size2.x + list_new_size2.y;
    int2 list_old_size2 = sizes_old[list_id];
    uint32_t list_old_size = list_old_size2.x + list_old_size2.y;

    if (!list_new_size) return;
    uint32_t tx = threadIdx.x;

    if (tx < list_new_size2.x) {
        new_neighbors[tx] = graph_new[list_id * width + tx];
    } else if (tx >= list_new_size2.x && tx < list_new_size) {
        new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
    }

    if (tx < list_old_size2.x) {
        old_neighbors[tx] = graph_old[list_id * width + tx];
    } else if (tx >= list_old_size2.x && tx < list_old_size) {
        old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
    }

    __syncthreads();

    remove_duplicates(new_neighbors,
                      list_new_size2.x,
                      new_neighbors + list_new_size2.x,
                      list_new_size2.y,
                      s_unique_counter[0],
                      0);

    remove_duplicates(old_neighbors,
                      list_old_size2.x,
                      old_neighbors + list_old_size2.x,
                      list_old_size2.y,
                      s_unique_counter[1],
                      1);
    __syncthreads();
    list_new_size = list_new_size2.x + s_unique_counter[0];
    list_old_size = list_old_size2.x + s_unique_counter[1];

    uint32_t warp_id = threadIdx.x / warp_size();
    uint32_t lane_id = threadIdx.x % warp_size();
    constexpr int num_warps = BLOCK_SIZE / warp_size();

    uint32_t warp_id_y = warp_id / 4;
    uint32_t warp_id_x = warp_id % 4;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0);
    for (uint32_t step = 0; step < ceildiv(static_cast<int>(data_dim), TILE_COL_WIDTH); step++) {
        uint32_t num_load_elems = (step == ceildiv(static_cast<int>(data_dim), TILE_COL_WIDTH) - 1)
                                  ? data_dim - step * TILE_COL_WIDTH
                                  : TILE_COL_WIDTH;
#pragma unroll
        for (uint32_t i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
            uint32_t idx = i * num_warps + warp_id;
            if (idx < list_new_size) {
                uint32_t neighbor_id = new_neighbors[idx];
                uint32_t idx_in_data = neighbor_id * data_dim;
                load_vec(s_nv[idx],
                         data + idx_in_data + step * TILE_COL_WIDTH,
                         num_load_elems,
                         TILE_COL_WIDTH,
                         lane_id);
            }
        }
        __syncthreads();

        for (uint32_t i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
            wmma::load_matrix_sync(a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
            wmma::load_matrix_sync(b_frag, s_nv[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
    }

    wmma::store_matrix_sync(
            s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
            c_frag,
            SKEWED_MAX_NUM_BI_SAMPLES,
            wmma::mem_row_major);
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
        if (i % SKEWED_MAX_NUM_BI_SAMPLES < list_new_size &&
            i / SKEWED_MAX_NUM_BI_SAMPLES < list_new_size) {
            if (l2_norms == nullptr) {
                s_distances[i] = -s_distances[i];
            } else {
                s_distances[i] = l2_norms[new_neighbors[i % SKEWED_MAX_NUM_BI_SAMPLES]] +
                                 l2_norms[new_neighbors[i / SKEWED_MAX_NUM_BI_SAMPLES]] -
                                 2.0 * s_distances[i];
            }
        } else {
            s_distances[i] = FLT_MAX;
        }
    }
    __syncthreads();

    for (uint32_t step = 0; step < ceildiv(static_cast<int>(list_new_size), num_warps); step++) {
        uint32_t idx_in_list = step * num_warps + tx / warp_size();
        if (idx_in_list >= list_new_size) continue;
        auto min_elem = get_min_item(s_list[idx_in_list], idx_in_list, new_neighbors, s_distances);
        if (min_elem.id() < gridDim.x) {
            insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
        }
    }

    if (!list_old_size) return;

    __syncthreads();

    wmma::fill_fragment(c_frag, 0.0);
    for (uint32_t step = 0; step < ceildiv(static_cast<int>(data_dim), TILE_COL_WIDTH); step++) {
        uint32_t num_load_elems = (step == ceildiv(static_cast<int>(data_dim), TILE_COL_WIDTH) - 1)
                                  ? data_dim - step * TILE_COL_WIDTH
                                  : TILE_COL_WIDTH;
        if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
            for (uint32_t i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
                uint32_t idx = i * num_warps + warp_id;
                if (idx < list_new_size) {
                    uint32_t neighbor_id = new_neighbors[idx];
                    uint32_t idx_in_data = neighbor_id * data_dim;
                    load_vec(s_nv[idx],
                             data + idx_in_data + step * TILE_COL_WIDTH,
                             num_load_elems,
                             TILE_COL_WIDTH,
                             lane_id);
                }
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
            uint32_t idx = i * num_warps + warp_id;
            if (idx < list_old_size) {
                uint32_t neighbor_id = old_neighbors[idx];
                uint32_t idx_in_data = neighbor_id * data_dim;
                load_vec(s_ov[idx],
                         data + idx_in_data + step * TILE_COL_WIDTH,
                         num_load_elems,
                         TILE_COL_WIDTH,
                         lane_id);
            }
        }
        __syncthreads();

        for (uint32_t i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
            wmma::load_matrix_sync(a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
            wmma::load_matrix_sync(b_frag, s_ov[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
    }

    wmma::store_matrix_sync(
            s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
            c_frag,
            SKEWED_MAX_NUM_BI_SAMPLES,
            wmma::mem_row_major);
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
        if (i % SKEWED_MAX_NUM_BI_SAMPLES < list_old_size &&
            i / SKEWED_MAX_NUM_BI_SAMPLES < list_new_size) {
            if (l2_norms == nullptr) {
                s_distances[i] = -s_distances[i];
            } else {
                s_distances[i] = l2_norms[old_neighbors[i % SKEWED_MAX_NUM_BI_SAMPLES]] +
                                 l2_norms[new_neighbors[i / SKEWED_MAX_NUM_BI_SAMPLES]] -
                                 2.0 * s_distances[i];
            }
        } else {
            s_distances[i] = FLT_MAX;
        }
    }
    __syncthreads();

    for (uint32_t step = 0; step < ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); step++) {
        uint32_t idx_in_list = step * num_warps + tx / warp_size();
        if (idx_in_list >= list_new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
        if (idx_in_list >= MAX_NUM_BI_SAMPLES + list_old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2)
            continue;
        ResultItem<Index_t> min_elem{INT_MAX, FLT_MAX};
        if (idx_in_list < MAX_NUM_BI_SAMPLES) {
            auto temp_min_item =
                    get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
            if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
        } else {
            auto temp_min_item = get_min_item(
                    s_list[idx_in_list], idx_in_list - MAX_NUM_BI_SAMPLES, new_neighbors, s_distances, false);
            if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
        }

        if (min_elem.id() < gridDim.x) {
            insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
        }
    }
#endif
}

template<typename Index_t>
uint32_t insert_to_ordered_list(InternalID_t<Index_t> *list,
                                DistData_t *dist_list,
                                const uint32_t width,
                                const InternalID_t<Index_t> neighbor_id,
                                const DistData_t dist) {
    if (dist > dist_list[width - 1]) { return width; }

    uint32_t idx_insert = width;
    bool position_found = false;
    for (uint32_t i = 0; i < width; i++) {
        if (list[i].id() == neighbor_id.id()) { return width; }
        if (!position_found && dist_list[i] > dist) {
            idx_insert = i;
            position_found = true;
        }
    }
    if (idx_insert == width) return idx_insert;

    memmove(list + idx_insert + 1, list + idx_insert, sizeof(*list) * (width - idx_insert - 1));
    memmove(dist_list + idx_insert + 1,
            dist_list + idx_insert,
            sizeof(*dist_list) * (width - idx_insert - 1));

    list[idx_insert] = neighbor_id;
    dist_list[idx_insert] = dist;
    return idx_insert;
}


template<typename Index_t>
GnndGraph<Index_t>::GnndGraph(const uint32_t nrow,
                              const uint32_t node_degree,
                              const uint32_t internal_node_degree,
                              const uint32_t num_samples,
                              uint32_t segment, uint32_t *segment_length)
        : nrow(nrow),
          node_degree(node_degree),
          num_samples(num_samples),
          segment_num(segment),
          h_dists{raft::make_host_matrix<DistData_t, uint32_t, raft::row_major>(nrow, node_degree)},
          h_graph_new(nrow * num_samples),
          h_list_sizes_new(nrow),
          h_graph_old(nrow * num_samples),
          h_list_sizes_old{nrow} {
    assert(node_degree % segment_size == 0);
    assert(internal_node_degree % segment_size == 0);

    for(int i=0;i<segment_num;i++){
        BloomFilter<Index_t> bloom(segment_length[i], internal_node_degree / segment_size, 3);
        bloom_filter.emplace_back(bloom);
    }

    num_segments = node_degree / segment_size;
    h_graph = nullptr;
}

template<typename Index_t>
void GnndGraph<Index_t>::sample_graph_new(InternalID_t<Index_t> *new_neighbors, const uint32_t width,
                                          raft::host_vector_view<uint32_t> segment_start_view,
                                          raft::host_vector_view<uint32_t> segment_length_view) {
    uint32_t segment = segment_start_view.extent(0);

#pragma omp parallel for
    for (uint32_t i = 0; i < nrow; i++) {
        uint32_t segment_id = 0;
        for (; segment_id < segment; segment_id++) {
            uint32_t segment_end = segment_start_view(segment_id) + segment_length_view(segment_id);
            if (i < segment_end) break;
        }

        auto list_new = h_graph_new.data() + i * num_samples;
        h_list_sizes_new[i].x = 0;
        h_list_sizes_new[i].y = 0;

        for (uint32_t j = 0; j < width; j++) {
            auto new_neighbor_id = new_neighbors[i * width + j].id();
            if ((uint32_t) new_neighbor_id >= segment_length_view(segment_id)) break;
            if (bloom_filter[segment_id].check(i-segment_start_view(segment_id), new_neighbor_id)) { continue; }
            bloom_filter[segment_id].add(i-segment_start_view(segment_id), new_neighbor_id);
            new_neighbors[i * width + j].mark_old();
            list_new[h_list_sizes_new[i].x++] = new_neighbor_id;
            if (h_list_sizes_new[i].x == num_samples) break;
        }
    }
}

template<typename Index_t>
void GnndGraph<Index_t>::init_random_graph(raft::host_vector_view<uint32_t> segment_start_view,
                                           raft::host_vector_view<uint32_t> segment_length_view) {
    uint32_t segment = segment_start_view.extent(0);
    for (uint32_t seg_idx = 0; seg_idx < num_segments; seg_idx++) {
        std::vector<std::vector<Index_t>> rand_seq(segment);
        for (uint32_t i = 0; i < segment; i++) {
            rand_seq[i].resize(segment_length_view(i) / num_segments);
            std::iota(rand_seq[i].begin(), rand_seq[i].end(), segment_start_view(i) / num_segments);
            auto gen = std::default_random_engine{seg_idx};
            std::shuffle(rand_seq[i].begin(), rand_seq[i].end(), gen);
        }

#pragma omp parallel for
        for (uint32_t i = 0; i < nrow; i++) {
            uint32_t segment_id = 0;
            for (; segment_id < segment; segment_id++) {
                uint32_t segment_end = segment_start_view(segment_id) + segment_length_view(segment_id);
                if (i < segment_end) break;
            }
            uint32_t base_idx = i * node_degree + seg_idx * SEGMENT_SIZE;
            auto h_neighbor_list = h_graph + base_idx;
            auto h_dist_list = h_dists.data_handle() + base_idx;
            for (uint32_t j = 0; j < SEGMENT_SIZE; j++) {
                uint32_t idx = base_idx + j;
                Index_t id = rand_seq[segment_id][idx % rand_seq[segment_id].size()] * num_segments + seg_idx;
                if ((uint32_t) id == i) {
                    id = rand_seq[segment_id][(idx + SEGMENT_SIZE) % rand_seq[segment_id].size()] * num_segments +
                         seg_idx;
                }
                h_neighbor_list[j].id_with_flag() = id - segment_start_view(segment_id);
                h_dist_list[j] = FLT_MAX;
            }
        }
    }
}

template<typename Index_t>
void GnndGraph<Index_t>::sample_graph(bool sample_new) {
#pragma omp parallel for
    for (uint32_t i = 0; i < nrow; i++) {
        h_list_sizes_old[i].x = 0;
        h_list_sizes_old[i].y = 0;
        h_list_sizes_new[i].x = 0;
        h_list_sizes_new[i].y = 0;

        auto list = h_graph + i * node_degree;
        auto list_old = h_graph_old.data() + i * num_samples;
        auto list_new = h_graph_new.data() + i * num_samples;
        for (uint32_t j = 0; j < SEGMENT_SIZE; j++) {
            for (uint32_t k = 0; k < num_segments; k++) {
                auto neighbor = list[k * SEGMENT_SIZE + j];
                if ((uint32_t) neighbor.id() >= nrow) continue;
                if (!neighbor.is_new()) {
                    if (h_list_sizes_old[i].x < num_samples) {
                        list_old[h_list_sizes_old[i].x++] = neighbor.id();
                    }
                } else if (sample_new) {
                    if (h_list_sizes_new[i].x < num_samples) {
                        list[k * SEGMENT_SIZE + j].mark_old();
                        list_new[h_list_sizes_new[i].x++] = neighbor.id();
                    }
                }
                if (h_list_sizes_old[i].x == num_samples && h_list_sizes_new[i].x == num_samples) { break; }
            }
            if (h_list_sizes_old[i].x == num_samples && h_list_sizes_new[i].x == num_samples) { break; }
        }
    }
}

template<typename Index_t>
void GnndGraph<Index_t>::sample_segment_graph(int segment_start, int segment_end, bool sample_new) {
#pragma omp parallel for
    for (uint32_t i = segment_start; i < segment_end; i++) {
        h_list_sizes_old[i].x = 0;
        h_list_sizes_old[i].y = 0;
        h_list_sizes_new[i].x = 0;
        h_list_sizes_new[i].y = 0;

        auto list = h_graph + i * node_degree;
        auto list_old = h_graph_old.data() + i * num_samples;
        auto list_new = h_graph_new.data() + i * num_samples;
        for (uint32_t j = 0; j < SEGMENT_SIZE; j++) {
            for (uint32_t k = 0; k < num_segments; k++) {
                auto neighbor = list[k * SEGMENT_SIZE + j];
                if ((uint32_t) neighbor.id() >= nrow) continue;
                if (!neighbor.is_new()) {
                    if (h_list_sizes_old[i].x < num_samples) {
                        list_old[h_list_sizes_old[i].x++] = neighbor.id();
                    }
                } else if (sample_new) {
                    if (h_list_sizes_new[i].x < num_samples) {
                        list[k * SEGMENT_SIZE + j].mark_old();
                        list_new[h_list_sizes_new[i].x++] = neighbor.id();
                    }
                }
                if (h_list_sizes_old[i].x == num_samples && h_list_sizes_new[i].x == num_samples) { break; }
            }
            if (h_list_sizes_old[i].x == num_samples && h_list_sizes_new[i].x == num_samples) { break; }
        }
    }
}

template<typename Index_t>
void GnndGraph<Index_t>::update_graph(const InternalID_t<Index_t> *new_neighbors,
                                      const DistData_t *new_dists,
                                      const uint32_t width,
                                      std::atomic<int64_t> &update_counter) {
#pragma omp parallel for
    for (uint32_t i = 0; i < nrow; i++) {
        for (uint32_t j = 0; j < width; j++) {
            auto new_neighbor_id = new_neighbors[i * width + j];
            auto new_dist = new_dists[i * width + j];
            if (new_dist == FLT_MAX) break;
            if ((uint32_t) new_neighbor_id.id() == i) continue;
            uint32_t seg_idx = new_neighbor_id.id() % num_segments;
            auto list = h_graph + i * node_degree + seg_idx * SEGMENT_SIZE;
            auto dist_list = h_dists.data_handle() + i * node_degree + seg_idx * SEGMENT_SIZE;
            uint32_t insert_pos =
                    insert_to_ordered_list(list, dist_list, SEGMENT_SIZE, new_neighbor_id, new_dist);
            if (i % counter_interval == 0 && insert_pos != SEGMENT_SIZE) { update_counter++; }
        }
    }
}

template<typename Index_t>
void GnndGraph<Index_t>::update_segment_graph(const InternalID_t<Index_t> *new_neighbors,
                                              const DistData_t *new_dists,
                                              const uint32_t width,
                                              std::atomic<int64_t> &update_counter,
                                              int segment_start, int segment_end) {
#pragma omp parallel for
    for (uint32_t i = segment_start; i < segment_end; i++) {
        for (uint32_t j = 0; j < width; j++) {
            auto new_neighbor_id = new_neighbors[i * width + j];
            auto new_dist = new_dists[i * width + j];
            if (new_dist == FLT_MAX) break;
            if ((uint32_t) new_neighbor_id.id() == i) continue;
            uint32_t seg_idx = new_neighbor_id.id() % num_segments;
            auto list = h_graph + i * node_degree + seg_idx * SEGMENT_SIZE;
            auto dist_list = h_dists.data_handle() + i * node_degree + seg_idx * SEGMENT_SIZE;
            uint32_t insert_pos =
                    insert_to_ordered_list(list, dist_list, SEGMENT_SIZE, new_neighbor_id, new_dist);
            if (i % counter_interval == 0 && insert_pos != SEGMENT_SIZE) { update_counter++; }
        }
    }
}

template<typename Index_t>
void GnndGraph<Index_t>::clear() {
    for(int i=0;i<segment_num;i++){
        bloom_filter[i].clear();
    }
}

template<typename Index_t>
GnndGraph<Index_t>::~GnndGraph() {
    assert(h_graph == nullptr);
}

template<typename Data_t, typename Index_t>
GNND<Data_t, Index_t>::GNND(raft::resources const &handle, const BuildConfig &build_config)
        : handle(handle),
          build_config_(build_config),
          graph_(build_config.max_dataset_size,
                 roundUp32(build_config.node_degree),
                 roundUp32(build_config.internal_node_degree ? build_config.internal_node_degree
                                                             : build_config.node_degree),
                 NUM_SAMPLES, build_config.segment_num, build_config.segment_length),
          nrow_(build_config.max_dataset_size),
          ndim_(build_config.dataset_dim),
          d_data_{raft::make_device_matrix<__half, uint32_t, raft::row_major>(handle, nrow_, build_config.dataset_dim)},
          l2_norms_{raft::make_device_vector<DistData_t, uint32_t>(handle, nrow_)},
          graph_buffer_{raft::make_device_matrix<ID_t, uint32_t, raft::row_major>(handle, nrow_, DEGREE_ON_DEVICE)},
          dists_buffer_{
                  raft::make_device_matrix<DistData_t, uint32_t, raft::row_major>(handle, nrow_, DEGREE_ON_DEVICE)},
          graph_host_buffer_(nrow_ * DEGREE_ON_DEVICE),
          dists_host_buffer_(nrow_ * DEGREE_ON_DEVICE),
          d_locks_{raft::make_device_vector<int, uint32_t>(handle, nrow_)},
          h_rev_graph_new_(nrow_ * NUM_SAMPLES),
          h_graph_old_(nrow_ * NUM_SAMPLES),
          h_rev_graph_old_(nrow_ * NUM_SAMPLES),
          d_list_sizes_new_{raft::make_device_vector<int2, uint32_t>(handle, nrow_)},
          d_list_sizes_old_{raft::make_device_vector<int2, uint32_t>(handle, nrow_)} {
    static_assert(NUM_SAMPLES <= 32);

    thrust::fill(thrust::device,
                 dists_buffer_.data_handle(),
                 dists_buffer_.data_handle() + dists_buffer_.size(),
                 FLT_MAX);
    thrust::fill(thrust::device,
                 reinterpret_cast<Index_t *>(graph_buffer_.data_handle()),
                 reinterpret_cast<Index_t *>(graph_buffer_.data_handle()) + graph_buffer_.size(),
                 INT_MAX);
    thrust::fill(thrust::device, d_locks_.data_handle(), d_locks_.data_handle() + d_locks_.size(), 0);
};

template<typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::add_reverse_edges(Index_t *graph_ptr,
                                              Index_t *h_rev_graph_ptr,
                                              Index_t *d_rev_graph_ptr,
                                              int2 *list_sizes,
                                              int segment_start,
                                              int segment_length,
                                              cudaStream_t stream) {
    add_rev_edges_kernel<<<segment_length, warp_size(), 0, stream>>>(
            graph_ptr + segment_start * NUM_SAMPLES, d_rev_graph_ptr + segment_start * NUM_SAMPLES,
            NUM_SAMPLES, list_sizes + segment_start);
    raft::copy(h_rev_graph_ptr + segment_start * NUM_SAMPLES, d_rev_graph_ptr + segment_start * NUM_SAMPLES,
               segment_length * NUM_SAMPLES, stream);
}

template<typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::local_join(int segment_start, int segment_length, cudaStream_t stream) {
    thrust::fill(thrust::device.on(stream),
                 dists_buffer_.data_handle() + segment_start * DEGREE_ON_DEVICE,
                 dists_buffer_.data_handle() + (segment_start + segment_length) * DEGREE_ON_DEVICE,
                 FLT_MAX);
    local_join_kernel<<<segment_length, BLOCK_SIZE, 0, stream>>>(
            thrust::raw_pointer_cast(graph_.h_graph_new.data()) + segment_start * NUM_SAMPLES,
            thrust::raw_pointer_cast(h_rev_graph_new_.data()) + segment_start * NUM_SAMPLES,
            d_list_sizes_new_.data_handle() + segment_start,
            thrust::raw_pointer_cast(h_graph_old_.data()) + segment_start * NUM_SAMPLES,
            thrust::raw_pointer_cast(h_rev_graph_old_.data()) + segment_start * NUM_SAMPLES,
            d_list_sizes_old_.data_handle() + segment_start,
            NUM_SAMPLES,
            d_data_.data_handle() + segment_start * ndim_,
            ndim_,
            graph_buffer_.data_handle() + segment_start * DEGREE_ON_DEVICE,
            dists_buffer_.data_handle() + segment_start * DEGREE_ON_DEVICE,
            DEGREE_ON_DEVICE,
            d_locks_.data_handle() + segment_start,
            l2_norms_.data_handle() + segment_start);
}

template<typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::build(std::optional<raft::device_matrix<Data_t>> &data, uint32_t nrow,
                                  Index_t *output_graph,
                                  raft::host_vector_view<uint32_t> h_segment_start_view,
                                  raft::host_vector_view<uint32_t> h_segment_length_view) {
    nrow_ = nrow;
    graph_.h_graph = (InternalID_t<Index_t> *) output_graph;

    for (uint32_t i = 0; i < h_segment_start_view.size(); i++) {
        update_counter_.emplace_back(std::make_unique<std::atomic<int64_t>>(0));
    }

    uint32_t shared_mem_for_preprocess =
            sizeof(Data_t) * ceildiv(build_config_.dataset_dim, static_cast<uint32_t>(warp_size())) * warp_size();
    preprocess_data_kernel<<<nrow_, warp_size(), shared_mem_for_preprocess, raft::resource::get_stream_from_stream_pool(
            handle)>>>(data->data_handle(),
                       d_data_.data_handle(),
                       build_config_.dataset_dim,
                       l2_norms_.data_handle());
    data.reset();

    thrust::fill(thrust::device.on(raft::resource::get_stream_from_stream_pool(handle)),
                 (Index_t *) graph_buffer_.data_handle(),
                 (Index_t *) graph_buffer_.data_handle() + graph_buffer_.size(),
                 get_max_value<int>());

    graph_.clear();
    graph_.init_random_graph(h_segment_start_view, h_segment_length_view);
    graph_.sample_graph(true);

    cudaDeviceSynchronize();

    auto update_and_sample = [&](bool update_graph, int segment_id, int segment_start, int segment_length) {
        if (*update_counter_[segment_id] == -1) {
            return;
        }
        if (update_graph) {
            *update_counter_[segment_id] = 0;
            graph_.update_segment_graph(thrust::raw_pointer_cast(graph_host_buffer_.data()),
                                        thrust::raw_pointer_cast(dists_host_buffer_.data()),
                                        DEGREE_ON_DEVICE,
                                        *update_counter_[segment_id], segment_start, segment_length);
            if (*update_counter_[segment_id] < build_config_.termination_threshold * nrow_ *
                                               build_config_.dataset_dim / counter_interval) {
                *update_counter_[segment_id] = -1;
            }
        }
        graph_.sample_segment_graph(segment_start, segment_start + segment_length, false);
    };

    uint32_t it;
    for (it = 0; it < build_config_.max_iterations; it++) {
        raft::copy(d_list_sizes_new_.data_handle(),
                   thrust::raw_pointer_cast(graph_.h_list_sizes_new.data()),
                   nrow_,
                   raft::resource::get_stream_from_stream_pool(handle));
        raft::copy(thrust::raw_pointer_cast(h_graph_old_.data()),
                   thrust::raw_pointer_cast(graph_.h_graph_old.data()),
                   nrow_ * NUM_SAMPLES,
                   raft::resource::get_stream_from_stream_pool(handle));
        raft::copy(d_list_sizes_old_.data_handle(),
                   thrust::raw_pointer_cast(graph_.h_list_sizes_old.data()),
                   nrow_,
                   raft::resource::get_stream_from_stream_pool(handle));
        cudaDeviceSynchronize();

        std::vector<std::thread> threads;

        for (uint32_t segment_id = 0; segment_id < h_segment_start_view.size(); segment_id++) {
            if(*update_counter_[segment_id] == -1){
                continue;
            }
            cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

            threads.emplace_back([&, segment_id, stream](){
                cudaSetDevice(1);
                std::thread update_and_sample_thread(update_and_sample, it, segment_id,
                                                     h_segment_start_view(segment_id),
                                                     h_segment_length_view(segment_id));

                add_reverse_edges(thrust::raw_pointer_cast(graph_.h_graph_new.data()),
                                  thrust::raw_pointer_cast(h_rev_graph_new_.data()),
                                  (Index_t *) dists_buffer_.data_handle(),
                                  d_list_sizes_new_.data_handle(),
                                  h_segment_start_view(segment_id),
                                  h_segment_length_view(segment_id),
                                  stream);

                add_reverse_edges(thrust::raw_pointer_cast(h_graph_old_.data()),
                                  thrust::raw_pointer_cast(h_rev_graph_old_.data()),
                                  (Index_t *) dists_buffer_.data_handle(),
                                  d_list_sizes_old_.data_handle(),
                                  h_segment_start_view(segment_id),
                                  h_segment_length_view(segment_id),
                                  stream);

                local_join(h_segment_start_view(segment_id), h_segment_length_view(segment_id), stream);

                update_and_sample_thread.join();
                cudaStreamSynchronize(stream);
            });
        }

        for (auto &thread : threads) {
            thread.join();
        }
        bool flag = true;
        for(auto &counter : update_counter_) {
            if (*counter != -1) {
                flag = false;
                break;
            }
        }
        if(flag) { break;}

        raft::copy(thrust::raw_pointer_cast(graph_host_buffer_.data()),
                   graph_buffer_.data_handle(),
                   nrow_ * DEGREE_ON_DEVICE,
                   raft::resource::get_stream_from_stream_pool(handle));
        raft::copy(thrust::raw_pointer_cast(dists_host_buffer_.data()),
                   dists_buffer_.data_handle(),
                   nrow_ * DEGREE_ON_DEVICE,
                   raft::resource::get_stream_from_stream_pool(handle));
        cudaDeviceSynchronize();

        graph_.sample_graph_new(thrust::raw_pointer_cast(graph_host_buffer_.data()), DEGREE_ON_DEVICE,
                                h_segment_start_view, h_segment_length_view);
//        graph_.sample_graph_new(thrust::raw_pointer_cast(graph_host_buffer_.data()), DEGREE_ON_DEVICE);
    }

    graph_.update_graph(thrust::raw_pointer_cast(graph_host_buffer_.data()),
                        thrust::raw_pointer_cast(dists_host_buffer_.data()),
                        DEGREE_ON_DEVICE,
                        *update_counter_[0]);

    static_assert(sizeof(decltype(*(graph_.h_dists.data_handle()))) >= sizeof(Index_t));
    Index_t *graph_shrink_buffer = (Index_t *) graph_.h_dists.data_handle();

    uint32_t segment_num = h_segment_start_view.extent(0);
#pragma omp parallel for
    for (uint32_t i = 0; i < nrow_; i++) {
        uint32_t segment_id = 0;
        for (; segment_id < segment_num; segment_id++) {
            uint32_t segment_end = h_segment_start_view(segment_id) + h_segment_length_view(segment_id);
            if (i < segment_end) break;
        }
        for (uint32_t j = 0; j < build_config_.node_degree; j++) {
            uint32_t idx = i * graph_.node_degree + j;
            uint32_t id = graph_.h_graph[idx].id();
            if (id < h_segment_length_view(segment_id)) {
                graph_shrink_buffer[i * build_config_.node_degree + j] = id;
            } else {
                graph_shrink_buffer[i * build_config_.node_degree + j] = xorshift32(idx) % h_segment_length_view(segment_id);
            }
        }
    }
    graph_.h_graph = nullptr;

#pragma omp parallel for
    for (uint32_t i = 0; i < nrow_; i++) {
        for (uint32_t j = 0; j < build_config_.node_degree; j++) {
            output_graph[i * build_config_.node_degree + j] =
                    graph_shrink_buffer[i * build_config_.node_degree + j];
        }
    }
}

template<typename Data_t>
raft::host_matrix<int>
build_nnd(raft::resources const &handle, NNDescentParameter &param,
          std::optional<raft::device_matrix<Data_t>> &d_data,
          raft::host_vector_view<uint32_t> h_segment_start_view,
          raft::host_vector_view<uint32_t> h_segment_length_view,
          std::string &result_file) {
    uint32_t intermediate_graph_degree = param.intermediate_graph_degree;
    uint32_t graph_degree = param.graph_degree;
    uint32_t num = d_data->extent(0);
    uint32_t dim = d_data->extent(1);

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float milliseconds = 0.0f;
    float build_time = 0.0f;

    if (intermediate_graph_degree < graph_degree) {
        graph_degree = intermediate_graph_degree;
    }

    uint32_t extended_graph_degree =
            roundUp32(static_cast<uint32_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
    uint32_t extended_intermediate_graph_degree = roundUp32(
            static_cast<uint32_t>(intermediate_graph_degree * (intermediate_graph_degree <= 32 ? 1.0 : 1.3)));

    auto int_graph = raft::make_host_matrix<int, uint32_t, raft::row_major>(num, extended_graph_degree);
    auto nnd_graph = raft::make_host_matrix<int>(num, graph_degree);

    BuildConfig build_config{.max_dataset_size      = num,
            .dataset_dim           = dim,
            .node_degree           = extended_graph_degree,
            .internal_node_degree  = extended_intermediate_graph_degree,
            .max_iterations        = param.max_iterations,
            .termination_threshold = param.termination_threshold,
            .segment_num           = h_segment_start_view.extent(0),
            .segment_length        = h_segment_length_view.data_handle(),};

    GNND<Data_t, int> nnd(handle, build_config);

    cudaEventRecord(start_time);
    nnd.build(d_data, num, int_graph.data_handle(),
              h_segment_start_view, h_segment_length_view);

#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < graph_degree; j++) {
            nnd_graph(i, j) = int_graph(i, j);
        }
    }
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;

    std::cout << "knn time: " << build_time << " s" << std::endl;
    std::ofstream result_out(result_file, std::ios::app);
    result_out<<build_time<<",";
    result_out.close();

    return nnd_graph;
}

raft::host_matrix<int>
build_knn(raft::resources const &handle, uint32_t knn_degree,
          raft::host_vector_view<uint32_t> h_segment_start_view,
          raft::host_vector_view<uint32_t> h_segment_length_view,
          raft::device_matrix_view<float> d_reorder_data_view) {
    using namespace cuvs::neighbors;
    uint32_t segment = h_segment_start_view.extent(0);
    uint32_t dim = d_reorder_data_view.extent(1);
    uint32_t num = d_reorder_data_view.extent(0);

    auto knn_graph = raft::make_host_matrix<int>(num, knn_degree);

#pragma omp parallel for
    for (uint32_t i = 0; i < segment; i++) {
        uint32_t length = h_segment_length_view(i);
        uint32_t start = h_segment_start_view(i);

        auto sub_dataset = raft::make_device_matrix<float, int64_t>(handle, length, dim);
        raft::copy(sub_dataset.data_handle(), d_reorder_data_view.data_handle() + start * dim, length * dim,
                   raft::resource::get_cuda_stream(handle));

        raft::device_matrix_view<const float, int64_t> data = sub_dataset.view();

        cagra::index_params index_params;
        index_params.graph_degree = knn_degree;
        index_params.intermediate_graph_degree = knn_degree;

        auto knn_build_params = index_params.graph_build_params;
        knn_build_params = cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);

        auto nn_descent_params =
                std::get<cagra::graph_build_params::nn_descent_params>(knn_build_params);
        if (nn_descent_params.graph_degree != index_params.intermediate_graph_degree) {
            nn_descent_params = cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
        }

        auto index = nn_descent::build(handle, nn_descent_params, data);

        raft::copy(knn_graph.data_handle() + start * index_params.intermediate_graph_degree,
                   reinterpret_cast<int *>(index.graph().data_handle()),
                   length * index_params.intermediate_graph_degree,
                   raft::resource::get_cuda_stream(handle));
    }
    return knn_graph;
}

template<typename Data_t, typename Index_t>
raft::host_matrix<int, Index_t>
build_knn_for_large(raft::resources const &handle, uint32_t knn_degree,
                    raft::host_vector_view<uint32_t> h_segment_start_view,
                    raft::host_vector_view<uint32_t> h_segment_length_view,
                    raft::host_matrix_view<Data_t, Index_t> h_reorder_data_view, std::string &result_file) {
    using namespace cuvs::neighbors;
    uint32_t segment = h_segment_start_view.extent(0);
    uint32_t dim = h_reorder_data_view.extent(1);
    uint32_t num = h_reorder_data_view.extent(0);

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0.0f;
    float build_time = 0.0f;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    auto knn_graph = raft::make_host_matrix<int, Index_t>(num, knn_degree);

    for (uint32_t i = 0; i < segment; i++) {
        uint32_t length = h_segment_length_view(i);
        uint32_t start = h_segment_start_view(i);

        Index_t data_start = static_cast<Index_t>(start) * dim;
        Index_t data_length = static_cast<Index_t>(length) * dim;
        Index_t knn_start = static_cast<Index_t>(start) * knn_degree;
        Index_t knn_length = static_cast<Index_t>(length) * knn_degree;

        auto sub_dataset = raft::make_host_matrix<Data_t, Index_t>(length, dim);
        raft::copy(sub_dataset.data_handle(), h_reorder_data_view.data_handle() + data_start,
                   data_length, raft::resource::get_cuda_stream(handle));

        raft::host_matrix_view<const Data_t, Index_t> data = sub_dataset.view();

        cagra::index_params index_params;
        index_params.graph_degree = knn_degree;
        index_params.intermediate_graph_degree = knn_degree;

        auto knn_build_params = index_params.graph_build_params;
        knn_build_params = cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);

        auto nn_descent_params =
                std::get<cagra::graph_build_params::nn_descent_params>(knn_build_params);
        if (nn_descent_params.graph_degree != index_params.intermediate_graph_degree) {
            nn_descent_params = cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
        }

        cudaEventRecord(start_time);
        auto index = nn_descent::build(handle, nn_descent_params, data);
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        build_time += milliseconds / 1000.0f;

        raft::copy(knn_graph.data_handle() + knn_start,
                   reinterpret_cast<int *>(index.graph().data_handle()),
                   knn_length,
                   raft::resource::get_cuda_stream(handle));
    }
    std::cout << "knn time: " << build_time << " s" << std::endl;

    std::ofstream result_out(result_file, std::ios::app);
    result_out<<build_time<<",";
    result_out.close();

    return knn_graph;
}