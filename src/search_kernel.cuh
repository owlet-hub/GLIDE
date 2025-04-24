
#include "hashmap.cuh"
#include "bitonic.cuh"
#include "distance.cuh"
#include "utils.cuh"

#pragma once

template<uint32_t MAX_CANDIDATE>
__device__ void
sort_for_candidate(float *candidate_dist_buffer, uint32_t *candidate_id_buffer, uint32_t graph_degree,
                   uint32_t beam, uint32_t multi_warps = 0) {
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;
    if (multi_warps == 0) {
        if (warp_id > 0) {
            return;
        }
        constexpr uint32_t N = (MAX_CANDIDATE + warp_size() - 1) / warp_size();
        float key[N];
        uint32_t val[N];
        for (uint32_t i = 0; i < N; i++) {
            uint32_t j = lane_id + (warpSize * i);

            key[i] = (j < graph_degree) ? candidate_dist_buffer[j] : get_max_value<float>();
            val[i] = (j < graph_degree) ? candidate_id_buffer[j] : get_max_value<uint32_t>();
        }

        bitonic::warp_sort<float, uint32_t, N>(key, val);

        for (uint32_t i = 0; i < N; i++) {
            uint32_t j = (N * lane_id) + i;
            if (j < graph_degree && j < beam) {
                candidate_dist_buffer[swizzling(j)] = key[i];
                candidate_id_buffer[swizzling(j)] = val[i];
            }
        }
    } else {
        constexpr uint32_t max_candidates_per_warp = (MAX_CANDIDATE + 1) / 2;
        constexpr uint32_t N = (max_candidates_per_warp + warp_size() - 1) / warp_size();
        float key[N];
        uint32_t val[N];
        if (warp_id < 2) {
            for (uint32_t i = 0; i < N; i++) {
                uint32_t jl = lane_id + (warpSize * i);
                uint32_t j = jl + (max_candidates_per_warp * warp_id);
                key[i] = (j < graph_degree) ? candidate_dist_buffer[j] : get_max_value<float>();
                val[i] = (j < graph_degree) ? candidate_id_buffer[j] : get_max_value<uint32_t>();
            }

            bitonic::warp_sort<float, uint32_t, N>(key, val);

            for (uint32_t i = 0; i < N; i++) {
                uint32_t jl = (N * lane_id) + i;
                uint32_t j = jl + (max_candidates_per_warp * warp_id);
                if (j < graph_degree && jl < beam) {
                    candidate_dist_buffer[swizzling(j)] = key[i];
                    candidate_id_buffer[swizzling(j)] = val[i];
                }
            }
        }
        __syncthreads();

        uint32_t num_warps_used = (beam + max_candidates_per_warp - 1) / max_candidates_per_warp;
        if (warp_id < num_warps_used) {
            for (uint32_t i = 0; i < N; i++) {
                uint32_t jl = (N * lane_id) + i;
                uint32_t kl = max_candidates_per_warp - 1 - jl;
                uint32_t j = jl + (max_candidates_per_warp * warp_id);
                uint32_t k = MAX_CANDIDATE - 1 - j;
                if (j >= graph_degree || k >= graph_degree || kl >= beam) continue;
                float temp_key = candidate_dist_buffer[swizzling(k)];
                if (key[i] == temp_key) continue;
                if ((warp_id == 0) == (key[i] > temp_key)) {
                    key[i] = temp_key;
                    val[i] = candidate_id_buffer[swizzling(k)];
                }
            }
        }
        if (num_warps_used > 1) { __syncthreads(); }
        if (warp_id < num_warps_used) {
            bitonic::warp_merge<float, uint32_t, N>(key, val, 32);

            for (uint32_t i = 0; i < N; i++) {
                uint32_t jl = (N * lane_id) + i;
                uint32_t j = jl + (max_candidates_per_warp * warp_id);
                if (j < graph_degree && j < beam) {
                    candidate_dist_buffer[swizzling(j)] = key[i];
                    candidate_id_buffer[swizzling(j)] = val[i];
                }
            }
        }
        if (num_warps_used > 1) { __syncthreads(); }
    }
}

template<uint32_t MAX_BEAM>
__device__ void
sort_for_beam(float *beam_dist_buffer, uint32_t *beam_id_buffer, uint32_t beam, float *candidate_dist_buffer,
              uint32_t *candidate_id_buffer, uint32_t graph_degree, uint32_t *workspace_buffer,
              const bool first, uint32_t multi_warps = 0) {
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;
    if (multi_warps == 0) {
        if (warp_id > 0) {
            return;
        }

        constexpr uint32_t N = (MAX_BEAM + warp_size() - 1) / warp_size();
        float key[N];
        uint32_t val[N];
        if (first) {
            for (uint32_t i = lane_id; i < beam; i += warpSize) {
                beam_dist_buffer[swizzling(i)] = (i < graph_degree) ?
                                                 candidate_dist_buffer[swizzling(i)] : get_max_value<float>();
                beam_id_buffer[swizzling(i)] = (i < graph_degree) ?
                                               candidate_id_buffer[swizzling(i)] : get_max_value<uint32_t>();
            }
        } else {
            for (uint32_t i = 0; i < N; i++) {
                uint32_t j = (N * lane_id) + i;

                key[i] = (j < beam) ? beam_dist_buffer[swizzling(j)] : get_max_value<float>();
                val[i] = (j < beam) ? beam_id_buffer[swizzling(j)] : get_max_value<uint32_t>();
            }

            for (uint32_t i = 0; i < N; i++) {
                uint32_t j = (N * lane_id) + i;
                uint32_t k = MAX_BEAM - 1 - j;
                if (k >= beam || k >= graph_degree) continue;

                float candidate_key = candidate_dist_buffer[swizzling(k)];
                if (key[i] > candidate_key) {
                    key[i] = candidate_key;
                    val[i] = candidate_id_buffer[swizzling(k)];
                }
            }

            bitonic::warp_merge<float, uint32_t, N>(key, val, 32);

            for (uint32_t i = 0; i < N; i++) {
                uint32_t j = (N * lane_id) + i;
                if (j < beam) {
                    beam_dist_buffer[swizzling(j)] = key[i];
                    beam_id_buffer[swizzling(j)] = val[i];
                }
            }
        }
    } else {
        constexpr uint32_t max_itopk_per_warp = (MAX_BEAM + 1) / 2;
        constexpr uint32_t N = (max_itopk_per_warp + warp_size() - 1) / warp_size();
        float key[N];
        uint32_t val[N];
        if (first) {
            if (warp_id < 2) {
                for (uint32_t i = threadIdx.x; i < beam; i += 2 * warpSize) {
                    beam_dist_buffer[swizzling(i)] = (i < graph_degree) ?
                                                     candidate_dist_buffer[swizzling(i)] : get_max_value<float>();
                    beam_id_buffer[swizzling(i)] = (i < graph_degree) ?
                                                   candidate_id_buffer[swizzling(i)] : get_max_value<uint32_t>();
                }
            }
            __syncthreads();
        } else {
            const uint32_t num_itopk_div2 = beam / 2;
            if (threadIdx.x < 3) {
                workspace_buffer[threadIdx.x] = num_itopk_div2;
            }
            __syncthreads();

            for (uint32_t k = threadIdx.x; k < min(graph_degree, beam); k += blockDim.x) {
                const uint32_t j = beam - 1 - k;
                const float itopk_key = beam_dist_buffer[swizzling(j)];
                const float candidate_key = candidate_dist_buffer[swizzling(k)];
                if (itopk_key > candidate_key) {
                    beam_dist_buffer[swizzling(j)] = candidate_key;
                    beam_id_buffer[swizzling(j)] = candidate_id_buffer[swizzling(k)];
                    if (j < num_itopk_div2) {
                        atomicMin(workspace_buffer + 2, j);
                    } else {
                        atomicMin(workspace_buffer + 1, j - num_itopk_div2);
                    }
                }
            }
            __syncthreads();

            for (uint32_t j = threadIdx.x; j < num_itopk_div2; j += blockDim.x) {
                uint32_t k = j + num_itopk_div2;
                float key_0 = beam_dist_buffer[swizzling(j)];
                float key_1 = beam_dist_buffer[swizzling(k)];
                if (key_0 > key_1) {
                    beam_dist_buffer[swizzling(j)] = key_1;
                    beam_dist_buffer[swizzling(k)] = key_0;
                    uint32_t val_0 = beam_id_buffer[swizzling(j)];
                    uint32_t val_1 = beam_id_buffer[swizzling(k)];
                    beam_id_buffer[swizzling(j)] = val_1;
                    beam_id_buffer[swizzling(k)] = val_0;
                    atomicMin(workspace_buffer + 0, j);
                }
            }
            if (threadIdx.x == blockDim.x - 1) {
                if (workspace_buffer[2] < num_itopk_div2) { workspace_buffer[1] = workspace_buffer[2]; }
            }
            __syncthreads();

            if (warp_id < 2) {
                const uint32_t turning_point = workspace_buffer[warp_id];
                for (uint32_t i = 0; i < N; i++) {
                    uint32_t k = beam;
                    uint32_t j = (N * lane_id) + i;
                    if (j < turning_point) {
                        k = j + (num_itopk_div2 * warp_id);
                    } else if (j >= (MAX_BEAM / 2 - num_itopk_div2)) {
                        j -= (MAX_BEAM / 2 - num_itopk_div2);
                        if ((turning_point <= j) && (j < num_itopk_div2)) { k = j + (num_itopk_div2 * warp_id); }
                    }

                    key[i] = (k < beam) ? beam_dist_buffer[swizzling(k)] : get_max_value<float>();
                    val[i] = (k < beam) ? beam_id_buffer[swizzling(k)] : get_max_value<uint32_t>();
                }

                bitonic::warp_merge<float, uint32_t, N>(key, val, 32);
                for (uint32_t i = 0; i < N; i++) {
                    uint32_t j = (N * lane_id) + i;
                    if (j < num_itopk_div2) {
                        uint32_t k = j + (num_itopk_div2 * warp_id);
                        beam_dist_buffer[swizzling(k)] = key[i];
                        beam_id_buffer[swizzling(k)] = val[i];
                    }
                }
            }
        }
    }
}

template<uint32_t MAX_BEAM, uint32_t MAX_CANDIDATE>
__device__ void
update_result_buffer(float *beam_dist_buffer, uint32_t *beam_id_buffer, uint32_t beam, float *candidate_dist_buffer,
                     uint32_t *candidate_id_buffer, uint32_t graph_degree, uint32_t *workspace_buffer,
                     const bool first, const uint32_t multi_warps_1, const uint32_t multi_warps_2) {
    sort_for_candidate<MAX_CANDIDATE>(candidate_dist_buffer, candidate_id_buffer,
                                      graph_degree, beam, multi_warps_1);

    sort_for_beam<MAX_BEAM>(beam_dist_buffer, beam_id_buffer, beam, candidate_dist_buffer,
                            candidate_id_buffer, graph_degree, workspace_buffer, first, multi_warps_2);
}

template<uint32_t MAX_CENTROID>
__device__ void
result_init(uint32_t *centroids_id_buffer, float *centroids_dist_buffer, uint32_t *candidate_id_buffer,
            float *candidate_dist_buffer, float *query_buffer, const float *data, const uint32_t *graph,
            const uint32_t *start_points, const float *centroids, uint32_t dim, uint32_t graph_degree,
            uint32_t centroid_num, uint32_t *visited_hashmap, uint32_t hash_bit, Metric metric) {
    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_num = blockDim.x / warpSize;

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(query_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroids_id_buffer[i] = i;
            centroids_dist_buffer[i] = dist;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroids_id_buffer, centroids_dist_buffer);
    __syncthreads();

    for (uint32_t neighbor_id = threadIdx.x; neighbor_id < graph_degree; neighbor_id += blockDim.x) {
        uint32_t node_id = start_points[centroids_id_buffer[0]];

        uint32_t index = graph[node_id * graph_degree + neighbor_id];

        uint32_t flag = (index != get_max_value<uint32_t>()) ? hashmap::insert(visited_hashmap, hash_bit, index) : 0;
        candidate_id_buffer[neighbor_id] = (flag == 1) ? index : get_max_value<uint32_t>();
    }
    __syncthreads();

    for (uint32_t neighbor_id = warp_id; neighbor_id < graph_degree; neighbor_id += warp_num) {
        uint32_t index = candidate_id_buffer[neighbor_id];

        float dist = (index != get_max_value<uint32_t>()) ?
                     compute_similarity_warp(query_buffer, data + index * dim, dim, metric) : get_max_value<float>();
        if (lane_id == 0) {
            candidate_dist_buffer[neighbor_id] = dist;
        }
    }
}

__device__ void
select_current_point(uint32_t *terminate_flag, uint32_t *select_buffer, uint32_t *beam_id_buffer, uint32_t beam) {
    if (threadIdx.x == 0) {
        select_buffer[0] = get_max_value<uint32_t>();
    }

    uint32_t unexplored_num = 0;
    for (uint32_t j = threadIdx.x; j < roundUp32(beam); j += warpSize) {
        uint32_t jj = swizzling(j);
        uint32_t index;
        int unexplored = 0;
        if (j < beam) {
            index = beam_id_buffer[jj];
            if ((index & index_mask::explored) == 0) {
                unexplored = 1;
            }
        }
        const uint32_t ballot_mask = __ballot_sync(warp_full_mask(), unexplored);
        if (unexplored) {
            const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + unexplored_num;
            if (i == 0) {
                select_buffer[i] = beam_id_buffer[jj];
                beam_id_buffer[jj] |= index_mask::explored;
            }
        }
        unexplored_num += __popc(ballot_mask);
        if (unexplored_num >= 1) {
            break;
        }
    }
    if (threadIdx.x == 0 && unexplored_num == 0) {
        atomicExch(terminate_flag, 1);
    }
}

__device__ inline void
hashmap_restore(uint32_t *hashmap, size_t hash_bit, uint32_t *beam_id_buffer, uint32_t beam, uint32_t first_tid = 0) {
    if (threadIdx.x < first_tid) return;
    for (unsigned i = threadIdx.x - first_tid; i < beam; i += blockDim.x - first_tid) {
        auto key = beam_id_buffer[i] & index_mask::clean;
        hashmap::insert(hashmap, hash_bit, key);
    }
}

__device__ void
compute_distance(uint32_t *candidate_id_buffer, float *candidate_dist_buffer, float *query_buffer,
                 const uint32_t *graph, uint32_t graph_degree, uint32_t *visited_hashmap, uint32_t hash_bit,
                 uint32_t *select_buffer, uint32_t dim, const float *data, Metric metric) {
    for (uint32_t i = threadIdx.x; i < graph_degree; i += blockDim.x) {
        uint32_t neighbor_id = get_max_value<uint32_t>();
        const auto current_point_id = select_buffer[0];
        if (current_point_id != get_max_value<uint32_t>()) {
            neighbor_id = graph[i + (graph_degree * current_point_id)];
        }
        uint32_t flag = (neighbor_id != get_max_value<uint32_t>()) ?
                        hashmap::insert(visited_hashmap, hash_bit, neighbor_id) : 0;
        candidate_id_buffer[i] = (flag == 1) ? neighbor_id : get_max_value<uint32_t>();
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    for (uint32_t i = warp_id; i < graph_degree; i += warp_num) {
        uint32_t index = candidate_id_buffer[i];

        float dist = (index != get_max_value<uint32_t>())
                     ? compute_similarity_warp(query_buffer, data + index * dim, dim, metric) : get_max_value<float>();
        if (lane_id == 0) {
            candidate_dist_buffer[i] = dist;
        }
    }
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM, uint32_t MAX_CENTROID>
__global__ void
search_kernel(uint32_t *result_ids, float *result_dists, uint32_t top_k, const float *data, const float *query,
              const uint32_t *graph, uint32_t graph_degree, uint32_t beam, uint32_t max_iterations,
              uint32_t min_iterations, uint32_t hash_bit, uint32_t hash_reset_interval, const uint32_t *start_points,
              uint32_t dim, uint32_t centroid_num, const float *centroids, Metric metric) {
    uint32_t query_id = blockIdx.x;

    extern __shared__ uint32_t shared_mem[];

    uint32_t result_buffer_size = beam + graph_degree;
    result_buffer_size = roundUp32(result_buffer_size);

    auto *query_buffer = reinterpret_cast<float *>(shared_mem);
    auto *result_id_buffer = reinterpret_cast<uint32_t *>(query_buffer + dim);
    auto *result_dist_buffer = reinterpret_cast<float *>(result_id_buffer + result_buffer_size);
    auto *select_buffer = reinterpret_cast<uint32_t *>(result_dist_buffer + result_buffer_size);
    auto *workspace_buffer = reinterpret_cast<uint32_t *>(select_buffer + 1);
    auto *terminate_flag = reinterpret_cast<uint32_t *>(workspace_buffer + 3);
    auto *visited_hashmap = reinterpret_cast<uint32_t *>(terminate_flag + 1);

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        query_buffer[i] = query[query_id * dim + i];
    }
    if (threadIdx.x == 0) {
        terminate_flag[0] = 0;
        workspace_buffer[0] = ~0u;
    }
    hashmap::init(visited_hashmap, hash_bit, 0);
    __syncthreads();

    result_init<MAX_CENTROID>(result_id_buffer, result_dist_buffer, result_id_buffer + beam, result_dist_buffer + beam,
                              query_buffer, data, graph, start_points, centroids, dim, graph_degree, centroid_num,
                              visited_hashmap, hash_bit, metric);
    __syncthreads();

    uint32_t iter = 0;
    const uint32_t multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATE > 128)) ? 1 : 0;
    const uint32_t multi_warps_2 = ((blockDim.x >= 64) && (MAX_BEAM > 256)) ? 1 : 0;
    while (true) {
        update_result_buffer<MAX_BEAM, MAX_CANDIDATE>(result_dist_buffer, result_id_buffer, beam,
                                                      result_dist_buffer + beam, result_id_buffer + beam, graph_degree,
                                                      workspace_buffer, (iter == 0), multi_warps_1, multi_warps_2);

        if ((iter + 1) % hash_reset_interval == 0) {
            unsigned hash_start_tid;
            if (blockDim.x == 32) {
                hash_start_tid = 0;
            } else if (blockDim.x == 64) {
                if (multi_warps_1 || multi_warps_2) {
                    hash_start_tid = 0;
                } else {
                    hash_start_tid = 32;
                }
            } else {
                if (multi_warps_1 || multi_warps_2) {
                    hash_start_tid = 64;
                } else {
                    hash_start_tid = 32;
                }
            }
            hashmap::init(visited_hashmap, hash_bit, hash_start_tid);
        }
        __syncthreads();

        if (iter + 1 == max_iterations) {
            break;
        }

        if (threadIdx.x < 32) {
            select_current_point(terminate_flag, select_buffer, result_id_buffer, beam);
        }
        if ((iter + 1) % hash_reset_interval == 0) {
            const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
            hashmap_restore(visited_hashmap, hash_bit, result_id_buffer, beam, first_tid);
        }
        __syncthreads();

        if (*terminate_flag && iter >= min_iterations) {
            break;
        }

        compute_distance(result_id_buffer + beam, result_dist_buffer + beam, query_buffer,
                         graph, graph_degree, visited_hashmap, hash_bit, select_buffer,
                         dim, data, metric);
        __syncthreads();

        iter++;
    }

    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
        uint32_t result_index = i + (top_k * query_id);
        uint32_t index = swizzling(i);
        result_dists[result_index] = result_dist_buffer[index];
        result_ids[result_index] = result_id_buffer[index] & index_mask::clean;
    }
}

__device__ void
result_init_for_large(uint32_t *candidate_id_buffer, float *candidate_dist_buffer, uint8_t *query_buffer,
                      const uint8_t *data, const uint32_t *graph, uint32_t start_points, uint32_t dim,
                      uint32_t graph_degree, uint32_t *visited_hashmap, uint32_t hash_bit, Metric metric) {
    for (uint32_t i = threadIdx.x; i < graph_degree; i += blockDim.x) {
        uint32_t index = graph[start_points * graph_degree + i];
        uint32_t flag = (index != get_max_value<uint32_t>()) ? hashmap::insert(visited_hashmap, hash_bit, index) : 0;
        candidate_id_buffer[i] = (flag == 1) ? index : get_max_value<uint32_t>();
    }
    __syncthreads();

    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;
    for (uint32_t i = warp_id; i < graph_degree; i += warp_num) {
        uint32_t index = candidate_id_buffer[i];
        float dist = (index != get_max_value<uint32_t>()) ?
                     compute_similarity_warp(query_buffer, data + index * dim, dim, metric) : get_max_value<float>();
        if (lane_id == 0) {
            candidate_dist_buffer[i] = dist;
        }
    }
}

__device__ void
compute_distance_for_large(uint32_t *candidate_id_buffer, float *candidate_dist_buffer, uint8_t *query_buffer,
                           const uint32_t *graph, uint32_t graph_degree, uint32_t *visited_hashmap, uint32_t hash_bit,
                           uint32_t *select_buffer, uint32_t dim, const uint8_t *data, Metric metric) {
    for (uint32_t i = threadIdx.x; i < graph_degree; i += blockDim.x) {
        uint32_t neighbor_id = get_max_value<uint32_t>();
        const auto current_point_id = select_buffer[0];
        if (current_point_id != get_max_value<uint32_t>()) {
            neighbor_id = graph[i + (graph_degree * current_point_id)];
        }
        uint32_t flag = (neighbor_id != get_max_value<uint32_t>()) ?
                        hashmap::insert(visited_hashmap, hash_bit, neighbor_id) : 0;
        candidate_id_buffer[i] = (flag == 1) ? neighbor_id : get_max_value<uint32_t>();
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    for (uint32_t i = warp_id; i < graph_degree; i += warp_num) {
        uint32_t index = candidate_id_buffer[i];

        float dist = (index != get_max_value<uint32_t>()) ?
                     compute_similarity_warp(query_buffer, data + index * dim, dim, metric) : get_max_value<float>();
        if (lane_id == 0) {
            candidate_dist_buffer[i] = dist;
        }
    }
}

template<uint32_t MAX_CENTROID>
__global__ void
select_segment_kernel(uint32_t *result_ids, const uint8_t *query, const float *centroids, uint32_t centroid_num,
                      uint32_t dim, uint32_t min_segment_num, float boundary_factor, Metric metric) {
    uint32_t query_id = blockIdx.x;
    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_num = blockDim.x / warpSize;

    extern __shared__ uint32_t shared_mem[];

    auto *query_buffer = reinterpret_cast<uint8_t *>(shared_mem);
    auto *centroids_id_buffer = reinterpret_cast<uint32_t *>(query_buffer + dim);
    auto *centroids_dist_buffer = reinterpret_cast<float *>(centroids_id_buffer + centroid_num);

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        query_buffer[i] = query[query_id * dim + i];
    }
    for (uint32_t i = threadIdx.x; i < centroid_num; i += blockDim.x) {
        result_ids[i] = get_max_value<uint32_t>();
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(query_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroids_id_buffer[i] = i;
            centroids_dist_buffer[i] = dist;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroids_id_buffer, centroids_dist_buffer);
    __syncthreads();

    if (warp_id == 0) {
        uint32_t allow_num = 0;
        for (uint32_t i = lane_id; i < roundUp32(centroid_num); i += warpSize) {

            uint32_t allow = (i < centroid_num &&
                              (centroids_dist_buffer[i] / centroids_dist_buffer[0] <= boundary_factor)) ? 1 : 0;
            const uint32_t ballot_mask = __ballot_sync(0xffffffff, allow);

            if (allow) {
                uint32_t result_index = (centroid_num * query_id) + i;
                result_ids[result_index] = centroids_id_buffer[i];
            }
            allow_num += __popc(ballot_mask);
        }

        if (allow_num < min_segment_num) {
            uint32_t need = min_segment_num - allow_num;
            for (uint32_t i = lane_id; i < need; i += warpSize) {
                uint32_t index = allow_num + i;
                uint32_t result_index = (centroid_num * query_id) + allow_num + i;
                result_ids[result_index] = centroids_id_buffer[index];
            }
        }

    }
    __syncthreads();
}

template<unsigned MAX_TOPK>
__device__ void
result_merge(float *final_result_dists, uint32_t *final_result_ids, float *result_dists, uint32_t *result_ids,
             uint32_t *map, uint32_t top_k) {
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;

    if (warp_id > 0) {
        return;
    }

    constexpr uint32_t N = (MAX_TOPK + warp_size() - 1) / warp_size();
    float key[N];
    uint32_t val[N];
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = (N * lane_id) + i;

        key[i] = (j < top_k) ? final_result_dists[j] : get_max_value<float>();
        val[i] = (j < top_k) ? final_result_ids[j] : get_max_value<uint32_t>();
    }

    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = (N * lane_id) + i;
        uint32_t k = MAX_TOPK - 1 - j;
        if (k >= top_k) continue;

        float result_key = result_dists[swizzling(k)];
        if (key[i] > result_key) {
            key[i] = result_key;
            val[i] = map[result_ids[swizzling(k)] & index_mask::clean];
        }
    }

    bitonic::warp_merge<float, uint32_t, N>(key, val, 32);

    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = (N * lane_id) + i;
        if (j < top_k) {
            final_result_dists[j] = key[i];
            final_result_ids[j] = val[i];
        }
    }
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM, uint32_t MAX_TOPK>
__global__ void
search_on_sub_kernel(uint32_t *final_result_ids, float *final_result_dists, uint32_t top_k,
                     const uint8_t *data, const uint8_t *query, uint32_t *segment_ids, const uint32_t *graph,
                     uint32_t *map, uint32_t graph_degree, uint32_t segment, uint32_t segment_id, uint32_t beam,
                     uint32_t max_iterations, uint32_t min_iterations, uint32_t hash_bit,
                     uint32_t hash_reset_interval, uint32_t start_points, uint32_t dim, Metric metric) {
    uint32_t query_id = blockIdx.x;

    extern __shared__ uint32_t shared_mem[];

    uint32_t result_buffer_size = beam + graph_degree;
    result_buffer_size = roundUp32(result_buffer_size);
    uint32_t query_buffer_size = roundUp32(dim);

    auto *query_buffer = reinterpret_cast<uint8_t *>(shared_mem);
    auto *result_id_buffer = reinterpret_cast<uint32_t *>(query_buffer + query_buffer_size);
    auto *result_dist_buffer = reinterpret_cast<float *>(result_id_buffer + result_buffer_size);
    auto *select_buffer = reinterpret_cast<uint32_t *>(result_dist_buffer + result_buffer_size);
    auto *workspace_buffer = reinterpret_cast<uint32_t *>(select_buffer + 1);
    auto *terminate_flag = reinterpret_cast<uint32_t *>(workspace_buffer + 3);
    auto *visited_hashmap = reinterpret_cast<uint32_t *>(terminate_flag + 1);
    auto *is_search = reinterpret_cast<uint32_t *>(visited_hashmap + hashmap::get_size(hash_bit));

    if (threadIdx.x == 0) {
        is_search[0] = 0;
        for (uint32_t i = 0; i < segment; i++) {
            if (segment_ids[query_id * segment + i] == segment_id) {
                is_search[0] = 1;
                break;
            }
            if (segment_ids[query_id * segment + i] == get_max_value<uint32_t>()) {
                break;
            }
        }
    }
    __syncthreads();

    if (is_search[0] == 0) {
        return;
    }

    for (uint32_t i = threadIdx.x; i < query_buffer_size; i += blockDim.x) {
        if (i < dim) {
            query_buffer[i] = query[query_id * dim + i];
        } else {
            query_buffer[i] = 0;
        }
    }
    if (threadIdx.x == 0) {
        terminate_flag[0] = 0;
        workspace_buffer[0] = ~0u;
    }
    hashmap::init(visited_hashmap, hash_bit, 0);
    __syncthreads();

    result_init_for_large(result_id_buffer + beam, result_dist_buffer + beam, query_buffer, data, graph,
                          start_points, dim, graph_degree, visited_hashmap, hash_bit, metric);
    __syncthreads();

    uint32_t iter = 0;
    const uint32_t multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATE > 128)) ? 1 : 0;
    const uint32_t multi_warps_2 = ((blockDim.x >= 64) && (MAX_BEAM > 256)) ? 1 : 0;
    while (true) {
        update_result_buffer<MAX_BEAM, MAX_CANDIDATE>(result_dist_buffer, result_id_buffer, beam,
                                                      result_dist_buffer + beam, result_id_buffer + beam, graph_degree,
                                                      workspace_buffer, (iter == 0), multi_warps_1, multi_warps_2);

        if ((iter + 1) % hash_reset_interval == 0) {
            unsigned hash_start_tid;
            if (blockDim.x == 32) {
                hash_start_tid = 0;
            } else if (blockDim.x == 64) {
                if (multi_warps_1 || multi_warps_2) {
                    hash_start_tid = 0;
                } else {
                    hash_start_tid = 32;
                }
            } else {
                if (multi_warps_1 || multi_warps_2) {
                    hash_start_tid = 64;
                } else {
                    hash_start_tid = 32;
                }
            }
            hashmap::init(visited_hashmap, hash_bit, hash_start_tid);
        }
        __syncthreads();

        if (iter + 1 == max_iterations) {
            break;
        }

        if (threadIdx.x < 32) {
            select_current_point(terminate_flag, select_buffer, result_id_buffer, beam);
        }
        if ((iter + 1) % hash_reset_interval == 0) {
            const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
            hashmap_restore(visited_hashmap, hash_bit, result_id_buffer, beam, first_tid);
        }
        __syncthreads();

        if (*terminate_flag && iter >= min_iterations) {
            break;
        }

        compute_distance_for_large(result_id_buffer + beam, result_dist_buffer + beam, query_buffer, graph,
                                   graph_degree, visited_hashmap, hash_bit, select_buffer, dim, data, metric);
        __syncthreads();

        iter++;
    }

    result_merge<MAX_TOPK>(final_result_dists + query_id * top_k, final_result_ids + query_id * top_k,
                           result_dist_buffer, result_id_buffer, map, top_k);
}