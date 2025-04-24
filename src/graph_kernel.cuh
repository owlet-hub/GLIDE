#include "search_kernel.cuh"

#pragma once

__device__ void
get_init(uint32_t *candidate_id_buffer, float *candidate_dist_buffer, float *point_buffer, uint32_t point_id,
         const float *data, const uint32_t *graph, uint32_t start_point, uint32_t dim, uint32_t knn_degree,
         uint32_t *visited_hashmap, uint32_t hash_bit, Metric metric) {
    for (uint32_t i = threadIdx.x; i < knn_degree; i += blockDim.x) {
        uint32_t index = graph[start_point * knn_degree + i];
        uint32_t flag = (index != get_max_value<uint32_t>()) ? hashmap::insert(visited_hashmap, hash_bit, index) : 0;
        candidate_id_buffer[i] = (flag == 1) ? index : get_max_value<uint32_t>();
    }
    __syncthreads();

    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;
    for (uint32_t i = warp_id; i < knn_degree; i += warp_num) {
        uint32_t index = candidate_id_buffer[i];
        float dist = (index != get_max_value<uint32_t>() && index != point_id) ?
                     compute_similarity_warp(point_buffer, data + index * dim, dim, metric) : get_max_value<float>();
        if (lane_id == 0) {
            candidate_dist_buffer[i] = dist;
        }
    }
}

__device__ void
compute_distance_for_build(uint32_t *candidate_id_buffer, float *candidate_dist_buffer, float *point_buffer,
                           uint32_t point_id, const uint32_t *graph, uint32_t knn_degree, uint32_t *visited_hashmap,
                           uint32_t hash_bit, uint32_t *select_buffer, uint32_t dim, const float *data, Metric metric) {
    for (uint32_t i = threadIdx.x; i < knn_degree; i += blockDim.x) {
        uint32_t neighbor_id = get_max_value<uint32_t>();
        uint32_t current_point_id = select_buffer[0];
        if (current_point_id != get_max_value<uint32_t>()) {
            neighbor_id = graph[i + (knn_degree * current_point_id)];
        }
        uint32_t flag = (neighbor_id != get_max_value<uint32_t>() && neighbor_id != point_id) ?
                        hashmap::insert(visited_hashmap, hash_bit, neighbor_id) : 0;
        candidate_id_buffer[i] = (flag == 1) ? neighbor_id : get_max_value<uint32_t>();
    }
    __syncthreads();

    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    for (uint32_t i = warp_id; i < knn_degree; i += warp_num) {
        uint32_t index = candidate_id_buffer[i];
        float dist = (index != get_max_value<uint32_t>()) ?
                     compute_similarity_warp(point_buffer, data + index * dim, dim, metric) : get_max_value<float>();
        if (lane_id == 0) {
            candidate_dist_buffer[i] = dist;
        }
    }
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM>
__device__ void
get_neighbors(float *point_buffer, uint32_t point_id, const float *data, uint32_t dim, uint32_t beam,
              const uint32_t *knn_graph, uint32_t *beam_id_buffer, float *beam_dist_buffer,
              uint32_t *candidate_id_buffer, float *candidate_dist_buffer, uint32_t *terminate_flag,
              uint32_t *workspace_buffer, uint32_t *visited_hashmap, uint32_t hash_bit, uint32_t hash_reset_interval,
              uint32_t knn_degree, uint32_t max_iterations, uint32_t min_iterations, uint32_t *select_buffer,
              Metric metric) {
    uint32_t iter = 0;
    const uint32_t multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATE > 128)) ? 1 : 0;
    const uint32_t multi_warps_2 = ((blockDim.x >= 64) && (MAX_BEAM > 256)) ? 1 : 0;
    while (true) {
        update_result_buffer<MAX_BEAM, MAX_CANDIDATE>(beam_dist_buffer, beam_id_buffer, beam, candidate_dist_buffer,
                                                      candidate_id_buffer, knn_degree, workspace_buffer, (iter == 0),
                                                      multi_warps_1, multi_warps_2);
        if ((iter + 1) % hash_reset_interval == 0) {
            uint32_t hash_start_tid;
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
            select_current_point(terminate_flag, select_buffer, beam_id_buffer, beam);
        }
        if ((iter + 1) % hash_reset_interval == 0) {
            uint32_t first_tid = ((blockDim.x <= 32) ? 0 : 32);
            hashmap_restore(visited_hashmap, hash_bit, beam_id_buffer, beam, first_tid);
        }
        __syncthreads();

        if (*terminate_flag && iter >= min_iterations) {
            break;
        }

        compute_distance_for_build(candidate_id_buffer, candidate_dist_buffer, point_buffer, point_id, knn_graph,
                                   knn_degree, visited_hashmap, hash_bit, select_buffer, dim, data, metric);
        __syncthreads();

        iter++;
    }

    for (uint32_t i = threadIdx.x; i < beam; i += blockDim.x) {
        beam_id_buffer[i] = beam_id_buffer[i] & index_mask::clean;
    }
}

__device__ void
sync_prune(const float *data, uint32_t segment_start, uint32_t dim, uint32_t *beam_id_buffer, float *beam_dist_buffer,
           uint32_t graph_degree, uint32_t max_graph_degree, uint32_t beam, uint32_t *neighbor_buffer,
           uint32_t *prune_buffer, uint32_t *bitmap, float relaxant_factor, Metric metric) {
    uint32_t *neighbor_size = &prune_buffer[0];
    uint32_t *pool_size = &prune_buffer[1];

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_num = blockDim.x / warpSize;

    if (threadIdx.x == 0) {
        neighbor_size[0] = 0;
        pool_size[0] = beam;
    } else if (threadIdx.x >= 32) {
        for (uint32_t i = threadIdx.x - 32; i < max_graph_degree; i += blockDim.x - 32) {
            neighbor_buffer[i] = get_max_value<uint32_t>();
        }
    }
    __syncthreads();

    while (neighbor_size[0] < graph_degree && pool_size[0] > 0) {
        uint32_t bitmap_size = ceildiv<uint32_t>(pool_size[0], 32);

        uint32_t id = beam_id_buffer[swizzling(0)];
        if (threadIdx.x == 0) {
            neighbor_buffer[neighbor_size[0]] = id + segment_start;
            neighbor_size[0]++;

            for (uint32_t i = 0; i < bitmap_size; i++) {
                bitmap[i] = 0;
            }
        }
        __syncthreads();

        for (uint32_t i = warp_id + 1; i < pool_size[0]; i += warp_num) {
            uint32_t index = beam_id_buffer[swizzling(i)];
            float dist = compute_similarity_warp(data + (segment_start + id) * dim,
                                                 data + (segment_start + index) * dim, dim, metric);

            if (lane_id == 0) {
                uint32_t allow = (relaxant_factor * dist < beam_dist_buffer[swizzling(i)]) ? 0 : 1;
                atomicOr(&bitmap[i / 32], (allow << (i % 32)));
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            uint32_t allow_num = 0;
            for (uint32_t i = lane_id; i < bitmap_size * 32; i += warpSize) {
                uint32_t bitmap_index = i / 32;
                uint32_t bit_index = i % 32;

                uint32_t allow = (i < pool_size[0] && (bitmap[bitmap_index] & (1 << bit_index))) ? 1 : 0;
                const uint32_t ballot_mask = bitmap[bitmap_index];

                uint32_t id_temp = (i < pool_size[0]) ? beam_id_buffer[swizzling(i)] : get_max_value<uint32_t>();
                float dist_temp = (i < pool_size[0]) ? beam_dist_buffer[swizzling(i)] : get_max_value<float>();

                if (allow) {
                    const auto next = __popc(ballot_mask & ((1 << lane_id) - 1)) + allow_num;
                    beam_id_buffer[swizzling(next)] = id_temp;
                    beam_dist_buffer[swizzling(next)] = dist_temp;
                }
                allow_num += __popc(ballot_mask);
            }
            if (lane_id == 0) {
                pool_size[0] = allow_num;
            }
        }
        __syncthreads();
    }
}

__device__ inline void
map_to_graph(uint32_t *graph, uint32_t *neighbor_buffer, uint32_t graph_degree) {
    for (uint32_t i = threadIdx.x; i < graph_degree; i += blockDim.x) {
        graph[i] = neighbor_buffer[i];
    }
}

__device__ inline void
map_to_graph_sub(uint32_t *graph, uint32_t *neighbor_buffer, const uint32_t *map, uint32_t graph_degree) {
    for (uint32_t i = threadIdx.x; i < graph_degree; i += blockDim.x) {
        graph[i] = (neighbor_buffer[i] == get_max_value<uint32_t>()) ?
                   get_max_value<uint32_t>() : map[neighbor_buffer[i]];
    }
}

__device__ void
map_to_graph_boundary(uint32_t *graph, uint32_t *s_graph, uint32_t *neighbor_buffer, const uint32_t *map,
                      uint32_t graph_degree, uint32_t max_graph_degree, uint32_t *bitmap, uint32_t max_bitmap_size) {
    uint32_t warp_id = threadIdx.x / warpSize;
    if (warp_id > 0) {
        return;
    }

    for (uint32_t i = threadIdx.x; i < max_bitmap_size; i += warpSize) {
        bitmap[i] = 0;
    }
    for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
        s_graph[i] = (i < graph_degree) ? graph[i] : get_max_value<uint32_t>();
    }
    __syncthreads();

    uint32_t valid_boundary_num = 0;
    uint32_t repetitive_boundary_num = 0;
    for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
        uint32_t valid = (i < graph_degree && neighbor_buffer[i] != get_max_value<uint32_t>()) ? 1 : 0;
        uint32_t index = (valid) ? map[neighbor_buffer[i]] : get_max_value<uint32_t>();
        uint32_t non_repetitive = 1;
        uint32_t j;
        for (j = 0; j < graph_degree; j++) {
            if (s_graph[j] == get_max_value<uint32_t>()) {
                break;
            }
            if (s_graph[j] == index) {
                non_repetitive = 0;
                break;
            }
        }

        uint32_t ballot_mask1 = __ballot_sync(warp_full_mask(), valid && non_repetitive);
        uint32_t ballot_mask2 = __ballot_sync(warp_full_mask(), valid && !non_repetitive);
        if (valid && non_repetitive) {
            uint32_t position = __popc(ballot_mask1 & ((1 << threadIdx.x) - 1)) + valid_boundary_num;
            neighbor_buffer[position] = index;
        }

        if (valid && !non_repetitive) {
            atomicOr(&bitmap[j / 32], (1 << (j % 32)));
        }

        valid_boundary_num += __popc(ballot_mask1);
        repetitive_boundary_num += __popc(ballot_mask2);
    }

    if (repetitive_boundary_num >= graph_degree / 2) {
        uint32_t valid_pos_num = 0;
        for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
            uint32_t valid_pos = (i < graph_degree && s_graph[i] == get_max_value<uint32_t>()) ? 1 : 0;
            uint32_t ballot_mask = __ballot_sync(warp_full_mask(), valid_pos);
            if (valid_pos) {
                uint32_t position = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + valid_pos_num;
                if (position < valid_boundary_num) {
                    graph[i] = neighbor_buffer[position];
                }
            }
            valid_pos_num += __popc(ballot_mask);
            if (valid_pos_num >= valid_boundary_num) {
                break;
            }
        }
    } else {
        uint32_t max_need_size = graph_degree / 2 - repetitive_boundary_num;
        uint32_t need_num = (valid_boundary_num < max_need_size) ? valid_boundary_num : max_need_size;
        uint32_t valid_pos_num = 0;
        for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
            uint32_t graph_index = (i / 32) * 32 + 31 - i % 32;
            uint32_t repetitive = (bitmap[graph_index / 32] & (1 << (graph_index % 32))) ? 1 : 0;
            uint32_t invalid = (graph_index < graph_degree && s_graph[graph_index] == get_max_value<uint32_t>()) ? 1
                                                                                                                 : 0;
            uint32_t ballot_mask = __ballot_sync(warp_full_mask(), invalid || !repetitive);
            if (invalid || !repetitive) {
                uint32_t position = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + valid_pos_num;
                if (position < need_num) {
                    graph[graph_index] = neighbor_buffer[position];
                }
            }
            valid_pos_num += __popc(ballot_mask);
            if (valid_pos_num >= need_num) {
                break;
            }
        }
    }
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM>
__global__ void
build_kernel(const float *data, uint32_t *graph, const uint32_t *start_points, const uint32_t *segment_starts,
             const uint32_t *segment_lengths, uint32_t knn_degree, uint32_t graph_degree, uint32_t max_graph_degree,
             uint32_t segment_num, uint32_t dim, const uint32_t *map, const uint32_t *knn_graph, uint32_t beam,
             uint32_t hash_bit, uint32_t hash_reset_interval, uint32_t max_iterations, uint32_t min_iterations,
             uint32_t num, uint32_t offset, float relaxant_factor, Metric metric) {
    uint32_t point_id = blockIdx.x + offset;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];

    uint32_t result_buffer_size = beam + knn_degree;
    result_buffer_size = roundUp32(result_buffer_size);

    auto *point_buffer = reinterpret_cast<float *>(shared_mem);
    auto *result_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *result_dist_buffer = reinterpret_cast<float *>(result_id_buffer + result_buffer_size);
    auto *neighbor_buffer = reinterpret_cast<uint32_t *>(result_dist_buffer + result_buffer_size);
    auto *visited_hashmap = reinterpret_cast<uint32_t *>(neighbor_buffer + max_graph_degree);
    auto *select_buffer = reinterpret_cast<uint32_t *>(visited_hashmap + hashmap::get_size(hash_bit));
    auto *workspace_buffer = reinterpret_cast<uint32_t *>(select_buffer + 1);
    auto *terminate_flag = reinterpret_cast<uint32_t *>(workspace_buffer + 3);
    auto *prune_buffer = reinterpret_cast<uint32_t *>(terminate_flag + 1);
    auto *bitmap = reinterpret_cast<uint32_t *>(prune_buffer + 2);

    uint32_t segment_id;
    for (segment_id = 0; segment_id < segment_num; segment_id++) {
        uint32_t right = segment_starts[segment_id] + segment_lengths[segment_id];
        if (point_id < right) {
            break;
        }
    }

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[point_id * dim + i];
    }
    if (threadIdx.x == 0) {
        terminate_flag[0] = 0;
        workspace_buffer[0] = ~0u;
    }
    hashmap::init(visited_hashmap, hash_bit, 0);
    __syncthreads();

    get_init(result_id_buffer + beam, result_dist_buffer + beam, point_buffer, point_id,
             data + segment_starts[segment_id] * dim, knn_graph + segment_starts[segment_id] * knn_degree,
             start_points[segment_id], dim, knn_degree, visited_hashmap, hash_bit, metric);
    __syncthreads();

    get_neighbors<MAX_CANDIDATE, MAX_BEAM>(point_buffer, point_id - segment_starts[segment_id],
                                           data + segment_starts[segment_id] * dim, dim, beam,
                                           knn_graph + segment_starts[segment_id] * knn_degree, result_id_buffer,
                                           result_dist_buffer, result_id_buffer + beam, result_dist_buffer + beam,
                                           terminate_flag, workspace_buffer, visited_hashmap, hash_bit,
                                           hash_reset_interval, knn_degree, max_iterations, min_iterations,
                                           select_buffer, metric);
    __syncthreads();

    sync_prune(data, segment_starts[segment_id], dim, result_id_buffer, result_dist_buffer, graph_degree,
               max_graph_degree, beam, neighbor_buffer, prune_buffer, bitmap, relaxant_factor, metric);

    map_to_graph_sub(graph + map[point_id] * graph_degree, neighbor_buffer, map, graph_degree);
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM>
__global__ void
build_boundary_kernel(const float *data, uint32_t *graph, const uint32_t *start_points, uint32_t knn_degree,
                      uint32_t graph_degree, uint32_t max_graph_degree, uint32_t dim, uint32_t bitmap_size,
                      uint32_t segment_id, uint32_t segment_start, const uint32_t *map, const uint32_t *knn_graph,
                      uint32_t beam, uint32_t hash_bit, uint32_t hash_reset_interval, uint32_t max_iterations,
                      uint32_t min_iterations, uint32_t num, uint32_t offset, float relaxant_factor, Metric metric) {
    uint32_t point_id = blockIdx.x + offset;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];

    uint32_t result_buffer_size = beam + knn_degree;
    result_buffer_size = roundUp32(result_buffer_size);

    auto *point_buffer = reinterpret_cast<float *>(shared_mem);
    auto *result_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *result_dist_buffer = reinterpret_cast<float *>(result_id_buffer + result_buffer_size);
    auto *neighbor_buffer = reinterpret_cast<uint32_t *>(result_dist_buffer + result_buffer_size);
    auto *s_graph = reinterpret_cast<uint32_t *>(neighbor_buffer + max_graph_degree);
    auto *visited_hashmap = reinterpret_cast<uint32_t *>(s_graph + max_graph_degree);
    auto *select_buffer = reinterpret_cast<uint32_t *>(visited_hashmap + hashmap::get_size(hash_bit));
    auto *workspace_buffer = reinterpret_cast<uint32_t *>(select_buffer + 1);
    auto *terminate_flag = reinterpret_cast<uint32_t *>(workspace_buffer + 3);
    auto *prune_buffer = reinterpret_cast<uint32_t *>(terminate_flag + 1);
    auto *bitmap = reinterpret_cast<uint32_t *>(prune_buffer + 2);

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[point_id * dim + i];
    }
    if (threadIdx.x == 0) {
        terminate_flag[0] = 0;
        workspace_buffer[0] = ~0u;
    }
    hashmap::init(visited_hashmap, hash_bit, 0);
    __syncthreads();

    get_init(result_id_buffer + beam, result_dist_buffer + beam, point_buffer, point_id,
             data + segment_start * dim, knn_graph + segment_start * knn_degree, start_points[segment_id],
             dim, knn_degree, visited_hashmap, hash_bit, metric);
    __syncthreads();

    get_neighbors<MAX_CANDIDATE, MAX_BEAM>(point_buffer, point_id - segment_start, data + segment_start * dim, dim,
                                           beam, knn_graph + segment_start * knn_degree, result_id_buffer,
                                           result_dist_buffer, result_id_buffer + beam, result_dist_buffer + beam,
                                           terminate_flag, workspace_buffer, visited_hashmap, hash_bit,
                                           hash_reset_interval, knn_degree, max_iterations, min_iterations,
                                           select_buffer, metric);
    __syncthreads();

    sync_prune(data, segment_start, dim, result_id_buffer, result_dist_buffer, graph_degree, max_graph_degree, beam,
               neighbor_buffer, prune_buffer, bitmap, relaxant_factor, metric);

    map_to_graph_boundary(graph + map[point_id] * graph_degree, s_graph, neighbor_buffer, map, graph_degree,
                          max_graph_degree, bitmap, bitmap_size);
}

__global__ void
reverse_graph_kernel(uint32_t *const destination_nodes, uint32_t *const reverse_graph,
                     uint32_t *const reverse_graph_degrees, uint32_t *graph, uint32_t num,
                     uint32_t graph_degree, uint32_t degree_id) {
    uint32_t thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
    uint32_t thread_num = blockDim.x * gridDim.x;

    for (uint32_t i = thread_id; i < num; i += thread_num) {
        destination_nodes[i] = graph[i * graph_degree + degree_id];
    }

    for (uint32_t source_node = thread_id; source_node < num; source_node += thread_num) {
        uint32_t destination_node = destination_nodes[source_node];

        if (destination_node >= num) {
            continue;
        }

        uint32_t have = 0;
        for (uint32_t i = 0; i < graph_degree; i++) {
            if (graph[destination_node * graph_degree + i] == source_node) {
                have = 1;
                break;
            }
        }

        if (have) {
            continue;
        }

        uint32_t reverse_index = atomicAdd(reverse_graph_degrees + destination_node, 1);
        if (reverse_index < graph_degree) {
            reverse_graph[destination_node * graph_degree + reverse_index] = source_node;
        }
    }
}

__global__ void
insert_reverse_kernel(uint32_t *graph, uint32_t *reverse_graph, uint32_t *reverse_graph_degrees,
                      uint32_t num, uint32_t graph_degree, uint32_t max_graph_degree) {
    uint32_t point_id = blockIdx.x;

    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *s_graph = reinterpret_cast<uint32_t *>(shared_mem);
    for (uint32_t i = threadIdx.x; i < max_graph_degree; i += blockDim.x) {
        s_graph[i] = (i < graph_degree) ? graph[point_id * graph_degree + i] : get_max_value<uint32_t>();
    }

    uint32_t num_reverse = (reverse_graph_degrees[point_id] > graph_degree) ?
                           graph_degree : reverse_graph_degrees[point_id];

    uint32_t num_have_position = 0;
    for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
        uint32_t have_position = (i < graph_degree && s_graph[i] == get_max_value<uint32_t>()) ? 1 : 0;

        uint32_t ballot_mask = __ballot_sync(warp_full_mask(), have_position);
        if (have_position) {
            uint32_t position = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_have_position;;
            if (position < num_reverse) {
                graph[point_id * graph_degree + i] = reverse_graph[point_id * graph_degree + position];
            }
        }
        num_have_position += __popc(ballot_mask);
        if (num_have_position >= num_reverse) {
            break;
        }
    }
}

__device__ void
select_point(uint32_t *beam_id_buffer, float *beam_dist_buffer, uint32_t pool_size, uint32_t point_id,
             uint32_t *position, uint32_t *s_graph, uint32_t graph_degree) {
    if (threadIdx.x == 0) {
        position[0] = get_max_value<uint32_t>();
    }

    uint32_t num_valid_points = 0;
    for (uint32_t i = threadIdx.x; i < roundUp32(pool_size); i += warpSize) {
        uint32_t id = (i < pool_size) ? beam_id_buffer[swizzling(i)] : get_max_value<uint32_t>();

        uint32_t valid = 0;
        if (id != get_max_value<uint32_t>() && id != point_id) {
            uint32_t exit = 0;
            for (uint32_t j = 0; j < graph_degree; j++) {
                if (s_graph[j] == id) {
                    exit = 1;
                    break;
                }
            }
            if (exit == 0) {
                valid = 1;
            }
        }

        const uint32_t ballot_mask = __ballot_sync(warp_full_mask(), valid);
        if (valid) {
            const auto pos = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_valid_points;
            if (pos < 1) {
                position[0] = i;
            }
        }
        num_valid_points += __popc(ballot_mask);
        if (num_valid_points >= 1) {
            break;
        }
    }
}

__device__ void
sync_prune_refine(const float *data, uint32_t point_id, uint32_t dim, uint32_t *beam_id_buffer, float *beam_dist_buffer,
                  uint32_t graph_degree, uint32_t beam, uint32_t *neighbor_buffer, uint32_t *s_graph,
                  uint32_t *prune_buffer, uint32_t *bitmap, float relaxant_factor, Metric metric) {
    uint32_t *neighbor_size = &prune_buffer[0];
    uint32_t *pool_size = &prune_buffer[1];
    uint32_t *need_num = &prune_buffer[2];
    uint32_t *position = &prune_buffer[3];

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_num = blockDim.x / warpSize;

    if (threadIdx.x == 0) {
        neighbor_size[0] = 0;
        pool_size[0] = beam;
    } else if (threadIdx.x >= 32) {
        for (unsigned i = threadIdx.x - 32; i < graph_degree; i += blockDim.x - 32) {
            neighbor_buffer[i] = get_max_value<uint32_t>();
        }
    }
    __syncthreads();

    while (neighbor_size[0] < need_num[0] && pool_size[0] > 0) {
        uint32_t bitmap_size = ceildiv<uint32_t>(pool_size[0], 32);;

        if (warp_id == 0) {
            select_point(beam_id_buffer, beam_dist_buffer, pool_size[0], point_id, position, s_graph, graph_degree);
        }
        __syncthreads();

        if (position[0] == get_max_value<uint32_t>()) {
            break;
        }

        uint32_t id = beam_id_buffer[swizzling(position[0])];
        if (threadIdx.x == 0) {
            neighbor_buffer[neighbor_size[0]] = id;
            neighbor_size[0]++;

            for (uint32_t i = 0; i < bitmap_size; i++) {
                bitmap[i] = 0;
            }
        }
        __syncthreads();

        for (uint32_t i = warp_id + position[0]; i < pool_size[0]; i += warp_num) {
            uint32_t index = beam_id_buffer[swizzling(i)];
            float dist = compute_similarity_warp(data + id * dim, data + index * dim, dim, metric);

            uint32_t allow = (relaxant_factor * dist < beam_dist_buffer[swizzling(i)]) ? 0 : 1;
            if (lane_id == 0) {
                atomicOr(&bitmap[i / 32], (allow << (i % 32)));
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            uint32_t allow_num = 0;
            for (uint32_t i = lane_id; i < bitmap_size * 32; i += warpSize) {
                uint32_t bitmap_index = i / 32;
                uint32_t bit_index = i % 32;

                uint32_t allow = (i < pool_size[0] && (bitmap[bitmap_index] & (1 << bit_index))) ? 1 : 0;
                const uint32_t ballot_mask = bitmap[bitmap_index];

                uint32_t id_temp = (i < pool_size[0]) ? beam_id_buffer[swizzling(i)] : get_max_value<uint32_t>();
                float dist_temp = (i < pool_size[0]) ? beam_dist_buffer[swizzling(i)] : get_max_value<float>();

                if (allow) {
                    const auto next = __popc(ballot_mask & ((1 << lane_id) - 1)) + allow_num;
                    beam_id_buffer[swizzling(next)] = id_temp;
                    beam_dist_buffer[swizzling(next)] = dist_temp;
                }
                allow_num += __popc(ballot_mask);
            }
            if (lane_id == 0) {
                pool_size[0] = allow_num;
            }
        }
        __syncthreads();
    }
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM, uint32_t MAX_CENTROID>
__global__ void
refine_kernel(const float *data, uint32_t *graph, const uint32_t *start_points, const float *centroids,
              uint32_t graph_degree, uint32_t max_graph_degree, uint32_t dim, uint32_t beam, uint32_t hash_bit,
              uint32_t hash_reset_interval, uint32_t max_iterations, uint32_t min_iterations, uint32_t num,
              uint32_t segment_num, float relaxant_factor, uint32_t offset, Metric metric) {
    uint32_t point_id = blockIdx.x + offset;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];

    uint32_t result_buffer_size = beam + graph_degree;
    result_buffer_size = roundUp32(result_buffer_size);

    auto *point_buffer = reinterpret_cast<float *>(shared_mem);
    auto *result_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *result_dist_buffer = reinterpret_cast<float *>(result_id_buffer + result_buffer_size);
    auto *neighbor_buffer = reinterpret_cast<uint32_t *>(result_dist_buffer + result_buffer_size);
    auto *s_graph = reinterpret_cast<uint32_t *>(neighbor_buffer + max_graph_degree);
    auto *visited_hashmap = reinterpret_cast<uint32_t *>(s_graph + max_graph_degree);
    auto *select_buffer = reinterpret_cast<uint32_t *>(visited_hashmap + hashmap::get_size(hash_bit));
    auto *workspace_buffer = reinterpret_cast<uint32_t *>(select_buffer + 1);
    auto *terminate_flag = reinterpret_cast<uint32_t *>(workspace_buffer + 3);
    auto *prune_buffer = reinterpret_cast<uint32_t *>(terminate_flag + 1);
    auto *bitmap = reinterpret_cast<uint32_t *>(prune_buffer + 4);

    uint32_t *need_num = &prune_buffer[2];
    for (uint32_t i = threadIdx.x; i < max_graph_degree; i += blockDim.x) {
        s_graph[i] = (i < graph_degree) ? graph[point_id * graph_degree + i] : get_max_value<uint32_t>();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        uint32_t valid_num = 0;
        for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
            uint32_t flag = (i < graph_degree && s_graph[i] == get_max_value<uint32_t>()) ? 1 : 0;
            uint32_t ballot_mask = __ballot_sync(warp_full_mask(), flag);
            valid_num += __popc(ballot_mask);
        }
        if (threadIdx.x == 0) {
            need_num[0] = valid_num;
        }
    }
    __syncthreads();

    if (need_num[0] == 0) {
        return;
    }

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[point_id * dim + i];
    }
    if (threadIdx.x == 0) {
        terminate_flag[0] = 0;
        workspace_buffer[0] = ~0u;
    }
    hashmap::init(visited_hashmap, hash_bit, 0);
    __syncthreads();

    result_init<MAX_CENTROID>(result_id_buffer, result_dist_buffer, result_id_buffer + beam, result_dist_buffer + beam,
                              point_buffer, data, graph, start_points, centroids, dim, graph_degree, segment_num,
                              visited_hashmap, hash_bit, metric);
    __syncthreads();

    get_neighbors<MAX_CANDIDATE, MAX_BEAM>(point_buffer, point_id, data, dim, beam, graph, result_id_buffer,
                                           result_dist_buffer, result_id_buffer + beam,
                                           result_dist_buffer + beam, terminate_flag,
                                           workspace_buffer, visited_hashmap, hash_bit, hash_reset_interval,
                                           graph_degree, max_iterations, min_iterations, select_buffer, metric);
    __syncthreads();

    sync_prune_refine(data, point_id, dim, result_id_buffer, result_dist_buffer, graph_degree,
                      beam, neighbor_buffer, s_graph, prune_buffer, bitmap, relaxant_factor, metric);
    __syncthreads();

    if (threadIdx.x < 32) {
        uint32_t valid_pos_num = 0;
        for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
            uint32_t valid_pos = (i < graph_degree && s_graph[i] == get_max_value<uint32_t>()) ? 1 : 0;
            uint32_t ballot_mask = __ballot_sync(warp_full_mask(), valid_pos);
            if (valid_pos) {
                uint32_t can_position = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + valid_pos_num;
                graph[point_id * graph_degree + i] = neighbor_buffer[can_position];
            }
            valid_pos_num += __popc(ballot_mask);
        }
    }
    __syncthreads();
}

__device__ void
get_init_for_large(uint32_t *candidate_id_buffer, float *candidate_dist_buffer, uint8_t *point_buffer,
                   uint32_t point_id, const uint8_t *data, const uint32_t *graph, uint32_t start_point,
                   uint32_t dim, uint32_t knn_degree, uint32_t *visited_hashmap, uint32_t hash_bit, Metric metric) {
    for (uint32_t i = threadIdx.x; i < knn_degree; i += blockDim.x) {
        uint32_t index = graph[static_cast<uint64_t>(start_point) * knn_degree + i];
        uint32_t flag = (index != get_max_value<uint32_t>()) ? hashmap::insert(visited_hashmap, hash_bit, index) : 0;
        candidate_id_buffer[i] = (flag == 1) ? index : get_max_value<uint32_t>();
    }
    __syncthreads();

    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;
    for (uint32_t i = warp_id; i < knn_degree; i += warp_num) {
        uint32_t index = candidate_id_buffer[i];
        float dist = (index != get_max_value<uint32_t>() && index != point_id) ?
                     compute_similarity_warp(point_buffer, data + static_cast<uint64_t>(index) * dim, dim, metric) :
                     get_max_value<float>();
        if (lane_id == 0) {
            candidate_dist_buffer[i] = dist;
        }
    }
}

__device__ void
compute_distance_for_large(uint32_t *candidate_id_buffer, float *candidate_dist_buffer, uint8_t *point_buffer,
                           uint32_t point_id, const uint32_t *graph, uint32_t knn_degree, uint32_t *visited_hashmap,
                           uint32_t hash_bit, uint32_t *select_buffer, uint32_t dim, const uint8_t *data,
                           Metric metric) {
    for (uint32_t i = threadIdx.x; i < knn_degree; i += blockDim.x) {
        uint32_t neighbor_id = get_max_value<uint32_t>();
        const auto current_point_id = select_buffer[0];
        if (current_point_id != get_max_value<uint32_t>()) {
            neighbor_id = graph[i + (static_cast<uint64_t>(knn_degree) * current_point_id)];
        }
        uint32_t flag = (neighbor_id != get_max_value<uint32_t>() && neighbor_id != point_id) ?
                        hashmap::insert(visited_hashmap, hash_bit, neighbor_id) : 0;
        candidate_id_buffer[i] = (flag == 1) ? neighbor_id : get_max_value<uint32_t>();
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    for (uint32_t i = warp_id; i < knn_degree; i += warp_num) {
        uint32_t index = candidate_id_buffer[i];
        float dist = (index != get_max_value<uint32_t>()) ?
                     compute_similarity_warp(point_buffer, data + static_cast<uint64_t>(index) * dim, dim, metric) :
                     get_max_value<float>();
        if (lane_id == 0) {
            candidate_dist_buffer[i] = dist;
        }
    }
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM>
__device__ void
get_neighbors_for_large(uint8_t *point_buffer, uint32_t point_id, const uint8_t *data, uint32_t dim, uint32_t beam,
                        const uint32_t *knn_graph, uint32_t *beam_id_buffer, float *beam_dist_buffer,
                        uint32_t *candidate_id_buffer, float *candidate_dist_buffer, uint32_t *terminate_flag,
                        uint32_t *workspace_buffer, uint32_t *visited_hashmap, uint32_t hash_bit,
                        uint32_t hash_reset_interval, uint32_t knn_degree, uint32_t max_iterations,
                        uint32_t min_iterations, uint32_t *select_buffer, Metric metric) {
    uint32_t iter = 0;
    const uint32_t multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATE > 128)) ? 1 : 0;
    const uint32_t multi_warps_2 = ((blockDim.x >= 64) && (MAX_BEAM > 256)) ? 1 : 0;
    while (true) {
        update_result_buffer<MAX_BEAM, MAX_CANDIDATE>(beam_dist_buffer, beam_id_buffer, beam, candidate_dist_buffer,
                                                      candidate_id_buffer, knn_degree, workspace_buffer, (iter == 0),
                                                      multi_warps_1, multi_warps_2);
        if ((iter + 1) % hash_reset_interval == 0) {
            uint32_t hash_start_tid;
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
            select_current_point(terminate_flag, select_buffer, beam_id_buffer, beam);
        }
        if ((iter + 1) % hash_reset_interval == 0) {
            uint32_t first_tid = ((blockDim.x <= 32) ? 0 : 32);
            hashmap_restore(visited_hashmap, hash_bit, beam_id_buffer, beam, first_tid);
        }
        __syncthreads();

        if (*terminate_flag && iter >= min_iterations) {
            break;
        }

        compute_distance_for_large(candidate_id_buffer, candidate_dist_buffer, point_buffer, point_id, knn_graph,
                                   knn_degree, visited_hashmap, hash_bit, select_buffer, dim, data, metric);
        __syncthreads();

        iter++;
    }

    for (uint32_t i = threadIdx.x; i < beam; i += blockDim.x) {
        beam_id_buffer[i] = beam_id_buffer[i] & index_mask::clean;
    }
}

__device__ void
sync_prune_for_large(const uint8_t *data, uint32_t dim, uint32_t *beam_id_buffer, float *beam_dist_buffer,
                     uint32_t graph_degree, uint32_t max_graph_degree, uint32_t beam, uint32_t *neighbor_buffer,
                     uint32_t *prune_buffer, uint32_t *bitmap, float relaxant_factor, Metric metric) {
    uint32_t *neighbor_size = &prune_buffer[0];
    uint32_t *pool_size = &prune_buffer[1];

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_num = blockDim.x / warpSize;

    if (threadIdx.x == 0) {
        neighbor_size[0] = 0;
        pool_size[0] = beam;
    } else if (threadIdx.x >= 32) {
        for (unsigned i = threadIdx.x - 32; i < max_graph_degree; i += blockDim.x - 32) {
            neighbor_buffer[i] = get_max_value<uint32_t>();
        }
    }
    __syncthreads();

    while (neighbor_size[0] < graph_degree && pool_size[0] > 0) {
        uint32_t bitmap_size = ceildiv<uint32_t>(pool_size[0], 32);

        uint32_t id = beam_id_buffer[swizzling(0)];
        if (threadIdx.x == 0) {
            neighbor_buffer[neighbor_size[0]] = id;
            neighbor_size[0]++;

            for (uint32_t i = 0; i < bitmap_size; i++) {
                bitmap[i] = 0;
            }
        }
        __syncthreads();

        for (uint32_t i = warp_id + 1; i < pool_size[0]; i += warp_num) {
            uint32_t index = beam_id_buffer[swizzling(i)];
            float dist = compute_similarity_warp(data + static_cast<uint64_t>(id) * dim,
                                                 data + static_cast<uint64_t>(index) * dim, dim, metric);

            if (lane_id == 0) {
                uint32_t allow = (relaxant_factor * dist < beam_dist_buffer[swizzling(i)]) ? 0 : 1;
                atomicOr(&bitmap[i / 32], (allow << (i % 32)));
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            uint32_t allow_num = 0;
            for (uint32_t i = lane_id; i < bitmap_size * 32; i += warpSize) {
                uint32_t bitmap_index = i / 32;
                uint32_t bit_index = i % 32;

                uint32_t allow = (i < pool_size[0] && (bitmap[bitmap_index] & (1 << bit_index))) ? 1 : 0;
                const uint32_t ballot_mask = bitmap[bitmap_index];

                uint32_t id_temp = (i < pool_size[0]) ? beam_id_buffer[swizzling(i)] : get_max_value<uint32_t>();
                float dist_temp = (i < pool_size[0]) ? beam_dist_buffer[swizzling(i)] : get_max_value<float>();

                if (allow) {
                    const auto next = __popc(ballot_mask & ((1 << lane_id) - 1)) + allow_num;
                    beam_id_buffer[swizzling(next)] = id_temp;
                    beam_dist_buffer[swizzling(next)] = dist_temp;
                }
                allow_num += __popc(ballot_mask);
            }
            if (lane_id == 0) {
                pool_size[0] = allow_num;
            }
        }
        __syncthreads();
    }
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM>
__global__ void
build_for_large_kernel(const uint8_t *data, uint32_t *graph, uint32_t start_point, uint32_t knn_degree,
                       uint32_t graph_degree, uint32_t max_graph_degree, uint32_t dim, const uint32_t *knn_graph,
                       uint32_t beam, uint32_t hash_bit, uint32_t hash_reset_interval, uint32_t max_iterations,
                       uint32_t min_iterations, uint32_t num, uint32_t offset, float relaxant_factor, Metric metric) {
    uint32_t point_id = blockIdx.x + offset;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];

    uint32_t result_buffer_size = beam + knn_degree;
    result_buffer_size = roundUp32(result_buffer_size);

    auto *point_buffer = reinterpret_cast<uint8_t *>(shared_mem);
    auto *result_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *result_dist_buffer = reinterpret_cast<float *>(result_id_buffer + result_buffer_size);
    auto *neighbor_buffer = reinterpret_cast<uint32_t *>(result_dist_buffer + result_buffer_size);
    auto *visited_hashmap = reinterpret_cast<uint32_t *>(neighbor_buffer + max_graph_degree);
    auto *select_buffer = reinterpret_cast<uint32_t *>(visited_hashmap + hashmap::get_size(hash_bit));
    auto *workspace_buffer = reinterpret_cast<uint32_t *>(select_buffer + 1);
    auto *terminate_flag = reinterpret_cast<uint32_t *>(workspace_buffer + 3);
    auto *prune_buffer = reinterpret_cast<uint32_t *>(terminate_flag + 1);
    auto *bitmap = reinterpret_cast<uint32_t *>(prune_buffer + 2);

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[point_id * dim + i];
    }
    if (threadIdx.x == 0) {
        terminate_flag[0] = 0;
        workspace_buffer[0] = ~0u;
    }
    hashmap::init(visited_hashmap, hash_bit, 0);
    __syncthreads();

    get_init_for_large(result_id_buffer + beam, result_dist_buffer + beam, point_buffer, point_id, data, knn_graph,
                       start_point, dim, knn_degree, visited_hashmap, hash_bit, metric);
    __syncthreads();

    get_neighbors_for_large<MAX_CANDIDATE, MAX_BEAM>(point_buffer, point_id, data, dim, beam, knn_graph,
                                                     result_id_buffer, result_dist_buffer, result_id_buffer + beam,
                                                     result_dist_buffer + beam, terminate_flag, workspace_buffer,
                                                     visited_hashmap, hash_bit, hash_reset_interval, knn_degree,
                                                     max_iterations, min_iterations, select_buffer, metric);
    __syncthreads();

    sync_prune_for_large(data, dim, result_id_buffer, result_dist_buffer, graph_degree, max_graph_degree, beam,
                         neighbor_buffer, prune_buffer, bitmap, relaxant_factor, metric);

    map_to_graph(graph + static_cast<uint64_t>(point_id) * graph_degree, neighbor_buffer, graph_degree);
}

__device__ void
sync_prune_refine_for_large(const uint8_t *data, uint32_t point_id, uint32_t dim, uint32_t *beam_id_buffer,
                            float *beam_dist_buffer, uint32_t graph_degree, uint32_t beam, uint32_t *neighbor_buffer,
                            uint32_t *s_graph, uint32_t *prune_buffer, uint32_t *bitmap, float relaxant_factor,
                            Metric metric) {
    uint32_t *neighbor_size = &prune_buffer[0];
    uint32_t *pool_size = &prune_buffer[1];
    uint32_t *need_num = &prune_buffer[2];
    uint32_t *position = &prune_buffer[3];

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_num = blockDim.x / warpSize;

    if (threadIdx.x == 0) {
        neighbor_size[0] = 0;
        pool_size[0] = beam;
    } else if (threadIdx.x >= 32) {
        for (unsigned i = threadIdx.x - 32; i < graph_degree; i += blockDim.x - 32) {
            neighbor_buffer[i] = get_max_value<uint32_t>();
        }
    }
    __syncthreads();

    while (neighbor_size[0] < need_num[0] && pool_size[0] > 0) {
        uint32_t bitmap_size = ceildiv<uint32_t>(pool_size[0], 32);;

        if (warp_id == 0) {
            select_point(beam_id_buffer, beam_dist_buffer, pool_size[0], point_id, position, s_graph, graph_degree);
        }
        __syncthreads();

        if (position[0] == get_max_value<uint32_t>()) {
            break;
        }

        uint32_t id = beam_id_buffer[swizzling(position[0])];
        if (threadIdx.x == 0) {
            neighbor_buffer[neighbor_size[0]] = id;
            neighbor_size[0]++;

            for (uint32_t i = 0; i < bitmap_size; i++) {
                bitmap[i] = 0;
            }
        }
        __syncthreads();

        for (uint32_t i = warp_id + position[0]; i < pool_size[0]; i += warp_num) {
            uint32_t index = beam_id_buffer[swizzling(i)];
            float dist = compute_similarity_warp(data + static_cast<uint64_t>(id) * dim,
                                                 data + static_cast<uint64_t>(index) * dim, dim, metric);

            uint32_t allow = (relaxant_factor * dist < beam_dist_buffer[swizzling(i)]) ? 0 : 1;
            if (lane_id == 0) {
                atomicOr(&bitmap[i / 32], (allow << (i % 32)));
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            uint32_t allow_num = 0;
            for (uint32_t i = lane_id; i < bitmap_size * 32; i += warpSize) {
                uint32_t bitmap_index = i / 32;
                uint32_t bit_index = i % 32;

                uint32_t allow = (i < pool_size[0] && (bitmap[bitmap_index] & (1 << bit_index))) ? 1 : 0;
                const uint32_t ballot_mask = bitmap[bitmap_index];

                uint32_t id_temp = (i < pool_size[0]) ? beam_id_buffer[swizzling(i)] : get_max_value<uint32_t>();
                float dist_temp = (i < pool_size[0]) ? beam_dist_buffer[swizzling(i)] : get_max_value<float>();

                if (allow) {
                    const auto next = __popc(ballot_mask & ((1 << lane_id) - 1)) + allow_num;
                    beam_id_buffer[swizzling(next)] = id_temp;
                    beam_dist_buffer[swizzling(next)] = dist_temp;
                }
                allow_num += __popc(ballot_mask);
            }
            if (lane_id == 0) {
                pool_size[0] = allow_num;
            }
        }
        __syncthreads();
    }
}

template<uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM>
__global__ void
refine_for_large_kernel(const uint8_t *data, uint32_t *graph, uint32_t start_point, uint32_t graph_degree,
                        uint32_t max_graph_degree, uint32_t dim, uint32_t beam, uint32_t hash_bit,
                        uint32_t hash_reset_interval, uint32_t max_iterations, uint32_t min_iterations, uint32_t num,
                        float relaxant_factor, uint32_t offset, Metric metric) {
    uint32_t point_id = blockIdx.x + offset;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];

    uint32_t result_buffer_size = beam + graph_degree;
    result_buffer_size = roundUp32(result_buffer_size);

    auto *point_buffer = reinterpret_cast<uint8_t *>(shared_mem);
    auto *result_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *result_dist_buffer = reinterpret_cast<float *>(result_id_buffer + result_buffer_size);
    auto *neighbor_buffer = reinterpret_cast<uint32_t *>(result_dist_buffer + result_buffer_size);
    auto *s_graph = reinterpret_cast<uint32_t *>(neighbor_buffer + max_graph_degree);
    auto *visited_hashmap = reinterpret_cast<uint32_t *>(s_graph + max_graph_degree);
    auto *select_buffer = reinterpret_cast<uint32_t *>(visited_hashmap + hashmap::get_size(hash_bit));
    auto *workspace_buffer = reinterpret_cast<uint32_t *>(select_buffer + 1);
    auto *terminate_flag = reinterpret_cast<uint32_t *>(workspace_buffer + 3);
    auto *prune_buffer = reinterpret_cast<uint32_t *>(terminate_flag + 1);
    auto *bitmap = reinterpret_cast<uint32_t *>(prune_buffer + 4);

    uint32_t *need_num = &prune_buffer[2];
    for (uint32_t i = threadIdx.x; i < max_graph_degree; i += blockDim.x) {
        s_graph[i] = (i < graph_degree) ? graph[point_id * graph_degree + i] : get_max_value<uint32_t>();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        uint32_t num_need = 0;
        for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
            uint32_t flag = (i < graph_degree && s_graph[i] == get_max_value<uint32_t>()) ? 1 : 0;
            uint32_t ballot_mask = __ballot_sync(warp_full_mask(), flag);
            num_need += __popc(ballot_mask);
        }
        if (threadIdx.x == 0) {
            need_num[0] = num_need;
        }
    }
    __syncthreads();

    if (need_num[0] == 0) {
        return;
    }

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[point_id * dim + i];
    }
    if (threadIdx.x == 0) {
        terminate_flag[0] = 0;
        workspace_buffer[0] = ~0u;
    }
    hashmap::init(visited_hashmap, hash_bit, 0);
    __syncthreads();

    get_init_for_large(result_id_buffer + beam, result_dist_buffer + beam, point_buffer, point_id, data, graph,
                       start_point, dim, graph_degree, visited_hashmap, hash_bit, metric);
    __syncthreads();

    get_neighbors_for_large<MAX_CANDIDATE, MAX_BEAM>(point_buffer, point_id, data, dim, beam, graph, result_id_buffer,
                                                     result_dist_buffer, result_id_buffer + beam,
                                                     result_dist_buffer + beam, terminate_flag, workspace_buffer,
                                                     visited_hashmap, hash_bit, hash_reset_interval, graph_degree,
                                                     max_iterations, min_iterations, select_buffer, metric);
    __syncthreads();

    sync_prune_refine_for_large(data, point_id, dim, result_id_buffer, result_dist_buffer, graph_degree,
                                beam, neighbor_buffer, s_graph, prune_buffer, bitmap, relaxant_factor, metric);
    __syncthreads();

    if (threadIdx.x < 32) {
        uint32_t valid_pos_num = 0;
        for (uint32_t i = threadIdx.x; i < max_graph_degree; i += warpSize) {
            uint32_t valid_pos = (i < graph_degree && s_graph[i] == get_max_value<uint32_t>()) ? 1 : 0;
            uint32_t ballot_mask = __ballot_sync(warp_full_mask(), valid_pos);
            if (valid_pos) {
                uint32_t can_position = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + valid_pos_num;
                graph[point_id * graph_degree + i] = neighbor_buffer[can_position];
            }
            valid_pos_num += __popc(ballot_mask);
        }
    }
    __syncthreads();
}