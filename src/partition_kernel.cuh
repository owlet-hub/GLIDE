#include "distance.cuh"
#include "utils.cuh"
#include <curand_kernel.h>

#pragma once

__global__ void
sample_kernel(const float *d_data, float *d_sample_data, const uint32_t *d_indices, uint32_t dim, uint32_t sample_num) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= sample_num) {
        return;
    }

    uint32_t sample_id = d_indices[point_id];
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        d_sample_data[point_id * dim + i] = d_data[sample_id * dim + i];
    }
}

__global__ void
sample_for_large_kernel(const uint8_t *d_data, float *d_sample_data, const uint32_t *d_indices, uint32_t dim,
                        uint32_t sample_num, uint32_t start_id, uint32_t end_id) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= sample_num) {
        return;
    }

    uint32_t sample_id = d_indices[point_id];
    if(sample_id < start_id || sample_id >= end_id){
        return;
    }

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        d_sample_data[point_id * dim + i] = static_cast<float>(d_data[sample_id * dim + i]);
    }
}

template<uint32_t MAX_CENTROID>
__global__ void
define_partition_kernel(const float *d_data, float *d_centroids, uint32_t num, uint32_t dim, uint32_t centroid_num,
                        uint32_t *d_segment_length, uint32_t *d_segment_index, float boundary_factor, Metric metric) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *point_buffer = reinterpret_cast<float *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = d_data[point_id * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, d_centroids + i * dim, dim, metric);
        if(lane_id == 0){
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        uint32_t segment_index = atomicAdd(d_segment_length + centroid_id_buffer[0], 1);
        uint32_t boundary_index = (centroid_num > 1 && centroid_dist_buffer[1] / centroid_dist_buffer[0] <= boundary_factor)
                                  ? atomicAdd(d_segment_length + centroid_num, 1) : get_max_value<uint32_t>();

        d_segment_index[point_id * 3] = centroid_id_buffer[0];
        d_segment_index[point_id * 3 + 1] = segment_index;
        d_segment_index[point_id * 3 + 2] = boundary_index;
    }
}

template<uint32_t MAX_CENTROID>
__global__ void
define_partition_for_large_kernel(const uint8_t *d_data, float *d_centroids, uint32_t batch_size,
                                  uint32_t dim, uint32_t centroid_num, uint32_t *d_segment_length,
                                  uint32_t *d_segment_index, Metric metric) {
    uint64_t point_id = blockIdx.x;
    if (point_id >= batch_size) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *point_buffer = reinterpret_cast<uint8_t *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = d_data[point_id * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, d_centroids + i * dim, dim, metric);
        if(lane_id == 0){
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        atomicAdd(d_segment_length + centroid_id_buffer[0], 1);
        d_segment_index[point_id] = centroid_id_buffer[0];
    }
}

__global__ void
reorder_kernel(const float *d_data, float *d_reorder_data, const uint32_t *d_segment_index, uint32_t *d_map,
               const uint32_t *d_segment_start, uint32_t num, uint32_t dim) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    uint32_t segment_id = d_segment_index[point_id * 3];
    uint32_t segment_index = d_segment_index[point_id * 3 + 1];
    uint32_t boundary_index = d_segment_index[point_id * 3 + 2];
    uint32_t segment_start = d_segment_start[segment_id];
    uint32_t reorder_id = segment_start + segment_index;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        d_reorder_data[reorder_id * dim + i] = d_data[point_id * dim + i];
    }
    if (threadIdx.x == 0) {
        d_map[reorder_id] = point_id;
    }

    if (boundary_index != get_max_value<uint32_t>()) {
        uint32_t boundary_id = num + boundary_index;
        for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
            d_reorder_data[boundary_id * dim + i] = d_data[point_id * dim + i];
        }
        if (threadIdx.x == 0) {
            d_map[boundary_id] = point_id;
        }
    }
}

__global__ void
reorder_for_large_kernel(const uint8_t *d_data, uint8_t *d_reorder_data, const uint32_t *d_segment_index,
                         uint32_t *d_mapping, uint32_t batch_size, uint32_t dim, uint32_t target_segment_id,
                         uint32_t *d_counts, uint32_t start_id) {
    uint64_t point_id = blockIdx.x;
    if (point_id >= batch_size) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *segment_index = reinterpret_cast<uint64_t *>(shared_mem);

    uint32_t segment_id = d_segment_index[point_id];
    if(segment_id != target_segment_id){
        return;
    }

    if(threadIdx.x == 0){
        segment_index[0] = atomicAdd(d_counts + segment_id, 1);
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        d_reorder_data[segment_index[0] * dim + i] = d_data[point_id * dim + i];
    }
    if (threadIdx.x == 0) {
        d_mapping[segment_index[0]] = point_id + start_id;
    }
    __syncthreads();
}