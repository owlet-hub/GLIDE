#include "distance.cuh"
#include "utils.cuh"
#include <curand_kernel.h>

#pragma once

template<typename Data_t, typename Index_t>
__global__ void
sample_kernel(const Data_t *data, float *sample_data, const uint32_t *sample_ids, uint32_t dim, uint32_t sample_num) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= sample_num) {
        return;
    }
    uint32_t sample_id = sample_ids[point_id];

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        sample_data[static_cast<Index_t>(point_id) * dim + i] =
                static_cast<float>(data[static_cast<Index_t>(sample_id) * dim + i]);
    }
}

template<uint32_t MAX_CENTROID, typename Data_t, typename Index_t>
__global__ void
define_partition_kernel(const Data_t *data, float *centroids, uint32_t num, uint32_t dim, uint32_t centroid_num,
                        uint32_t *segment_lengths, uint32_t *segment_indexes, float boundary_fact, Metric metric) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *point_buffer = reinterpret_cast<Data_t *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        uint32_t segment_index = atomicAdd(segment_lengths + centroid_id_buffer[0], 1);
        uint32_t boundary_index = (centroid_num > 1 &&
                                   centroid_dist_buffer[1] / centroid_dist_buffer[0] <= boundary_fact)
                                  ? atomicAdd(segment_lengths + centroid_num, 1) : get_max_value<uint32_t>();

        segment_indexes[point_id * 3] = centroid_id_buffer[0];
        segment_indexes[point_id * 3 + 1] = segment_index;
        segment_indexes[point_id * 3 + 2] = boundary_index;
    }
}

template<typename Data_t, typename Index_t>
__global__ void
reorder_kernel(const Data_t *data, Data_t *reorder_data, const uint32_t *segment_indexes, uint32_t *map,
               const uint32_t *segment_starts, uint32_t num, uint32_t dim) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    uint32_t segment_id = segment_indexes[point_id * 3];
    uint32_t segment_index = segment_indexes[point_id * 3 + 1];
    uint32_t boundary_index = segment_indexes[point_id * 3 + 2];
    uint32_t segment_start = segment_starts[segment_id];

    uint32_t reorder_id = segment_start + segment_index;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        reorder_data[static_cast<Index_t>(reorder_id) * dim + i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    if (threadIdx.x == 0) {
        map[reorder_id] = point_id;
    }

    if (boundary_index != get_max_value<uint32_t>()) {
        uint32_t boundary_id = num + boundary_index;
        for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
            reorder_data[static_cast<Index_t>(boundary_id) * dim + i] = data[static_cast<Index_t>(point_id) * dim + i];
        }
        if (threadIdx.x == 0) {
            map[boundary_id] = point_id;
        }
    }
}

template<typename Data_t, typename Index_t>
__global__ void
sample_for_large_kernel(const Data_t *data, float *sample_data, const uint32_t *sample_ids, uint32_t dim,
                        uint32_t sample_num, uint32_t start_id, uint32_t batch_size) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= sample_num) {
        return;
    }

    int sample_id = sample_ids[point_id] - start_id;
    if (sample_id < 0 || sample_id >= batch_size) {
        return;
    }

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        sample_data[static_cast<Index_t>(point_id) * dim + i] =
                static_cast<float>(data[static_cast<Index_t>(sample_id) * dim + i]);
    }
}

template<uint32_t MAX_CENTROIDS, typename Data_t, typename Index_t>
__global__ void
define_partition_for_large_kernel(const Data_t *data, float *centroids, uint32_t batch_size, uint32_t dim,
                                  uint32_t centroid_num, uint32_t *segment_lengths, uint32_t *segment_indexes,
                                  Metric metric) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= batch_size) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];

    auto *point_buffer = reinterpret_cast<Data_t *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROIDS>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        atomicAdd(segment_lengths + centroid_id_buffer[0], 1);
        segment_indexes[point_id] = centroid_id_buffer[0];
    }
}

template<typename Data_t, typename Index_t>
__global__ void
reorder_for_large_kernel(Data_t *data, Data_t *reorder_data, const uint32_t *segment_indexes, uint32_t *map,
                         uint32_t batch_size, uint32_t dim, uint32_t target_segment_id, uint32_t *d_counts,
                         uint32_t offset) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= batch_size) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *segment_index = reinterpret_cast<uint32_t *>(shared_mem);

    uint32_t segment_id = segment_indexes[point_id];
    if (segment_id != target_segment_id) {
        return;
    }

    if (threadIdx.x == 0) {
        segment_index[0] = atomicAdd(d_counts + segment_id, 1);
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        reorder_data[static_cast<Index_t>(segment_index[0]) * dim + i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    if (threadIdx.x == 0) {
        map[segment_index[0]] = point_id + offset;
    }
    __syncthreads();
}

template<uint32_t MAX_CENTROIDS, typename Data_t, typename Index_t>
__global__ void
define_partition_merge_kernel(const Data_t *data, float *centroids, uint32_t batch_size, uint32_t dim,
                                  uint32_t centroid_num, uint32_t *segment_lengths, uint32_t *segment_indexes,
                                  float boundary_fact, Metric metric) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= batch_size) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];

    auto *point_buffer = reinterpret_cast<Data_t *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROIDS>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        atomicAdd(segment_lengths + centroid_id_buffer[0], 1);
        segment_indexes[point_id*2] = centroid_id_buffer[0];
        if(centroid_num > 1 && centroid_dist_buffer[1] / centroid_dist_buffer[0] <= boundary_fact){
            atomicAdd(segment_lengths + centroid_num, 1);
            segment_indexes[point_id*2+1] = centroid_num;
        }else{
            segment_indexes[point_id*2+1] = get_max_value<uint32_t>();
        }
    }
}

template<typename Data_t, typename Index_t>
__global__ void
reorder_merge_kernel(Data_t *data, Data_t *reorder_data, const uint32_t *segment_indexes, uint32_t *map,
                     uint32_t batch_size, uint32_t dim, uint32_t target_segment_id, uint32_t *d_counts,
                     bool boundary, uint32_t offset) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= batch_size) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *segment_index = reinterpret_cast<uint32_t *>(shared_mem);

    uint32_t segment_id = (boundary)?segment_indexes[point_id*2+1]:segment_indexes[point_id*2];
    if (!boundary&&segment_id != target_segment_id) {
        return;
    }
    if(boundary&&segment_id==get_max_value<uint32_t>()){
        return;
    }

    if (threadIdx.x == 0) {
        segment_index[0] = atomicAdd(d_counts + segment_id, 1);
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        reorder_data[static_cast<Index_t>(segment_index[0]) * dim + i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    if (threadIdx.x == 0) {
        map[segment_index[0]] = point_id + offset;
    }
    __syncthreads();
}


template<typename Data_t, typename Index_t>
__global__ void
sequential_partition_kernel(Data_t *data, float *centroids, uint32_t *segment_start, uint32_t *segment_length,
                       uint32_t dim, uint32_t centroid_num) {
    uint32_t dim_id = blockIdx.x;
    uint32_t segment_id = blockIdx.y;
    if (dim_id>=dim||segment_id >= centroid_num) {
        return;
    }
    uint32_t start = segment_start[segment_id];
    uint32_t length = segment_length[segment_id];

    auto *data_ptr = data + start * dim;

    float sum = 0.0f;
    for(uint32_t i=threadIdx.x;i<length;i+=blockDim.x){
        sum += static_cast<float>(data_ptr[static_cast<Index_t>(i)*dim + dim_id]);
    }

    for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    extern __shared__ uint32_t shared_mem[];
    auto *sum_buffer = reinterpret_cast<float *>(shared_mem);

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    if(lane_id==0){
        sum_buffer[warp_id] = sum;
    }
    __syncthreads();

    if(warp_id==0){
        float block_sum = (lane_id < warp_num) ? sum_buffer[lane_id] : 0.0f;
        for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane_id == 0) {
            centroids[segment_id * dim + dim_id] = block_sum / static_cast<float>(length);
        }
    }
}

template<uint32_t MAX_CENTROID, typename Data_t, typename Index_t>
__global__ void
random_identification_1_kernel(const Data_t *data, float *centroids, uint32_t num, uint32_t dim, uint32_t centroid_num,
                             uint32_t *segment_starts, uint32_t *segment_lengths, uint32_t *segment_indexes,
                             uint32_t *segment_boundary_nums, float boundary_fact, float boundary_part, Metric metric) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *point_buffer = reinterpret_cast<Data_t *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t segment_id = 0;
    for (; segment_id < centroid_num; segment_id++) {
        uint32_t right = segment_starts[segment_id] + segment_lengths[segment_id];
        if (point_id < right) {
            break;
        }
    }

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        uint32_t boundary_index = get_max_value<uint32_t>();
        bool candidate = (centroid_id_buffer[0]!=segment_id) || (centroid_num > 1 &&
                            centroid_dist_buffer[1] / centroid_dist_buffer[0] > boundary_fact);
        if(candidate){
            uint32_t cur_boundary = atomicAdd(segment_boundary_nums + segment_id, 1);
            uint32_t max_boundary = static_cast<uint32_t>(
                    static_cast<float>(segment_lengths[segment_id]) * boundary_part);
            if (cur_boundary < max_boundary) {
                boundary_index = atomicAdd(segment_lengths + centroid_num, 1);
            }
        }
        segment_indexes[point_id * 3] = segment_id;
        segment_indexes[point_id * 3 + 1] = point_id - segment_starts[segment_id];
        segment_indexes[point_id * 3 + 2] = boundary_index;
    }
}

template<uint32_t MAX_CENTROID, typename Data_t, typename Index_t>
__global__ void
random_identification_2_kernel(const Data_t *data, float *centroids, uint32_t num, uint32_t dim, uint32_t centroid_num,
              uint32_t *segment_lengths, uint32_t *segment_indexes, float boundary_fact, Metric metric) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *point_buffer = reinterpret_cast<Data_t *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        uint32_t segment_index = atomicAdd(segment_lengths + centroid_id_buffer[0], 1);
        uint32_t boundary_index = (centroid_num > 1 &&
                                   centroid_dist_buffer[1] / centroid_dist_buffer[0] > boundary_fact) ? 1 : 0;

        segment_indexes[point_id * 3] = centroid_id_buffer[0];
        segment_indexes[point_id * 3 + 1] = segment_index;
        segment_indexes[point_id * 3 + 2] = boundary_index;
    }
}

__global__ void
define_kernel(uint32_t num, uint32_t centroid_num, uint32_t *segment_lengths,
              uint32_t *segment_indexes, uint32_t *segment_boundary_nums, float boundary_part) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    uint32_t segment_id = segment_indexes[point_id * 3];
    uint32_t is_candidate = segment_indexes[point_id * 3 + 2];

    if (threadIdx.x == 0) {
        uint32_t boundary_index = get_max_value<uint32_t>();
        if(is_candidate==1){
            uint32_t cur_boundary = atomicAdd(segment_boundary_nums + segment_id, 1);
            uint32_t max_boundary = static_cast<uint32_t>(
                    static_cast<float>(segment_lengths[segment_id]) * boundary_part);
            if (cur_boundary < max_boundary) {
                boundary_index = atomicAdd(segment_lengths + centroid_num, 1);
            }
        }
        segment_indexes[point_id * 3 + 2] = boundary_index;
    }
}

template<uint32_t MAX_CENTROID, typename Data_t, typename Index_t>
__global__ void
boundary_identification_kernel(const Data_t *data, float *centroids, uint32_t num, uint32_t dim, uint32_t centroid_num,
                             uint32_t *segment_starts, uint32_t *segment_lengths, uint32_t *segment_indexes,
                             uint32_t *segment_boundary_nums, float boundary_fact, float boundary_part, Metric metric) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *point_buffer = reinterpret_cast<Data_t *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t segment_id = 0;
    for (; segment_id < centroid_num; segment_id++) {
        uint32_t right = segment_starts[segment_id] + segment_lengths[segment_id];
        if (point_id < right) {
            break;
        }
    }

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        uint32_t boundary_index = get_max_value<uint32_t>();
        bool candidate = (centroid_id_buffer[0]!=segment_id) || (centroid_num > 1 &&
                         centroid_dist_buffer[1] / centroid_dist_buffer[0] <= boundary_fact);
        if(candidate){
            uint32_t cur_boundary = atomicAdd(segment_boundary_nums + segment_id, 1);
            uint32_t max_boundary = static_cast<uint32_t>(
                    static_cast<float>(segment_lengths[segment_id]) * boundary_part);
            if (cur_boundary < max_boundary) {
                boundary_index = atomicAdd(segment_lengths + centroid_num, 1);
            }
        }
        segment_indexes[point_id * 3] = segment_id;
        segment_indexes[point_id * 3 + 1] = point_id - segment_starts[segment_id];
        segment_indexes[point_id * 3 + 2] = boundary_index;
    }
}

template<uint32_t MAX_CENTROID, typename Data_t, typename Index_t>
__global__ void
data_locality_kernel(const Data_t *data, float *centroids, uint32_t num, uint32_t dim, uint32_t centroid_num,
                     uint32_t *segment_indexes, uint32_t *locality_score,
                     float boundary_fact, Metric metric) {
    uint32_t point_id = blockIdx.x;
    if (point_id >= num) {
        return;
    }

    extern __shared__ uint32_t shared_mem[];
    auto *point_buffer = reinterpret_cast<Data_t *>(shared_mem);
    auto *centroid_id_buffer = reinterpret_cast<uint32_t *>(point_buffer + dim);
    auto *centroid_dist_buffer = reinterpret_cast<float *>(centroid_id_buffer + centroid_num);

    uint32_t segment_id = segment_indexes[point_id * 3];

    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t warp_num = blockDim.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        point_buffer[i] = data[static_cast<Index_t>(point_id) * dim + i];
    }
    __syncthreads();

    for (uint32_t i = warp_id; i < centroid_num; i += warp_num) {
        float dist = compute_similarity_warp(point_buffer, centroids + i * dim, dim, metric);
        if (lane_id == 0) {
            centroid_dist_buffer[i] = dist;
            centroid_id_buffer[i] = i;
        }
    }
    __syncthreads();

    sort_with_key_value<MAX_CENTROID>(centroid_num, 0, centroid_id_buffer, centroid_dist_buffer);

    if (threadIdx.x == 0) {
        if(centroid_id_buffer[0]==segment_id){
            if(centroid_num > 1 &&
               centroid_dist_buffer[1] / centroid_dist_buffer[0] > boundary_fact){
                atomicAdd(locality_score + segment_id, 1);
            }else if(centroid_num==1){
                atomicAdd(locality_score + segment_id, 1);
            }
        }
    }
}