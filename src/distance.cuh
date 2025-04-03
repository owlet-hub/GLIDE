
#pragma once

#include <cfloat>
#include "bitonic.cuh"
#include "params.cuh"
#include "utils.cuh"

template<typename Value1_t, typename Value2_t>
__device__ inline float compute_similarity_warp(const Value1_t *point_1, const Value2_t *point_2,
                                                uint32_t dim, Metric metric) {
    uint32_t lane_id = threadIdx.x % warpSize;

    float dist = 0.0f;
    for (uint32_t i = lane_id; i < dim; i += warpSize) {
        float diff = static_cast<float>(point_1[i]) - static_cast<float>(point_2[i]);
        dist += diff * diff;
    }
    for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
        dist += __shfl_down_sync(0xffffffff, dist, offset);
    }
    if(metric == Metric::Euclidean){
        return sqrt(dist);
    }else if(metric == Metric::Cosine){
        return dist;
    }
}

template<uint32_t MAX_NUMBER, typename Val_t, typename Key_t>
__device__ void sort_with_key_value(uint32_t number, uint32_t work_warp_id,
                                    Val_t *val_buffer, Key_t *key_buffer) {
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;

    if (warp_id != work_warp_id) {
        return;
    }
    constexpr uint32_t N = (MAX_NUMBER + 31) / 32;
    Key_t keys[N];
    Val_t vals[N];
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = lane_id + (32 * i);

        Key_t key = (j < number) ? key_buffer[j] : get_max_value<Key_t>();
        Val_t val = (j < number) ? val_buffer[j] : get_max_value<Val_t>();

        keys[i] = key;
        vals[i] = val;
    }

    bitonic::warp_sort<Key_t, Val_t, N>(keys, vals);

    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = (N * lane_id) + i;
        if (j < number) {
            key_buffer[j] = keys[i];
            val_buffer[j] = vals[i];
        }
    }
}

template<uint32_t MAX_NUMBER, typename Key_t>
__device__ void sort_with_key(uint32_t number, uint32_t work_warp_id, Key_t *key_buffer) {
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t warp_id = threadIdx.x / warpSize;

    if (warp_id != work_warp_id) {
        return;
    }
    constexpr uint32_t N = (MAX_NUMBER + 31) / 32;
    Key_t keys[N];
    Key_t vals[N];
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = lane_id + (32 * i);

        Key_t key = (j < number) ? key_buffer[j] : get_max_value<Key_t>();

        keys[i] = key;
        vals[i] = key;
    }

    bitonic::warp_sort<Key_t, Key_t, N>(keys, vals);

    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = (N * lane_id) + i;
        if (j < number) {
            key_buffer[j] = keys[i];
        }
    }
}