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

#include <cstdint>

namespace bitonic {

    namespace detail {

        template<class K, class V>
        __device__ inline void swap_if_needed(K &k0, V &v0, K &k1, V &v1, const bool asc) {
            if ((k0 != k1) && ((k0 < k1) != asc)) {
                const auto tmp_k = k0;
                k0 = k1;
                k1 = tmp_k;
                const auto tmp_v = v0;
                v0 = v1;
                v1 = tmp_v;
            }
        }

        template<class K, class V>
        __device__ inline void swap_if_needed(K &k0, V &v0, const uint32_t lane_offset, const bool asc) {
            auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
            auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
            if ((k0 != k1) && ((k0 < k1) != asc)) {
                k0 = k1;
                v0 = v1;
            }
        }

        template<class K, class V, uint32_t N>
        struct warp_merge_core {
            __device__ inline void operator()(K k[N], V v[N], const uint32_t range, const bool asc) {
                const auto lane_id = threadIdx.x % warpSize;

                if (range == 1) {
                    for(uint32_t b = 2; b <= N; b <<= 1) {
                        for(uint32_t c = b / 2; c >= 1; c >>= 1) {
#pragma unroll
                            for(uint32_t i = 0; i < N; i++) {
                                uint32_t j = i ^ c;
                                if (i >= j) continue;
                                const auto line_id = i + (N * lane_id);
                                const auto p = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
                                swap_if_needed(k[i], v[i], k[j], v[j], p);
                            }
                        }
                    }
                    return;
                }

                const uint32_t b = range;
                for(uint32_t c = b / 2; c >= 1; c >>= 1) {
                    const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
                    for(uint32_t i = 0; i < N; i++) {
                        swap_if_needed(k[i], v[i], c, p);
                    }
                }
                const auto p = ((lane_id & b) == 0);
                for(uint32_t c = N / 2; c >= 1; c >>= 1) {
#pragma unroll
                    for(uint32_t i = 0; i < N; i++) {
                        uint32_t j = i ^ c;
                        if (i >= j) continue;
                        swap_if_needed(k[i], v[i], k[j], v[j], p);
                    }
                }
            }
        };

        template<class K, class V>
        struct warp_merge_core<K, V, 6> {
            __device__ inline void operator()(K k[6], V v[6], const uint32_t range, const bool asc) {
                constexpr uint32_t N=6;
                const auto lane_id = threadIdx.x % warpSize;

                if (range == 1) {
                    for(uint32_t i = 0; i < N; i += 3) {
                        const auto p = (i == 0);
                        swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
                        swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], p);
                        swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
                    }
                    const auto p = ((lane_id & 1) == 0);
                    for(uint32_t i = 0; i < 3; i++) {
                        uint32_t j = i + 3;
                        swap_if_needed(k[i], v[i], k[j], v[j], p);
                    }
                    for(uint32_t i = 0; i < N; i += 3) {
                        swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
                        swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], p);
                        swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
                    }
                    return;
                }

                const uint32_t b = range;
                for(uint32_t c = b / 2; c >= 1; c >>= 1) {
                    const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
                    for(uint32_t i = 0; i < N; i++) {
                        swap_if_needed(k[i], v[i], c, p);
                    }
                }
                const auto p = ((lane_id & b) == 0);
                for(uint32_t i = 0; i < 3; i++) {
                    uint32_t j = i + 3;
                    swap_if_needed(k[i], v[i], k[j], v[j], p);
                }
                for(uint32_t i = 0; i < N; i += N / 2) {
                    swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
                    swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], p);
                    swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
                }
            }
        };

        template<class K, class V>
        struct warp_merge_core<K, V, 3> {
            __device__ inline void operator()(K k[3], V v[3], const uint32_t range, const bool asc) {
                constexpr uint32_t N=3;
                const auto lane_id = threadIdx.x % warpSize;

                if (range == 1) {
                    const auto p = ((lane_id & 1) == 0);
                    swap_if_needed(k[0], v[0], k[1], v[1], p);
                    swap_if_needed(k[1], v[1], k[2], v[2], p);
                    swap_if_needed(k[0], v[0], k[1], v[1], p);
                    return;
                }

                const uint32_t b = range;
                for(uint32_t c = b / 2; c >= 1; c >>= 1) {
                    const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
                    for(uint32_t i = 0; i < N; i++) {
                        swap_if_needed(k[i], v[i], c, p);
                    }
                }
                const auto p = ((lane_id & b) == 0);
                swap_if_needed(k[0], v[0], k[1], v[1], p);
                swap_if_needed(k[1], v[1], k[2], v[2], p);
                swap_if_needed(k[0], v[0], k[1], v[1], p);
            }
        };

        template<class K, class V>
        struct warp_merge_core<K, V, 2> {
            __device__ inline void operator()(K k[2], V v[2], const uint32_t range, const bool asc) {
                constexpr uint32_t N=2;
                const auto lane_id = threadIdx.x % warpSize;

                if (range == 1) {
                    const auto p = ((lane_id & 1) == 0);
                    swap_if_needed(k[0], v[0], k[1], v[1], p);
                    return;
                }

                const uint32_t b = range;
                for(uint32_t c = b / 2; c >= 1; c >>= 1) {
                    const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
                    for(uint32_t i = 0; i < N; i++) {
                        swap_if_needed(k[i], v[i], c, p);
                    }
                }
                const auto p = ((lane_id & b) == 0);
                swap_if_needed(k[0], v[0], k[1], v[1], p);
            }
        };

        template<class K, class V>
        struct warp_merge_core<K, V, 1> {
            __device__ inline void operator()(K k[1], V v[1], const uint32_t range, const bool asc) {
                const auto lane_id = threadIdx.x % warpSize;
                const uint32_t b = range;
                for(uint32_t c = b / 2; c >= 1; c >>= 1) {
                    const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
                    swap_if_needed(k[0], v[0], c, p);
                }
            }
        };

    }

    template<class K, class V, uint32_t N>
    __device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true) {
        detail::warp_merge_core<K, V, N>{}(k, v, range, asc);
    }

    template<class K, class V, uint32_t N>
    __device__ void warp_sort(K k[N], V v[N], const bool asc = true) {
        for (std::uint32_t range = 1; range <= warpSize; range <<= 1) {
            warp_merge<K, V, N>(k, v, range, asc);
        }
    }

}  // namespace bitonic

