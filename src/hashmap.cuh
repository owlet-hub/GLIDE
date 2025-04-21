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


namespace hashmap {

    __host__ __device__ inline uint32_t get_size(const uint32_t bitlen) { return 1U << bitlen; }

    __device__ inline void init(uint32_t *const table, const unsigned bitlen, unsigned FIRST_TID = 0) {
        if (threadIdx.x < FIRST_TID) return;
        for (unsigned i = threadIdx.x - FIRST_TID; i < get_size(bitlen); i += blockDim.x - FIRST_TID) {
            table[i] = 0xffffffffu;
        }
    }


    __device__ inline uint32_t insert(uint32_t *const table, const uint32_t bitlen, const uint32_t key) {
        // Open addressing is used for collision resolution
        const uint32_t size = get_size(bitlen);
        const uint32_t bit_mask = size - 1;
#if 1
        // Linear probing
        uint32_t index = (key ^ (key >> bitlen)) & bit_mask;
        constexpr uint32_t stride = 1;
#else
        // Double hashing
  uint32_t index        = key & bit_mask;
  const uint32_t stride = (key >> bitlen) * 2 + 1;
#endif
        for (unsigned i = 0; i < size; i++) {
            const uint32_t old = atomicCAS(&table[index], ~static_cast<uint32_t>(0), key);
            if (old == ~static_cast<uint32_t>(0)) {
                return 1;
            } else if (old == key) {
                return 0;
            }
            index = (index + stride) & bit_mask;
        }
        return 0;
    }
}  // namespace hashmap


namespace bloomfilter {

    __device__ inline uint32_t num_bits_per_set(uint32_t vector_number) { return vector_number; }

    __device__ inline uint32_t hash_0(uint32_t value) {
        value *= 1103515245;
        value += 12345;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        return value;
    }

    __device__ inline uint32_t hash_1(uint32_t value) {
        value *= 1664525;
        value += 1013904223;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        return value;
    }

    __device__ inline void add(bool *filter, uint32_t key, uint32_t vector_number, uint32_t num_hashs) {
        uint32_t hash = bloomfilter::hash_0(key);
        if(!filter[hash % num_bits_per_set(vector_number)]){
            filter[hash % num_bits_per_set(vector_number)] = true;
        }
        for (uint32_t i = 1; i < num_hashs; i++) {
            hash = hash + bloomfilter::hash_1(key);
            if(!filter[hash % num_bits_per_set(vector_number)]){
                filter[hash % num_bits_per_set(vector_number)] = true;
            }
        }
    }

    __device__ inline bool check(bool *filter, uint32_t key, uint32_t vector_number, uint32_t num_hashs) {
        bool is_present = true;
        uint32_t hash = hash_0(key);
        is_present &= filter[hash % num_bits_per_set(vector_number)];

        if (!is_present) return false;
        for (uint32_t i = 1; i < num_hashs; i++) {
            hash = hash + hash_1(key);
            is_present &= filter[hash % num_bits_per_set(vector_number)];
            if (!is_present) return false;
        }
        return true;
    }

    __device__ inline void clear(bool *filter, uint32_t vector_number) {
        for (uint32_t i = threadIdx.x; i < vector_number; i+=blockDim.x) {
            filter[i] = false;
        }
    }
}
