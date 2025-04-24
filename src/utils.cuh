#include <cfloat>

#pragma once

struct index_mask {
    static constexpr uint32_t explored = static_cast<uint32_t>(1) << (sizeof(uint32_t) * 8 - 1);
    static constexpr uint32_t clean = ~explored;
};

__host__ __device__ constexpr inline uint32_t warp_size() { return 32; }

__host__ __device__ constexpr inline uint32_t warp_full_mask() { return 0xffffffff; }

__host__ __device__ inline uint32_t xorshift32(uint32_t u) {
    u ^= u >> 13;
    u ^= u << 27;
    u ^= u >> 5;
    return u * 0x9E3779B9;
}

static constexpr __host__ __device__ __forceinline__ uint32_t roundUp32(uint32_t p) noexcept {
    return (p + 31) & (~31);
}

template<typename IntType>
constexpr inline __host__ __device__ IntType ceildiv(IntType a, IntType b) {
    return (a + b - 1) / b;
}

template<class T, unsigned X_MAX = 1024>
__host__ __device__ inline T swizzling(T x) {
    if constexpr (X_MAX <= 1024) {
        return (x) ^ ((x) >> 5);
    } else {
        return (x) ^ (((x) >> 5) & 0x1f);
    }
}

template<typename Type>
__host__ __device__ inline Type get_max_value();

template<>
__host__ __device__ inline int get_max_value<int>() {
    return INT_MAX;
};

template<>
__host__ __device__ inline float get_max_value<float>() {
    return FLT_MAX;
};

template<>
__host__ __device__ inline uint32_t get_max_value<uint32_t>() {
    return 0xffffffffu;
};