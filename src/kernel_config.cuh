
#pragma once

#include "partition_kernel.cuh"
#include "graph_kernel.cuh"
#include "search_kernel.cuh"

inline
uint32_t set_block_size(raft::resources const& handle, uint32_t graph_degree,
                    uint32_t thread_block_size, uint32_t shared_mem_size, uint32_t max_jobs) {
    constexpr uint32_t min_block_size = 64;
    constexpr uint32_t max_block_size = 512;

    uint32_t block_size = thread_block_size;
    if (block_size == 0) {
        block_size = min_block_size;

        constexpr unsigned ulimit_smem_size_cta32 = 4096;
        while (shared_mem_size > ulimit_smem_size_cta32 / 32 * block_size) {
            block_size *= 2;
        }

        cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
        while ((block_size < max_block_size) &&
               (graph_degree * warp_size() >= block_size * 2) &&
               (max_jobs <= (1024 / (block_size * 2)) * deviceProp.multiProcessorCount)) {
            block_size *= 2;
        }
    }
    return block_size;
}

struct define_partition_kernel_config {
    using kernel_t = decltype(&define_partition_kernel<32>);

    static auto choose_kernel(uint32_t centroids_num) -> kernel_t
    {
        if (centroids_num <= 32) {
            return define_partition_kernel<32>;
        } else if (centroids_num <= 64) {
            return define_partition_kernel<64>;
        } else if (centroids_num <= 128) {
            return define_partition_kernel<128>;
        } else if (centroids_num <= 256) {
            return define_partition_kernel<256>;
        } else {
            throw std::invalid_argument("Unsupported centroids_num");
        }
    }
};

struct define_partition_for_large_kernel_config {
    using kernel_t = decltype(&define_partition_for_large_kernel<32>);

    static auto choose_kernel(uint32_t centroid_num) -> kernel_t {
        if (centroid_num <= 32) {
            return define_partition_for_large_kernel<32>;
        } else if (centroid_num <= 64) {
            return define_partition_for_large_kernel<64>;
        } else if (centroid_num <= 128) {
            return define_partition_for_large_kernel<128>;
        } else if (centroid_num <= 256) {
            return define_partition_for_large_kernel<256>;
        } else {
            throw std::invalid_argument("Unsupported centroid_num");
        }
    }
};

struct build_sub_kernel_config {
    using kernel_t = decltype(&build_sub_kernel<64, 64>);

    template <uint32_t MAX_BEAM>
    static auto choose_kernel_candidate(uint32_t graph_degree) -> kernel_t {
        if (graph_degree <= 64) {
            return build_sub_kernel<MAX_BEAM, 64>;
        } else if (graph_degree <= 128) {
            return build_sub_kernel<MAX_BEAM, 128>;
        } else if (graph_degree <= 256) {
            return build_sub_kernel<MAX_BEAM, 256>;
        } else {
            throw std::invalid_argument("Unsupported candidate_size");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree) -> kernel_t {
        if (beam <= 64) {
            return choose_kernel_candidate<64>(graph_degree);
        } else if (beam <= 128) {
            return choose_kernel_candidate<128>(graph_degree);
        } else if (beam <= 256) {
            return choose_kernel_candidate<256>(graph_degree);
        } else if (beam <= 512) {
            return choose_kernel_candidate<512>(graph_degree);
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }
};

struct build_boundary_kernel_config {
    using kernel_t = decltype(&build_boundary_kernel<64, 64>);

    template <uint32_t MAX_BEAM>
    static auto choose_kernel_candidate(uint32_t graph_degree) -> kernel_t {
        if (graph_degree <= 64) {
            return build_boundary_kernel<MAX_BEAM, 64>;
        } else if (graph_degree <= 128) {
            return build_boundary_kernel<MAX_BEAM, 128>;
        } else if (graph_degree <= 256) {
            return build_boundary_kernel<MAX_BEAM, 256>;
        } else {
            throw std::invalid_argument("Unsupported candidate_size");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree) -> kernel_t {
        if (beam <= 64) {
            return choose_kernel_candidate<64>(graph_degree);
        } else if (beam <= 128) {
            return choose_kernel_candidate<128>(graph_degree);
        } else if (beam <= 256) {
            return choose_kernel_candidate<256>(graph_degree);
        } else if (beam <= 512) {
            return choose_kernel_candidate<512>(graph_degree);
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }
};

struct build_for_large_kernel_config {
    using kernel_t = decltype(&build_for_large_kernel<64, 64>);

    template <uint32_t MAX_BEAM>
    static auto choose_kernel_candidate(uint32_t graph_degree) -> kernel_t {
        if (graph_degree <= 64) {
            return build_for_large_kernel<MAX_BEAM, 64>;
        } else if (graph_degree <= 128) {
            return build_for_large_kernel<MAX_BEAM, 128>;
        } else if (graph_degree <= 256) {
            return build_for_large_kernel<MAX_BEAM, 256>;
        } else {
            throw std::invalid_argument("Unsupported candidate_size");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree) -> kernel_t {
        if (beam <= 64) {
            return choose_kernel_candidate<64>(graph_degree);
        } else if (beam <= 128) {
            return choose_kernel_candidate<128>(graph_degree);
        } else if (beam <= 256) {
            return choose_kernel_candidate<256>(graph_degree);
        } else if (beam <= 512) {
            return choose_kernel_candidate<512>(graph_degree);
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }
};

struct refine_kernel_config {
    using kernel_t = decltype(&refine_kernel<64, 64, 32>);

    template <uint32_t MAX_BEAM, uint32_t MAX_CANDIDATE>
    static auto choose_kernel_centroid(uint32_t centroid_num) -> kernel_t {
        if (centroid_num <= 32) {
            return refine_kernel<MAX_BEAM, MAX_CANDIDATE, 32>;
        } else if (centroid_num <= 64) {
            return refine_kernel<MAX_BEAM, MAX_CANDIDATE, 64>;
        } else if (centroid_num <= 128) {
            return refine_kernel<MAX_BEAM, MAX_CANDIDATE, 128>;
        } else {
            throw std::invalid_argument("Unsupported centroid_num");
        }
    }

    template <uint32_t MAX_BEAM>
    static auto choose_kernel_candidate(uint32_t graph_degree, uint32_t centroid_num) -> kernel_t {
        if (graph_degree <= 64) {
            return choose_kernel_centroid<MAX_BEAM, 64>(centroid_num);
        } else if (graph_degree <= 128) {
            return choose_kernel_centroid<MAX_BEAM, 128>(centroid_num);
        } else if (graph_degree <= 256) {
            return choose_kernel_centroid<MAX_BEAM, 256>(centroid_num);
        } else {
            throw std::invalid_argument("Unsupported candidate_size");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree, uint32_t centroid_num) -> kernel_t {
        if (beam <= 64) {
            return choose_kernel_candidate<64>(graph_degree, centroid_num);
        } else if (beam <= 128) {
            return choose_kernel_candidate<128>(graph_degree, centroid_num);
        } else if (beam <= 256) {
            return choose_kernel_candidate<256>(graph_degree, centroid_num);
        } else if (beam <= 512) {
            return choose_kernel_candidate<512>(graph_degree, centroid_num);
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }
};

struct refine_for_large_kernel_config {
    using kernel_t = decltype(&refine_for_large_kernel<64, 64>);

    template <uint32_t MAX_BEAM>
    static auto choose_kernel_candidate(uint32_t graph_degree) -> kernel_t {
        if (graph_degree <= 64) {
            return refine_for_large_kernel<MAX_BEAM, 64>;
        } else if (graph_degree <= 128) {
            return refine_for_large_kernel<MAX_BEAM, 128>;
        } else if (graph_degree <= 256) {
            return refine_for_large_kernel<MAX_BEAM, 256>;
        } else {
            throw std::invalid_argument("Unsupported candidate_size");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree) -> kernel_t {
        if (beam <= 64) {
            return choose_kernel_candidate<64>(graph_degree);
        } else if (beam <= 128) {
            return choose_kernel_candidate<128>(graph_degree);
        } else if (beam <= 256) {
            return choose_kernel_candidate<256>(graph_degree);
        } else if (beam <= 512) {
            return choose_kernel_candidate<512>(graph_degree);
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }
};

struct search_kernel_config {
    using kernel_t = decltype(&search_kernel<64, 64, 32>);

    template <uint32_t MAX_BEAM, uint32_t MAX_CANDIDATE>
    static auto choose_kernel_centroid(uint32_t centroid_num) -> kernel_t {
        if (centroid_num <= 32) {
            return search_kernel<MAX_BEAM, MAX_CANDIDATE, 32>;
        } else if (centroid_num <= 64) {
            return search_kernel<MAX_BEAM, MAX_CANDIDATE, 64>;
        } else if (centroid_num <= 128) {
            return search_kernel<MAX_BEAM, MAX_CANDIDATE, 128>;
        } else {
            throw std::invalid_argument("Unsupported centroid_num");
        }
    }

    template <uint32_t MAX_BEAM>
    static auto choose_kernel_candidate(uint32_t graph_degree, uint32_t centroid_num) -> kernel_t {
        if (graph_degree <= 64) {
            return choose_kernel_centroid<MAX_BEAM, 64>(centroid_num);
        } else if (graph_degree <= 128) {
            return choose_kernel_centroid<MAX_BEAM, 128>(centroid_num);
        } else if (graph_degree <= 256) {
            return choose_kernel_centroid<MAX_BEAM, 256>(centroid_num);
        } else {
            throw std::invalid_argument("Unsupported candidate_size");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree, uint32_t centroid_num) -> kernel_t {
        if (beam <= 64) {
            return choose_kernel_candidate<64>(graph_degree, centroid_num);
        } else if (beam <= 128) {
            return choose_kernel_candidate<128>(graph_degree, centroid_num);
        } else if (beam <= 256) {
            return choose_kernel_candidate<256>(graph_degree, centroid_num);
        } else if (beam <= 512) {
            return choose_kernel_candidate<512>(graph_degree, centroid_num);
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }
};

struct select_segment_kernel_config {
    using kernel_t = decltype(&select_segment_kernel<32>);

    static auto choose_kernel(uint32_t centroid_num) -> kernel_t {
        if (centroid_num <= 32) {
            return select_segment_kernel<32>;
        } else if (centroid_num <= 64) {
            return select_segment_kernel<64>;
        } else if (centroid_num <= 128) {
            return select_segment_kernel<128>;
        } else if (centroid_num <= 256) {
            return select_segment_kernel<256>;
        } else {
            throw std::invalid_argument("Unsupported centroid_num");
        }
    }
};

struct search_sub_kernel_config {
    using kernel_t = decltype(&search_sub_kernel<64, 64, 32>);

    template <uint32_t MAX_BEAM, uint32_t MAX_CANDIDATE>
    static auto choose_kernel_topk(uint32_t topk) -> kernel_t {
        if (topk <= 32) {
            return search_sub_kernel<MAX_BEAM, MAX_CANDIDATE, 32>;
        } else if (topk <= 64) {
            return search_sub_kernel<MAX_BEAM, MAX_CANDIDATE, 64>;
        } else if (topk <= 128) {
            return search_sub_kernel<MAX_BEAM, MAX_CANDIDATE, 128>;
        } else {
            throw std::invalid_argument("Unsupported centroid_num");
        }
    }

    template <uint32_t MAX_BEAM>
    static auto choose_kernel_candidate(uint32_t graph_degree, uint32_t topk) -> kernel_t {
        if (graph_degree <= 64) {
            return choose_kernel_topk<MAX_BEAM, 64>(topk);
        } else if (graph_degree <= 128) {
            return choose_kernel_topk<MAX_BEAM, 128>(topk);
        } else if (graph_degree <= 256) {
            return choose_kernel_topk<MAX_BEAM, 256>(topk);
        } else {
            throw std::invalid_argument("Unsupported candidate_size");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree, uint32_t topk) -> kernel_t {
        if (beam <= 64) {
            return choose_kernel_candidate<64>(graph_degree, topk);
        } else if (beam <= 128) {
            return choose_kernel_candidate<128>(graph_degree, topk);
        } else if (beam <= 256) {
            return choose_kernel_candidate<256>(graph_degree, topk);
        } else if (beam <= 512) {
            return choose_kernel_candidate<512>(graph_degree, topk);
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }
};