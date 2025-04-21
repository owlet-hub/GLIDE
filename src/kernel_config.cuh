
#pragma once

#include "partition_kernel.cuh"
#include "graph_kernel.cuh"
#include "search_kernel.cuh"

uint32_t set_dataset_block_dim(uint32_t dim){
    constexpr uint32_t max_dataset_block_dim = 512;
    uint32_t dataset_block_dim = 128;
    while (dataset_block_dim < dim && dataset_block_dim < max_dataset_block_dim) {
        dataset_block_dim *= 2;
    }
    return dataset_block_dim;
}

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

struct build_kernel_config {
    using kernel_t = decltype(&build_kernel<64, 64>);

    template <uint32_t MAX_CANDIDATE>
    static auto choose_kernel_search_intermediate_size(uint32_t beam) -> kernel_t
    {
        if (beam <= 64) {
            return build_kernel<MAX_CANDIDATE, 64>;
        } else if (beam <= 128) {
            return build_kernel<MAX_CANDIDATE, 128>;
        } else if (beam <= 256) {
            return build_kernel<MAX_CANDIDATE, 256>;
        } else if (beam <= 512) {
            return build_kernel<MAX_CANDIDATE, 512>;
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree) -> kernel_t
    {
        if (graph_degree <= 64) {
            return choose_kernel_search_intermediate_size<64>(beam);
        } else if (graph_degree <= 128) {
            return choose_kernel_search_intermediate_size<128>(beam);
        } else if (graph_degree <= 256) {
            return choose_kernel_search_intermediate_size<256>(beam);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }
};

struct build_boundary_kernel_config {
    using kernel_t = decltype(&build_boundary_kernel<64, 64>);

    template <uint32_t MAX_CANDIDATE>
    static auto choose_kernel_search_intermediate_size(uint32_t beam) -> kernel_t
    {
        if (beam <= 64) {
            return build_boundary_kernel<MAX_CANDIDATE, 64>;
        } else if (beam <= 128) {
            return build_boundary_kernel<MAX_CANDIDATE, 128>;
        } else if (beam <= 256) {
            return build_boundary_kernel<MAX_CANDIDATE, 256>;
        } else if (beam <= 512) {
            return build_boundary_kernel<MAX_CANDIDATE, 512>;
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree) -> kernel_t
    {
        if (graph_degree <= 64) {
            return choose_kernel_search_intermediate_size<64>(beam);
        } else if (graph_degree <= 128) {
            return choose_kernel_search_intermediate_size<128>(beam);
        } else if (graph_degree <= 256) {
            return choose_kernel_search_intermediate_size<256>(beam);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }
};

struct refine_for_large_kernel_config {
    using kernel_t = decltype(&refine_for_large_kernel<64, 64>);

    template <uint32_t MAX_CANDIDATE>
    static auto choose_kernel_search_intermediate_size(uint32_t beam) -> kernel_t
    {
        if (beam <= 64) {
            return refine_for_large_kernel<MAX_CANDIDATE, 64>;
        } else if (beam <= 128) {
            return refine_for_large_kernel<MAX_CANDIDATE, 128>;
        } else if (beam <= 256) {
            return refine_for_large_kernel<MAX_CANDIDATE, 256>;
        } else if (beam <= 512) {
            return refine_for_large_kernel<MAX_CANDIDATE, 512>;
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree) -> kernel_t
    {
        if (graph_degree <= 64) {
            return choose_kernel_search_intermediate_size<64>(beam);
        } else if (graph_degree <= 128) {
            return choose_kernel_search_intermediate_size<128>(beam);
        } else if (graph_degree <= 256) {
            return choose_kernel_search_intermediate_size<256>(beam);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }
};

struct refine_kernel_config {
    using kernel_t = decltype(&refine_kernel<64, 64, 32>);

    template <uint32_t MAX_CENTROID, uint32_t MAX_CANDIDATE>
    static auto choose_kernel_topk_size(uint32_t beam) -> kernel_t
    {
        if (beam <= 64) {
            return refine_kernel<MAX_CANDIDATE, 64, MAX_CENTROID>;
        } else if (beam <= 128) {
            return refine_kernel<MAX_CANDIDATE, 128, MAX_CENTROID>;
        } else if (beam <= 256) {
            return refine_kernel<MAX_CANDIDATE, 256, MAX_CENTROID>;
        } else if (beam <= 512) {
            return refine_kernel<MAX_CANDIDATE, 512, MAX_CENTROID>;
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }

    template <uint32_t MAX_CENTROID>
    static auto choose_kernel_candidate_size(uint32_t graph_degree, uint32_t beam) -> kernel_t
    {
        if (graph_degree <= 64) {
            return choose_kernel_topk_size<MAX_CENTROID, 64>(beam);
        } else if (graph_degree <= 128) {
            return choose_kernel_topk_size<MAX_CENTROID, 128>(beam);
        } else if (graph_degree <= 256) {
            return choose_kernel_topk_size<MAX_CENTROID, 256>(beam);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree, uint32_t centroid_num) -> kernel_t
    {
        if (centroid_num <= 32) {
            return choose_kernel_candidate_size<32>(graph_degree, beam);
        } else if (graph_degree <= 64) {
            return choose_kernel_candidate_size<64>(graph_degree, beam);
        } else if (graph_degree <= 128) {
            return choose_kernel_candidate_size<128>(graph_degree, beam);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }
};

struct search_kernel_config {
    using kernel_t = decltype(&search_kernel<64, 64, 32>);

    template <uint32_t MAX_CENTROID, uint32_t MAX_CANDIDATE>
    static auto choose_kernel_topk_size(uint32_t beam) -> kernel_t
    {
        if (beam <= 64) {
            return search_kernel<MAX_CANDIDATE, 64, MAX_CENTROID>;
        } else if (beam <= 128) {
            return search_kernel<MAX_CANDIDATE, 128, MAX_CENTROID>;
        } else if (beam <= 256) {
            return search_kernel<MAX_CANDIDATE, 256, MAX_CENTROID>;
        } else if (beam <= 512) {
            return search_kernel<MAX_CANDIDATE, 512, MAX_CENTROID>;
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }

    template <uint32_t MAX_CENTROID>
    static auto choose_kernel_candidate_size(uint32_t graph_degree, uint32_t beam) -> kernel_t
    {
        if (graph_degree <= 64) {
            return choose_kernel_topk_size<MAX_CENTROID, 64>(beam);
        } else if (graph_degree <= 128) {
            return choose_kernel_topk_size<MAX_CENTROID, 128>(beam);
        } else if (graph_degree <= 256) {
            return choose_kernel_topk_size<MAX_CENTROID, 256>(beam);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree, uint32_t centroid_num) -> kernel_t
    {
        if (centroid_num <= 32) {
            return choose_kernel_candidate_size<32>(graph_degree, beam);
        } else if (graph_degree <= 64) {
            return choose_kernel_candidate_size<64>(graph_degree, beam);
        } else if (graph_degree <= 128) {
            return choose_kernel_candidate_size<128>(graph_degree, beam);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }
};

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

struct define_partition_for_large_dataset_kernel_config {
    using kernel_t = decltype(&define_partition_for_large_dataset_kernel<32>);

    static auto choose_kernel(uint32_t centroids_num) -> kernel_t
    {
        if (centroids_num <= 32) {
            return define_partition_for_large_dataset_kernel<32>;
        } else if (centroids_num <= 64) {
            return define_partition_for_large_dataset_kernel<64>;
        } else if (centroids_num <= 128) {
            return define_partition_for_large_dataset_kernel<128>;
        } else if (centroids_num <= 256) {
            return define_partition_for_large_dataset_kernel<256>;
        } else {
            throw std::invalid_argument("Unsupported centroids_num");
        }
    }
};

struct build_for_large_dataset_kernel_config {
    using kernel_t = decltype(&build_for_large_dataset_kernel<64, 64>);

    template <uint32_t MAX_CANDIDATE>
    static auto choose_kernel_search_intermediate_size(uint32_t beam) -> kernel_t
    {
        if (beam <= 64) {
            return build_for_large_dataset_kernel<MAX_CANDIDATE, 64>;
        } else if (beam <= 128) {
            return build_for_large_dataset_kernel<MAX_CANDIDATE, 128>;
        } else if (beam <= 256) {
            return build_for_large_dataset_kernel<MAX_CANDIDATE, 256>;
        } else if (beam <= 512) {
            return build_for_large_dataset_kernel<MAX_CANDIDATE, 512>;
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree) -> kernel_t
    {
        if (graph_degree <= 64) {
            return choose_kernel_search_intermediate_size<64>(beam);
        } else if (graph_degree <= 128) {
            return choose_kernel_search_intermediate_size<128>(beam);
        } else if (graph_degree <= 256) {
            return choose_kernel_search_intermediate_size<256>(beam);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }
};

struct select_segment_kernel_config {
    using kernel_t = decltype(&select_segment_kernel<32>);

    static auto choose_kernel(uint32_t centroid_size) -> kernel_t
    {
        if (centroid_size <= 32) {
            return select_segment_kernel<32>;
        } else if (centroid_size <= 64) {
            return select_segment_kernel<64>;
        } else if (centroid_size <= 128) {
            return select_segment_kernel<128>;
        } else if (centroid_size <= 256) {
            return select_segment_kernel<256>;
        } else {
            throw std::invalid_argument("Unsupported centroid_size");
        }
    }
};

struct search_on_sub_kernel_config {
    using kernel_t = decltype(&search_on_sub_kernel<64, 64, 32>);

    template <uint32_t MAX_CANDIDATE, uint32_t MAX_BEAM>
    static auto choose_kernel_topk(uint32_t topk) -> kernel_t
    {
        if (topk <= 32) {
            return search_on_sub_kernel<MAX_CANDIDATE, MAX_BEAM, 32>;
        } else if (topk <= 64) {
            return search_on_sub_kernel<MAX_CANDIDATE, MAX_BEAM, 64>;
        } else if (topk <= 128) {
            return search_on_sub_kernel<MAX_CANDIDATE, MAX_BEAM, 128>;
        } else {
            throw std::invalid_argument("Unsupported topk");
        }
    }

    template <uint32_t MAX_CANDIDATE>
    static auto choose_kernel_beam(uint32_t beam, uint32_t topk) -> kernel_t
    {
        if (beam <= 64) {
            return choose_kernel_topk<MAX_CANDIDATE, 64>(topk);
        } else if (beam <= 128) {
            return choose_kernel_topk<MAX_CANDIDATE, 128>(topk);
        } else if (beam <= 256) {
            return choose_kernel_topk<MAX_CANDIDATE, 256>(topk);
        } else if (beam <= 512) {
            return choose_kernel_topk<MAX_CANDIDATE, 512>(topk);
        } else {
            throw std::invalid_argument("Unsupported beam");
        }
    }

    static auto choose_kernel(uint32_t beam, uint32_t graph_degree, uint32_t topk) -> kernel_t
    {
        if (graph_degree <= 64) {
            return choose_kernel_beam<64>(beam, topk);
        } else if (graph_degree <= 128) {
            return choose_kernel_beam<128>(beam, topk);
        } else if (graph_degree <= 256) {
            return choose_kernel_beam<256>(beam, topk);
        } else {
            throw std::invalid_argument("Unsupported graph_degree");
        }
    }
};