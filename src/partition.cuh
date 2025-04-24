#include <cuvs/cluster/kmeans.hpp>
#include <random>
#include <thrust/execution_policy.h>
#include "partition_kernel.cuh"
#include "params.cuh"
#include "kernel_config.cuh"

void
get_centroids(raft::device_resources &handle,
              raft::device_matrix_view<float> d_data_view,
              raft::device_matrix_view<float, int> d_centroid_view,
              uint32_t cluster_num, float sample_factor, float &build_time) {
    using namespace cuvs::cluster;
    uint32_t num = d_data_view.extent(0);
    uint32_t dim = d_data_view.extent(1);
    int sample_num = num * sample_factor;

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float milliseconds = 0;

    cudaEventRecord(start_time);
    auto d_sample_data = raft::make_device_matrix<float, int>(handle, sample_num, dim);
    std::vector<uint32_t> indices(num);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);
    auto d_indices = raft::make_device_vector<uint32_t>(handle, sample_num);
    raft::copy(d_indices.data_handle(), indices.data(),
               sample_num, raft::resource::get_cuda_stream(handle));

    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    uint32_t max_size = deviceProp.maxThreadsPerBlock;
    uint32_t threads_per_block = std::min(dim, max_size);
    uint32_t blocks_per_grid = sample_num;
    cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

    sample_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_data_view.data_handle(),
                                                                     d_sample_data.data_handle(),
                                                                     d_indices.data_handle(),
                                                                     dim,
                                                                     sample_num);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    kmeans::params params;
    params.n_clusters = cluster_num;
    int iter;
    float inertia;

    kmeans::fit(handle, params, d_sample_data.view(), std::nullopt, d_centroid_view,
                raft::make_host_scalar_view(&inertia), raft::make_host_scalar_view(&iter));
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

void
define_partition(raft::device_resources &handle,
                 raft::device_matrix_view<float> d_data_view,
                 raft::device_matrix_view<float> d_centroid_view,
                 raft::host_vector_view<uint32_t> h_segment_start_view,
                 raft::host_vector_view<uint32_t> h_segment_length_view,
                 raft::device_matrix_view<uint32_t> d_segment_index_view,
                 float boundary_fact, Metric metric, float &build_time) {
    uint32_t centroid_num = d_centroid_view.extent(0);
    uint32_t num = d_data_view.extent(0);
    uint32_t dim = d_data_view.extent(1);

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float milliseconds = 0;

    auto d_segment_length = raft::make_device_vector<uint32_t>(handle, h_segment_length_view.size());
    thrust::fill(thrust::device, d_segment_length.data_handle(),
                 d_segment_length.data_handle() + d_segment_length.size(), 0);

    uint32_t shared_mem_size = dim * sizeof(float) + centroid_num * sizeof(float) + centroid_num * sizeof(uint32_t);
    uint32_t threads_per_block = 256;
    uint32_t blocks_per_grid = num;

    cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);
    auto kernel = define_partition_kernel_config::choose_kernel(centroid_num);

    cudaEventRecord(start_time);
    kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(
            d_data_view.data_handle(),
            d_centroid_view.data_handle(),
            num,
            dim,
            centroid_num,
            d_segment_length.data_handle(),
            d_segment_index_view.data_handle(),
            boundary_fact,
            metric);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;

    raft::copy(h_segment_length_view.data_handle(), d_segment_length.data_handle(),
               d_segment_length.size(), raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);

    uint32_t start = 0;
    for (uint32_t i = 0; i < h_segment_length_view.size(); i++) {
        h_segment_start_view(i) = start;
        start += h_segment_length_view(i);
    }
}

void
reorder(raft::device_resources &handle,
        raft::device_matrix_view<float> d_data_view,
        raft::device_vector_view<uint32_t> d_map_view,
        raft::device_matrix_view<float> d_reorder_data_view,
        raft::host_vector_view<uint32_t> h_segment_start_view,
        raft::device_matrix_view<uint32_t> d_segment_index_view,
        float &build_time) {
    uint32_t num = d_data_view.extent(0);
    uint32_t dim = d_data_view.extent(1);

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float milliseconds = 0;

    uint32_t threads_per_block = 128;
    uint32_t blocks_per_grid = num;
    cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

    cudaEventRecord(start_time);
    auto d_segment_start = raft::make_device_vector<uint32_t>(handle, h_segment_start_view.size());
    raft::copy(d_segment_start.data_handle(), h_segment_start_view.data_handle(),
               h_segment_start_view.size(), raft::resource::get_cuda_stream(handle));

    reorder_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_data_view.data_handle(),
                                                                      d_reorder_data_view.data_handle(),
                                                                      d_segment_index_view.data_handle(),
                                                                      d_map_view.data_handle(),
                                                                      d_segment_start.data_handle(),
                                                                      num,
                                                                      dim);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

void preprocess(raft::device_resources &handle, PartitionParameter param,
                raft::device_matrix_view<float> d_data_view,
                std::optional<raft::device_matrix<float>> &d_reorder_data,
                std::optional<raft::device_vector<uint32_t>> &d_map,
                raft::host_vector_view<uint32_t> h_segment_start_view,
                raft::host_vector_view<uint32_t> h_segment_length_view,
                raft::device_matrix_view<float> d_centroid_view,
                std::string &result_file) {
    uint32_t num = d_data_view.extent(0);
    uint32_t dim = d_data_view.extent(1);

    float build_time = 0.0f;

    auto d_segment_index = raft::make_device_matrix<uint32_t, uint32_t>(handle, num, 3);

    get_centroids(handle, d_data_view, d_centroid_view, param.centroid_num, param.sample_factor, build_time);

    define_partition(handle, d_data_view, d_centroid_view, h_segment_start_view, h_segment_length_view,
                     d_segment_index.view(), param.boundary_factor, param.metric, build_time);


    uint32_t reorder_num = num + h_segment_length_view(param.centroid_num);
    d_reorder_data.emplace(raft::make_device_matrix<float, uint32_t>(handle, reorder_num, dim));
    d_map.emplace(raft::make_device_vector<uint32_t>(handle, reorder_num));

    reorder(handle, d_data_view, d_map->view(), d_reorder_data->view(),
            h_segment_start_view, d_segment_index.view(), build_time);

    std::cout << "preprocessing time: " << build_time << " s" << std::endl;
    std::ofstream result_out(result_file, std::ios::app);
    result_out << build_time << ",";
    result_out.close();
}

void get_centroids_for_large(raft::device_resources &handle,
                             raft::host_matrix_view<uint8_t, uint64_t> h_data_view,
                             raft::device_matrix_view<uint8_t, uint64_t> d_data_view,
                             raft::device_matrix_view<float, int> d_centroid_view,
                             uint32_t cluster_num, float sample_factor, uint32_t batch_num, float &build_time) {
    using namespace cuvs::cluster;
    uint32_t num = h_data_view.extent(0);
    uint32_t dim = h_data_view.extent(1);
    int sample_num = num * sample_factor;

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float milliseconds = 0;

    auto d_sample_data = raft::make_device_matrix<float, int>(handle, sample_num, dim);

    cudaEventRecord(start_time);
    std::vector<uint32_t> indices(num);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);
    auto d_indices = raft::make_device_vector<uint32_t>(handle, sample_num);
    raft::copy(d_indices.data_handle(), indices.data(), sample_num,
               raft::resource::get_stream_from_stream_pool(handle));
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;

    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    uint32_t max_size = deviceProp.maxThreadsPerBlock;
    uint32_t threads_per_block = std::min(dim, max_size);
    uint32_t blocks_per_grid = sample_num;

    uint32_t batch_size = num / batch_num;
    float sampling_time = 0.0f;
    for (uint32_t i = 0; i < batch_num; i++) {
        uint32_t start_id = i * batch_size;
        raft::copy(d_data_view.data_handle(), h_data_view.data_handle() + static_cast<uint64_t>(start_id) * dim,
                   static_cast<uint64_t>(batch_size) * dim, raft::resource::get_cuda_stream(handle));

        cudaEventRecord(start_time);
        sample_for_large_kernel<<<blocks_per_grid, threads_per_block>>>(
                d_data_view.data_handle(),
                d_sample_data.data_handle(),
                d_indices.data_handle(),
                dim,
                sample_num,
                start_id,
                batch_size);
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        sampling_time += milliseconds / 1000.0f;
    }
    build_time += sampling_time;

    kmeans::params params;
    params.n_clusters = cluster_num;
    int iter;
    float inertia;

    cudaEventRecord(start_time);
    kmeans::fit(handle, params, d_sample_data.view(), std::nullopt, d_centroid_view,
                raft::make_host_scalar_view(&inertia), raft::make_host_scalar_view(&iter));
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

void define_partition_for_large(raft::device_resources &handle,
                                raft::host_matrix_view<uint8_t, uint64_t> h_data_view,
                                raft::device_matrix_view<uint8_t, uint64_t> d_data_view,
                                raft::device_matrix_view<float, int> d_centroid_view,
                                raft::host_vector_view<uint32_t> segment_start_view,
                                raft::host_vector_view<uint32_t> segment_length_view,
                                raft::device_vector_view<uint32_t> d_segment_index_view,
                                uint32_t batch_num, Metric metric, float &build_time) {
    uint32_t centroid_num = d_centroid_view.extent(0);
    uint32_t num = h_data_view.extent(0);
    uint32_t dim = h_data_view.extent(1);

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float milliseconds = 0;

    auto d_segment_length = raft::make_device_vector<uint32_t>(handle, segment_length_view.size());
    thrust::fill(thrust::device, d_segment_length.data_handle(),
                 d_segment_length.data_handle() + d_segment_length.size(), 0);

    uint32_t shared_mem_size = dim * sizeof(uint8_t) + centroid_num * sizeof(float) + centroid_num * sizeof(uint32_t);

    uint32_t batch_size = num / batch_num;

    uint32_t threads_per_block = 128;
    uint32_t blocks_per_grid = batch_size;

    cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);
    auto kernel = define_partition_for_large_kernel_config::choose_kernel(centroid_num);

    float define_time = 0.0f;
    for (uint32_t i = 0; i < batch_num; i++) {
        uint32_t start_id = i * batch_size;

        raft::copy(d_data_view.data_handle(), h_data_view.data_handle() + static_cast<uint64_t>(start_id) * dim,
                   static_cast<uint64_t>(batch_size) * dim, raft::resource::get_cuda_stream(handle));

        auto d_segment_index_ptr = d_segment_index_view.data_handle() + start_id;

        cudaEventRecord(start_time);
        kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(
                d_data_view.data_handle(),
                d_centroid_view.data_handle(),
                batch_size,
                dim,
                centroid_num,
                d_segment_length.data_handle(),
                d_segment_index_ptr,
                metric);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        define_time += milliseconds / 1000.0f;
    }
    build_time += define_time;

    raft::copy(segment_length_view.data_handle(), d_segment_length.data_handle(),
               segment_length_view.size(), raft::resource::get_stream_from_stream_pool(handle));
    raft::resource::sync_stream(handle);

    uint32_t start = 0;
    for (uint32_t i = 0; i < segment_length_view.size(); i++) {
        segment_start_view(i) = start;
        start += segment_length_view(i);
    }
}

void reorder_for_large_dataset(raft::device_resources &handle,
                               raft::host_matrix_view<uint8_t, uint64_t> h_data_view,
                               raft::device_matrix_view<uint8_t, uint64_t> d_data_view,
                               raft::host_vector_view<uint32_t> segment_start_view,
                               raft::host_vector_view<uint32_t> mapping_view,
                               raft::host_matrix_view<uint8_t, uint64_t> reorder_data_view,
                               raft::device_vector_view<uint32_t> d_segment_index_view,
                               uint32_t batch_num, float &build_time) {
    uint32_t num = h_data_view.extent(0);
    uint32_t dim = h_data_view.extent(1);
    uint32_t batch_size = num / batch_num;
    uint32_t segment = segment_start_view.size();

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    float milliseconds = 0;
    float reorder_time = 0.0f;

    uint32_t threads_per_block = 128;
    uint32_t blocks_per_grid = batch_size;
    cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

    std::vector<uint32_t> current_counts(segment, 0);
    for (uint32_t i = 0; i < batch_num; i++) {
        uint32_t start_id = i * batch_size;

        auto d_counts = raft::make_device_vector<uint32_t>(handle, segment);
        thrust::fill(thrust::device, d_counts.data_handle(),
                     d_counts.data_handle() + d_counts.size(), 0);

        auto d_segment_index_ptr = d_segment_index_view.data_handle() + start_id;
        raft::copy(d_data_view.data_handle(), h_data_view.data_handle() + static_cast<uint64_t>(start_id) * dim,
                   static_cast<uint64_t>(batch_size) * dim, raft::resource::get_cuda_stream(handle));

        for (uint32_t segment_id = 0; segment_id < segment; segment_id++) {
            uint32_t segment_start = segment_start_view(segment_id);

            auto d_batch_data = raft::make_device_matrix<uint8_t, uint64_t>(handle, batch_size, dim);
            auto d_batch_mapping = raft::make_device_vector<uint32_t>(handle, batch_size);

            cudaEventRecord(start_time);
            reorder_for_large_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
                    d_data_view.data_handle(),
                    d_batch_data.data_handle(),
                    d_segment_index_ptr,
                    d_batch_mapping.data_handle(),
                    batch_size,
                    dim,
                    segment_id,
                    d_counts.data_handle(),
                    start_id);
            RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&milliseconds, start_time, stop_time);
            reorder_time += milliseconds / 1000.0f;

            auto h_counts = raft::make_host_vector<uint32_t>(segment);
            raft::copy(h_counts.data_handle(), d_counts.data_handle(), h_counts.size(),
                       raft::resource::get_stream_from_stream_pool(handle));

            raft::copy(mapping_view.data_handle() + segment_start + current_counts[segment_id],
                       d_batch_mapping.data_handle(),
                       h_counts(segment_id),
                       raft::resource::get_cuda_stream(handle));
            raft::copy(reorder_data_view.data_handle() +
                       static_cast<uint64_t>(segment_start + current_counts[segment_id]) * dim,
                       d_batch_data.data_handle(),
                       static_cast<uint64_t>(h_counts(segment_id)) * dim,
                       raft::resource::get_cuda_stream(handle));

            current_counts[segment_id] += h_counts(segment_id);
        }
    }
    build_time += reorder_time;
}

void preprocess_for_large(raft::device_resources &handle, PartitionParameter param,
                          raft::host_matrix_view<uint8_t, uint64_t> data_view,
                          raft::device_matrix_view<uint8_t, uint64_t> d_data_view,
                          raft::host_matrix_view<uint8_t, uint64_t> reorder_data_view,
                          raft::host_vector_view<uint32_t> mapping_view,
                          raft::host_vector_view<uint32_t> segment_start_view,
                          raft::host_vector_view<uint32_t> segment_length_view,
                          raft::device_matrix_view<float> d_centroid_view,
                          uint32_t batch_num, std::string &result_file) {
    uint32_t num = data_view.extent(0);

    float build_time = 0.0f;

    auto d_segment_index = raft::make_device_vector<uint32_t>(handle, num);

    get_centroids_for_large(handle, data_view, d_data_view, d_centroid_view, param.centroid_num, param.sample_factor,
                            batch_num, build_time);

    define_partition_for_large(handle, data_view, d_data_view, d_centroid_view, segment_start_view, segment_length_view,
                               d_segment_index.view(), batch_num, param.metric, build_time);

    reorder_for_large_dataset(handle, data_view, d_data_view, segment_start_view, mapping_view,
                              reorder_data_view, d_segment_index.view(), batch_num, build_time);

    std::cout << "preprocessing time: " << build_time << " s" << std::endl;
    std::ofstream result_out(result_file, std::ios::app);
    result_out << build_time << ",";
    result_out.close();
}