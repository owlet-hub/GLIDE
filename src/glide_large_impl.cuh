#include <utility>
#include <algorithm>
#include <cuvs/neighbors/cagra.hpp>

#include "glide_large.cuh"
#include "graph_kernel.cuh"
#include "search_kernel.cuh"
#include "kernel_config.cuh"

#include <omp.h>
#include <random>
#include <thrust/execution_policy.h>

template<typename Data_t, typename Index_t>
GLIDE_large<Data_t, Index_t>::GLIDE_large(raft::device_resources &handle, uint32_t graph_degree, Metric metric,
                                          std::string &reorder_file, std::string &map_file,
                                          std::string &centroid_file, std::string &segment_file)
        : handle(handle),
          metric(metric),
          h_data(load_matrix_data<Data_t, Index_t>(reorder_file)),
          h_map(load_vector_data(map_file)),
          h_centroids(load_matrix_data<float, uint32_t>(centroid_file)),
          h_segment_start(load_segment_start(segment_file)),
          h_segment_length(load_segment_length(segment_file)),
          h_start_point{raft::make_host_vector<uint32_t>(segment_num())},
          h_graph{raft::make_host_matrix<uint32_t, Index_t, raft::row_major>(num(), graph_degree)} {
}
template<typename Data_t, typename Index_t>
GLIDE_large<Data_t, Index_t>::GLIDE_large(raft::device_resources &handle, Metric metric,
                                          std::string &reorder_file, std::string &map_file,
                                          std::string &centroid_file, std::string &segment_file,
                                          std::string &start_point_file, std::string &graph_file)
        : handle(handle),
          metric(metric),
          h_data(load_matrix_data<Data_t, Index_t>(reorder_file)),
          h_map(load_vector_data(map_file)),
          h_centroids(load_matrix_data<float, uint32_t>(centroid_file)),
          h_segment_start(load_segment_start(segment_file)),
          h_segment_length(load_segment_length(segment_file)),
          h_start_point{load_vector_data(start_point_file)},
          h_graph{load_matrix_data<uint32_t, Index_t>(graph_file)} {
}

template<typename Data_t, typename Index_t>
void
GLIDE_large<Data_t, Index_t>::load(raft::host_matrix_view<uint32_t, Index_t> h_graph_view,
                                   raft::host_vector_view<uint32_t> h_start_point_view) {
    raft::copy(h_graph.data_handle(), h_graph_view.data_handle(),
               h_graph_view.size(), raft::resource::get_cuda_stream(handle));
    raft::copy(h_start_point.data_handle(), h_start_point_view.data_handle(),
               h_start_point_view.size(), raft::resource::get_cuda_stream(handle));
}

template<typename Data_t, typename Index_t>
void
GLIDE_large<Data_t, Index_t>::start_point_select(float &build_time) {
    thread_local std::mt19937 generator(std::random_device{}());

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    cudaEventRecord(start_time);
    for (uint32_t i = 0; i < segment_num(); i++) {
        std::uniform_int_distribution<uint32_t> dist(0, segment_length_view()(i) - 1);
        uint32_t id = dist(generator);

        h_start_point(i) = id;
    }
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

template<typename Data_t, typename Index_t>
void
GLIDE_large<Data_t, Index_t>::subgraph_build_and_merge(SearchParameter &param, float relaxant_factor, float &build_time,
                                                       raft::host_matrix_view<uint32_t, Index_t> h_knn_graph_view) {
    uint32_t knn_degree = h_knn_graph_view.extent(1);

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);

    uint32_t max_size = deviceProp.maxGridSize[1];

    uint32_t result_buffer_size = param.beam + knn_degree;
    result_buffer_size = roundUp32(result_buffer_size);
    uint32_t bitmap_size = ceildiv<uint32_t>(param.beam, 32);
    uint32_t max_graph_degree = roundUp32(graph_degree());

    uint32_t shared_mem_size = dim() * sizeof(Data_t) +
                               result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                               max_graph_degree * sizeof(uint32_t) +
                               hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 3 * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 2 * sizeof(uint32_t) +
                               bitmap_size * sizeof(uint32_t);
    assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

    auto kernel = build_for_large_kernel_config<Data_t, Index_t>::choose_kernel(param.beam, knn_degree);

    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    for (uint32_t segment_id = 0; segment_id < segment_num(); segment_id++) {
        uint32_t start = segment_start_view()(segment_id);
        uint32_t length = segment_length_view()(segment_id);
        uint32_t start_point_select = start_point_view()(segment_id);

        Index_t data_start_pos = static_cast<Index_t>(start) * dim();
        Index_t graph_start_pos = static_cast<Index_t>(start) * graph_degree();
        Index_t knn_start_pos = static_cast<Index_t>(start) * knn_degree;
        Index_t data_length = static_cast<Index_t>(length) * dim();
        Index_t graph_length = static_cast<Index_t>(length) * graph_degree();
        Index_t knn_length = static_cast<Index_t>(length) * knn_degree;

        auto d_data = raft::make_device_matrix<Data_t, Index_t>(handle, length, dim());
        raft::copy(d_data.data_handle(), data_view().data_handle() + data_start_pos,
                   data_length, raft::resource::get_cuda_stream(handle));
        auto d_knn_graph = raft::make_device_matrix<uint32_t, Index_t>(handle, length, knn_degree);
        raft::copy(d_knn_graph.data_handle(), h_knn_graph_view.data_handle() + knn_start_pos,
                   knn_length, raft::resource::get_cuda_stream(handle));

        auto d_graph = raft::make_device_matrix<uint32_t, Index_t>(handle, length, graph_degree());

        uint32_t max_points = std::min(length, max_size);

        cudaEventRecord(start_time);
        for (uint32_t pid = 0; pid < length; pid += max_points) {
            uint32_t number_for_build = std::min(max_points, length - pid);

            uint32_t threads_per_block = set_block_size(handle, knn_degree, param.search_block_size,
                                                        shared_mem_size, number_for_build);
            uint32_t blocks_per_grim = number_for_build;

            kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_data.data_handle(),
                                                                                    d_graph.data_handle(),
                                                                                    start_point_select,
                                                                                    knn_degree,
                                                                                    graph_degree(),
                                                                                    max_graph_degree,
                                                                                    dim(),
                                                                                    d_knn_graph.data_handle(),
                                                                                    param.beam,
                                                                                    param.hash_bit,
                                                                                    param.hash_reset_interval,
                                                                                    param.max_iterations,
                                                                                    param.min_iterations,
                                                                                    length,
                                                                                    pid,
                                                                                    relaxant_factor,
                                                                                    metric);
            RAFT_CUDA_TRY(cudaPeekAtLastError());
        }
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        build_time += milliseconds / 1000.0f;

        raft::copy(h_graph.data_handle() + graph_start_pos, d_graph.data_handle(),
                   graph_length, raft::resource::get_cuda_stream(handle));
    }
}

template<typename Data_t, typename Index_t>
void
GLIDE_large<Data_t, Index_t>::reverse_graph(float &build_time) {
    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    for (uint32_t segment_id = 0; segment_id < segment_num(); segment_id++) {
        uint32_t start = segment_start_view()(segment_id);
        uint32_t length = segment_length_view()(segment_id);

        Index_t graph_start_pos = static_cast<Index_t>(start) * graph_degree();
        Index_t graph_length = static_cast<Index_t>(length) * graph_degree();

        auto d_graph = raft::make_device_matrix<uint32_t, Index_t>(handle, length, graph_degree());
        raft::copy(d_graph.data_handle(), h_graph.data_handle() + graph_start_pos,
                   graph_length, raft::resource::get_cuda_stream(handle));

        auto d_reverse_graph = raft::make_device_matrix<uint32_t, Index_t>(handle, length, graph_degree());
        auto d_reverse_graph_degree = raft::make_device_vector<uint32_t>(handle, length);
        auto d_destination_nodes = raft::make_device_vector<uint32_t>(handle, length);

        thrust::fill(thrust::device.on(raft::resource::get_stream_from_stream_pool(handle)),
                     d_reverse_graph.data_handle(),
                     d_reverse_graph.data_handle() + graph_length,
                     get_max_value<uint32_t>());
        thrust::fill(thrust::device.on(raft::resource::get_stream_from_stream_pool(handle)),
                     d_reverse_graph_degree.data_handle(),
                     d_reverse_graph_degree.data_handle() + length,
                     0);

        cudaEventRecord(start_time);
        for (uint32_t degree_id = 0; degree_id < graph_degree(); degree_id++) {
            cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

            uint32_t threads_per_block = 512;
            uint32_t blocks_per_grid = (length + threads_per_block - 1) / threads_per_block;

            reverse_graph_kernel<Index_t><<<blocks_per_grid, threads_per_block, 0, stream>>>(
                    d_destination_nodes.data_handle(),
                    d_reverse_graph.data_handle(),
                    d_reverse_graph_degree.data_handle(),
                    d_graph.data_handle(),
                    length, graph_degree(),
                    degree_id);
            RAFT_CUDA_TRY(cudaPeekAtLastError());
        }
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        build_time += milliseconds / 1000.0f;

        uint32_t threads_per_block = 32;
        uint32_t blocks_per_grid = length;
        cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);
        uint32_t max_graph_degree = roundUp32(graph_degree());
        uint32_t shared_mem_size = max_graph_degree * sizeof(uint32_t);

        cudaEventRecord(start_time);
        insert_reverse_kernel<Index_t><<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(
                d_graph.data_handle(),
                d_reverse_graph.data_handle(),
                d_reverse_graph_degree.data_handle(),
                length,
                graph_degree(),
                max_graph_degree);
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        build_time += milliseconds / 1000.0f;

        raft::copy(h_graph.data_handle() + graph_start_pos, d_graph.data_handle(),
                   graph_length, raft::resource::get_cuda_stream(handle));
    }
}

template<typename Data_t, typename Index_t>
void
GLIDE_large<Data_t, Index_t>::refine(SearchParameter &param, float relaxant_factor, float &build_time) {
    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    uint32_t max_size = deviceProp.maxGridSize[1];

    uint32_t result_buffer_size = param.beam + graph_degree();
    result_buffer_size = roundUp32(result_buffer_size);
    uint32_t bitmap_size = ceildiv<uint32_t>(param.beam, 32);
    uint32_t max_graph_degree = roundUp32(graph_degree());

    uint32_t shared_mem_size = dim() * sizeof(Data_t) +
                               result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                               max_graph_degree * sizeof(uint32_t) +
                               max_graph_degree * sizeof(uint32_t) +
                               hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 3 * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 4 * sizeof(uint32_t) +
                               bitmap_size * sizeof(uint32_t);
    assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

    auto kernel = refine_for_large_kernel_config<Data_t, Index_t>::choose_kernel(param.beam, graph_degree());

    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    for (uint32_t segment_id = 0; segment_id < segment_num(); segment_id++) {
        uint32_t start = segment_start_view()(segment_id);
        uint32_t length = segment_length_view()(segment_id);
        uint32_t start_point_select = start_point_view()(segment_id);

        Index_t data_start_pos = static_cast<Index_t>(start) * dim();
        Index_t graph_start_pos = static_cast<Index_t>(start) * graph_degree();
        Index_t data_length = static_cast<Index_t>(length) * dim();
        Index_t graph_length = static_cast<Index_t>(length) * graph_degree();

        auto d_data = raft::make_device_matrix<Data_t, Index_t>(handle, length, dim());
        raft::copy(d_data.data_handle(), data_view().data_handle() + data_start_pos,
                   data_length, raft::resource::get_cuda_stream(handle));
        auto d_graph = raft::make_device_matrix<uint32_t, Index_t>(handle, length, graph_degree());
        raft::copy(d_graph.data_handle(), graph_view().data_handle() + graph_start_pos,
                   graph_length, raft::resource::get_cuda_stream(handle));

        uint32_t max_points = std::min(length, max_size);

        cudaEventRecord(start_time);
        for (uint32_t pid = 0; pid < length; pid += max_points) {
            uint32_t number_for_build = std::min(max_points, length - pid);

            uint32_t threads_per_block = set_block_size(handle, graph_degree(), param.search_block_size,
                                                        shared_mem_size, number_for_build);
            uint32_t blocks_per_grim = number_for_build;

            kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_data.data_handle(),
                                                                                    d_graph.data_handle(),
                                                                                    start_point_select,
                                                                                    graph_degree(),
                                                                                    max_graph_degree,
                                                                                    dim(),
                                                                                    param.beam,
                                                                                    param.hash_bit,
                                                                                    param.hash_reset_interval,
                                                                                    param.max_iterations,
                                                                                    param.min_iterations,
                                                                                    length,
                                                                                    relaxant_factor,
                                                                                    pid,
                                                                                    metric);
            RAFT_CUDA_TRY(cudaPeekAtLastError());
        }
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        build_time += milliseconds / 1000.0f;

        raft::copy(h_graph.data_handle() + graph_start_pos, d_graph.data_handle(),
                   graph_length, raft::resource::get_cuda_stream(handle));
    }
}

template<typename Data_t, typename Index_t>
void
GLIDE_large<Data_t, Index_t>::build(IndexParameter &build_param, SearchParameter &search_param_knn,
                                    SearchParameter &search_param_refine,
                                    raft::host_matrix_view<uint32_t, Index_t> h_knn_graph_view,
                                    std::string &result_file) {
    float build_time = 0.0f;

    start_point_select(build_time);

    subgraph_build_and_merge(search_param_knn, build_param.relaxant_factor, build_time, h_knn_graph_view);

    reverse_graph(build_time);

    if (search_param_refine.beam != 0) {
        refine(search_param_refine, build_param.relaxant_factor, build_time);
    }

    std::cout << "index build time: " << build_time << " s" << std::endl;

    std::ofstream result_out(result_file, std::ios::app);
    result_out << build_time << ",";
    result_out.close();
}

template<typename Data_t, typename Index_t>
void
GLIDE_large<Data_t, Index_t>::search(SearchParameter &param, uint32_t min_segment_num, float boundary_factor,
                                     raft::device_matrix_view<Data_t, Index_t> d_query_view,
                                     raft::host_matrix_view<uint32_t> h_result_ids_view,
                                     raft::host_matrix_view<float> h_result_distances_view,
                                     std::string &result_file) {
    uint32_t query_number = d_query_view.extent(0);
    uint32_t top_k = param.top_k;

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    float search_time = 0.0f;

    auto d_final_result_ids = raft::make_device_matrix<uint32_t>(handle, query_number, top_k);
    auto d_final_result_distances = raft::make_device_matrix<float>(handle, query_number, top_k);
    auto d_segment_ids = raft::make_device_matrix<uint32_t>(handle, query_number, centroids_num());

    thrust::fill(thrust::device, d_final_result_ids.data_handle(),
                 d_final_result_ids.data_handle() + d_final_result_ids.size(), 0xffffffffu);
    thrust::fill(thrust::device, d_final_result_distances.data_handle(),
                 d_final_result_distances.data_handle() + d_final_result_distances.size(), FLT_MAX);

    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    uint32_t max_grid_size = deviceProp.maxGridSize[1];
    uint32_t max_queries = std::min(query_number, max_grid_size);

    auto select_kernel = select_segment_kernel_config<Data_t, Index_t>::choose_kernel(centroids_num());
    auto sub_kernel = search_on_sub_kernel_config<Data_t, Index_t>::choose_kernel(param.beam, graph_degree(), top_k);

    {
        auto d_centroids = raft::make_device_matrix<float>(handle, centroids_view().extent(0),
                                                           centroids_view().extent(1));
        raft::copy(d_centroids.data_handle(), centroids_view().data_handle(),
                   centroids_view().size(), raft::resource::get_cuda_stream(handle));

        uint32_t shared_mem_size = dim() * sizeof(Data_t) + centroids_num() * (sizeof(uint32_t) + sizeof(float));
        assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

        cudaEventRecord(start_time);
        for (uint32_t qid = 0; qid < query_number; qid += max_queries) {
            uint32_t number_for_search = std::min(max_queries, query_number - qid);

            uint32_t threads_per_block = 128;
            uint32_t blocks_per_grim = number_for_search;

            auto d_segment_ids_ptr = d_segment_ids.data_handle() + qid * centroids_num();
            auto d_query_ptr = d_query_view.data_handle() + qid * dim();
            cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

            select_kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_segment_ids_ptr,
                                                                                           d_query_ptr,
                                                                                           d_centroids.data_handle(),
                                                                                           centroids_num(),
                                                                                           dim(),
                                                                                           min_segment_num,
                                                                                           boundary_factor,
                                                                                           metric);
            RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        }
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        search_time += milliseconds / 1000.0f;
    }

    for (uint32_t segment_id = 0; segment_id < segment_num(); segment_id++) {
        uint32_t start = segment_start_view()(segment_id);
        uint32_t length = segment_length_view()(segment_id);
        uint32_t start_point_select = start_point_view()(segment_id);

        Index_t data_start_pos = static_cast<Index_t>(start) * dim();
        Index_t graph_start_pos = static_cast<Index_t>(start) * graph_degree();
        Index_t data_length = static_cast<Index_t>(length) * dim();
        Index_t graph_length = static_cast<Index_t>(length) * graph_degree();

        auto d_graph = raft::make_device_matrix<uint32_t, Index_t>(handle, length, graph_degree());
        auto d_map = raft::make_device_vector<uint32_t>(handle, length);
        auto d_data = raft::make_device_matrix<Data_t, Index_t>(handle, length, dim());

        raft::copy(d_data.data_handle(), data_view().data_handle() + data_start_pos,
                   data_length, raft::resource::get_cuda_stream(handle));
        raft::copy(d_graph.data_handle(), graph_view().data_handle() + graph_start_pos,
                   graph_length, raft::resource::get_cuda_stream(handle));
        raft::copy(d_map.data_handle(), map_view().data_handle() + start,
                   length, raft::resource::get_cuda_stream(handle));

        uint32_t result_buffer_size = param.beam + graph_degree();
        result_buffer_size = roundUp32(result_buffer_size);
        uint32_t shared_mem_size = dim() * sizeof(Data_t) +
                                   result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                                   hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                                   1 * sizeof(uint32_t) + 3 * sizeof(uint32_t) +
                                   1 * sizeof(uint32_t) + 1 * sizeof(uint32_t);
        assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

        cudaEventRecord(start_time);
        for (uint32_t qid = 0; qid < query_number; qid += max_queries) {
            uint32_t number_for_search = std::min(max_queries, query_number - qid);

            uint32_t threads_per_block = set_block_size(handle, graph_degree(), param.search_block_size,
                                                        shared_mem_size, number_for_search);
            uint32_t blocks_per_grim = number_for_search;

            auto d_final_result_ids_ptr = d_final_result_ids.data_handle() + qid * top_k;
            auto d_final_result_distances_ptr = d_final_result_distances.data_handle() + qid * top_k;
            auto d_segment_ids_ptr = d_segment_ids.data_handle() + qid * centroids_num();
            auto d_query_ptr = d_query_view.data_handle() + qid * dim();
            cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

            sub_kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_final_result_ids_ptr,
                                                                                        d_final_result_distances_ptr,
                                                                                        top_k,
                                                                                        d_data.data_handle(),
                                                                                        d_query_ptr,
                                                                                        d_segment_ids_ptr,
                                                                                        d_graph.data_handle(),
                                                                                        d_map.data_handle(),
                                                                                        graph_degree(),
                                                                                        centroids_num(),
                                                                                        segment_id,
                                                                                        param.beam,
                                                                                        param.max_iterations,
                                                                                        param.min_iterations,
                                                                                        param.hash_bit,
                                                                                        param.hash_reset_interval,
                                                                                        start_point_select,
                                                                                        dim(),
                                                                                        metric);
            RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        }
        cudaEventRecord(stop_time);
        cudaEventSynchronize(stop_time);
        cudaEventElapsedTime(&milliseconds, start_time, stop_time);
        search_time += milliseconds / 1000.0f;
    }
    std::cout << "query time: " << search_time << " s" << "   " << "QPS: " << (float) query_number / search_time
              << std::endl;

    std::ofstream result;
    result.setf(std::ios::fixed);
    result.open(result_file, std::ios::app);
    result << search_time << "," << (float) query_number / search_time << ",";
    result.close();

    raft::copy(h_result_ids_view.data_handle(), d_final_result_ids.data_handle(), query_number * top_k,
               raft::resource::get_cuda_stream(handle));
    raft::copy(h_result_distances_view.data_handle(), d_final_result_distances.data_handle(), query_number * top_k,
               raft::resource::get_cuda_stream(handle));
}