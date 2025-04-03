#include <utility>
#include <algorithm>
#include <cuvs/neighbors/cagra.hpp>

#include "glide_large.cuh"
#include "graph_kernel.cuh"
#include "search_kernel.cuh"
#include "kernel_config.cuh"
#include "utils.cuh"

#include <omp.h>
#include <random>
#include <thrust/execution_policy.h>

GLIDE_for_large::
GLIDE_for_large(raft::device_resources &handle, uint32_t graph_degree, Metric metric, std::string reorder_file,
                std::string map_file, std::string centroid_file, std::string segment_file)
        : handle(handle),
          metric(metric),
          data(load_data<uint8_t, uint64_t>(reorder_file)),
          map(load_map(map_file)),
          centroids(load_data<float, uint32_t>(centroid_file)),
          segment_start(load_segment_start(segment_file)),
          segment_length(load_segment_length(segment_file)),
          start_points{raft::make_host_vector<uint32_t>(segment_num())},
          graph{raft::make_host_matrix<uint32_t, uint64_t, raft::row_major>(num(), graph_degree)} {
}

GLIDE_for_large::
GLIDE_for_large(raft::device_resources &handle, Metric metric, std::string reorder_file, std::string map_file,
                std::string centroid_file, std::string segment_file, std::string start_point_file,
                std::string graph_file)
        : handle(handle),
          metric(metric),
          data(load_data<uint8_t, uint64_t>(reorder_file)),
          map(load_map(map_file)),
          centroids(load_data<float, uint32_t>(centroid_file)),
          segment_start(load_segment_start(segment_file)),
          segment_length(load_segment_length(segment_file)),
          start_points{load_start_point(start_point_file)},
          graph{load_graph<uint32_t, uint64_t>(graph_file)} {

}

void GLIDE_for_large::
load(raft::host_matrix_view<uint32_t, uint64_t> h_graph_view,
     raft::host_vector_view<uint32_t> h_start_point_view) {
    raft::copy(graph.data_handle(), h_graph_view.data_handle(),
               h_graph_view.size(), raft::resource::get_cuda_stream(handle));
    raft::copy(start_points.data_handle(), h_start_point_view.data_handle(),
               h_start_point_view.size(), raft::resource::get_cuda_stream(handle));
}

void GLIDE_for_large::
start_point_select(float &build_time) {
    thread_local std::mt19937 generator(std::random_device{}());

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    cudaEventRecord(start_time);
    for (uint32_t i = 0; i < segment_num(); i++) {
        std::uniform_int_distribution<uint32_t> dist(0, segment_length_view()(i) - 1);
        uint32_t id = dist(generator);

        start_points(i) = id;
    }
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

void GLIDE_for_large::
subgraph_build_and_merge(SearchParameter &param, float relaxant_factor, float &build_time,
                         raft::host_matrix_view<uint32_t, uint64_t> h_knn_graph_view) {
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
    uint32_t shared_mem_size = dim() * sizeof(uint8_t) +
                               result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                               max_graph_degree * sizeof(uint32_t) +
                               hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) +
                               3 * sizeof(uint32_t) + 1 * sizeof(uint32_t) + 2 * sizeof(uint32_t) +
                               bitmap_size * sizeof(uint32_t);
    assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

    auto kernel = build_for_large_kernel_config::choose_kernel(param.beam, knn_degree);

    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    for (uint32_t segment_id = 0; segment_id < segment_num(); segment_id++) {
        uint32_t start = segment_start_view()(segment_id);
        uint32_t length = segment_length_view()(segment_id);
        uint32_t start_point = start_point_view()(segment_id);

        uint64_t data_start = static_cast<uint64_t>(start) * dim();
        uint64_t data_length = static_cast<uint64_t>(length) * dim();
        uint64_t graph_start = static_cast<uint64_t>(start) * graph_degree();
        uint64_t graph_length = static_cast<uint64_t>(length) * graph_degree();
        uint64_t knn_start = static_cast<uint64_t>(start) * knn_degree;
        uint64_t knn_length = static_cast<uint64_t>(length) * knn_degree;

        auto d_data = raft::make_device_matrix<uint8_t, uint64_t>(handle, length, dim());
        raft::copy(d_data.data_handle(), data_view().data_handle() + data_start,
                   data_length, raft::resource::get_cuda_stream(handle));
        auto d_knn_graph = raft::make_device_matrix<uint32_t, uint64_t>(handle, length, knn_degree);
        raft::copy(d_knn_graph.data_handle(), h_knn_graph_view.data_handle() + knn_start,
                   knn_length, raft::resource::get_cuda_stream(handle));

        auto d_graph = raft::make_device_matrix<uint32_t, uint64_t>(handle, length, graph_degree());

        uint32_t max_points = std::min(length, max_size);

        cudaEventRecord(start_time);
        for (uint32_t pid = 0; pid < length; pid += max_points) {
            uint32_t number_for_build = std::min(max_points, length - pid);

            uint32_t threads_per_block = set_block_size(handle, knn_degree, param.search_block_size,
                                                        shared_mem_size, number_for_build);
            uint32_t blocks_per_grim = number_for_build;
            kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_data.data_handle(),
                                                                                    d_graph.data_handle(),
                                                                                    start_point,
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

        raft::copy(graph.data_handle() + graph_start, d_graph.data_handle(),
                   graph_length, raft::resource::get_cuda_stream(handle));
    }
}

void GLIDE_for_large::
reverse_graph(float &build_time) {
    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    for (uint32_t segment_id = 0; segment_id < segment_num(); segment_id++) {
        uint32_t start = segment_start_view()(segment_id);
        uint32_t length = segment_length_view()(segment_id);

        uint64_t graph_start = static_cast<uint64_t>(start) * graph_degree();
        uint64_t graph_length = static_cast<uint64_t>(length) * graph_degree();

        auto d_graph = raft::make_device_matrix<uint32_t, uint64_t>(handle, length, graph_degree());
        raft::copy(d_graph.data_handle(), graph.data_handle() + graph_start,
                   graph_length, raft::resource::get_cuda_stream(handle));

        auto d_reverse_graph = raft::make_device_matrix<uint32_t, uint64_t>(handle, length, graph_degree());
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
            reverse_graph_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
                    d_destination_nodes.data_handle(),
                    d_reverse_graph.data_handle(),
                    d_reverse_graph_degree.data_handle(),
                    d_graph.data_handle(),
                    length,
                    graph_degree(),
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
        insert_reverse_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(
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

        raft::copy(graph.data_handle() + graph_start, d_graph.data_handle(),
                   graph_length, raft::resource::get_cuda_stream(handle));
    }
}

void GLIDE_for_large::
refine(SearchParameter &param, float relaxant_factor, float &build_time) {
    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    uint32_t max_size = deviceProp.maxGridSize[1];

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    uint32_t result_buffer_size = param.beam + graph_degree();
    result_buffer_size = roundUp32(result_buffer_size);
    uint32_t bitmap_size = ceildiv<uint32_t>(param.beam, 32);
    uint32_t max_graph_degree = roundUp32(graph_degree());

    uint32_t shared_mem_size = dim() * sizeof(uint8_t) +
                               result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                               max_graph_degree * sizeof(uint32_t) +
                               max_graph_degree * sizeof(uint32_t) +
                               hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) +
                               3 * sizeof(uint32_t) + 1 * sizeof(uint32_t) + 4 * sizeof(uint32_t) +
                               bitmap_size * sizeof(uint32_t);
    assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

    auto kernel = refine_for_large_kernel_config::choose_kernel(param.beam, graph_degree());

    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    for (uint32_t segment_id = 0; segment_id < segment_num(); segment_id++) {
        uint32_t start = segment_start_view()(segment_id);
        uint32_t length = segment_length_view()(segment_id);
        uint32_t start_point = start_point_view()(segment_id);

        uint64_t data_start = static_cast<uint64_t>(start) * dim();
        uint64_t data_length = static_cast<uint64_t>(length) * dim();
        uint64_t graph_start = static_cast<uint64_t>(start) * graph_degree();
        uint64_t graph_length = static_cast<uint64_t>(length) * graph_degree();

        auto d_data = raft::make_device_matrix<uint8_t, uint64_t>(handle, length, dim());
        raft::copy(d_data.data_handle(), data_view().data_handle() + data_start,
                   data_length, raft::resource::get_cuda_stream(handle));
        auto d_graph = raft::make_device_matrix<uint32_t, uint64_t>(handle, length, graph_degree());
        raft::copy(d_graph.data_handle(), graph_view().data_handle() + graph_start,
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
                                                                                    start_point,
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

        raft::copy(graph.data_handle() + graph_start, d_graph.data_handle(),
                   graph_length, raft::resource::get_cuda_stream(handle));
    }
}

void GLIDE_for_large::
build(IndexParameter &build_param, SearchParameter &search_param_knn, SearchParameter &search_param_refine,
      raft::host_matrix_view<uint32_t, uint64_t> h_knn_graph_view, std::string &result_file) {
    std::cout << "reorder_number: " << num() << std::endl;
    std::cout << "dim: " << dim() << std::endl;

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

void GLIDE_for_large::
search(SearchParameter &param, uint32_t min_segment_num, float boundary_factor,
       raft::device_matrix_view<uint8_t> d_query_view,
       raft::host_matrix_view<uint32_t> h_result_id_view,
       raft::host_matrix_view<float> h_result_dist_view,
       std::string &result_file) {
    uint32_t query_num = d_query_view.extent(0);
    uint32_t topk = param.topk;

    cudaEvent_t time_start, time_stop;
    float milliseconds = 0;
    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);

    float search_time = 0.0f;

    auto d_final_result_ids = raft::make_device_matrix<uint32_t>(handle, query_num, topk);
    auto d_final_result_dists = raft::make_device_matrix<float>(handle, query_num, topk);
    auto d_segment_ids = raft::make_device_matrix<uint32_t>(handle, query_num, centroid_num());

    thrust::fill(thrust::device, d_final_result_ids.data_handle(),
                 d_final_result_ids.data_handle() + d_final_result_ids.size(),
                 get_max_value<uint32_t>());
    thrust::fill(thrust::device, d_final_result_dists.data_handle(),
                 d_final_result_dists.data_handle() + d_final_result_dists.size(),
                 get_max_value<float>());

    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    uint32_t max_grid_size = deviceProp.maxGridSize[1];
    uint32_t max_queries = std::min(query_num, max_grid_size);

    auto select_kernel = select_segment_kernel_config::choose_kernel(centroid_num());
    auto sub_kernel = search_sub_kernel_config::choose_kernel(param.beam, graph_degree(), topk);

    {
        auto d_centroids = raft::make_device_matrix<float>(handle, centroid_num(), dim());
        raft::copy(d_centroids.data_handle(), centroids_view().data_handle(),
                   centroids_view().size(), raft::resource::get_cuda_stream(handle));

        uint32_t shared_mem_size =
                dim() * sizeof(uint8_t) + centroid_num() * sizeof(uint32_t) + centroid_num() * sizeof(float);
        assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

        cudaEventRecord(time_start);
        for (uint32_t qid = 0; qid < query_num; qid += max_queries) {
            uint32_t number_for_search = std::min(max_queries, query_num - qid);

            uint32_t threads_per_block = 128;
            uint32_t blocks_per_grim = number_for_search;
            auto d_segment_ids_ptr = d_segment_ids.data_handle() + qid * centroid_num();
            auto d_query_ptr = d_query_view.data_handle() + qid * dim();
            cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

            select_kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_segment_ids_ptr,
                                                                                           d_query_ptr,
                                                                                           d_centroids.data_handle(),
                                                                                           centroid_num(),
                                                                                           dim(),
                                                                                           min_segment_num,
                                                                                           boundary_factor,
                                                                                           metric);
            RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        }
        cudaEventRecord(time_stop);
        cudaEventSynchronize(time_stop);
        cudaEventElapsedTime(&milliseconds, time_start, time_stop);
        search_time += milliseconds / 1000.0f;
        std::cout << "select time: " << milliseconds / 1000.0f << " s" << std::endl;
    }

    for (uint32_t segment_id = 0; segment_id < segment_num(); segment_id++) {
        uint32_t start = segment_start_view()(segment_id);
        uint32_t length = segment_length_view()(segment_id);
        uint32_t start_point = start_point_view()(segment_id);

        uint64_t data_start = static_cast<uint64_t>(start) * dim();
        uint64_t data_length = static_cast<uint64_t>(length) * dim();
        uint64_t graph_start = static_cast<uint64_t>(start) * graph_degree();
        uint64_t graph_length = static_cast<uint64_t>(length) * graph_degree();

        auto d_graph = raft::make_device_matrix<uint32_t, uint64_t>(handle, length, graph_degree());
        auto d_map = raft::make_device_vector<uint32_t>(handle, length);
        auto d_data = raft::make_device_matrix<uint8_t, uint64_t>(handle, length, dim());

        raft::copy(d_data.data_handle(), data_view().data_handle() + data_start,
                   data_length, raft::resource::get_cuda_stream(handle));
        raft::copy(d_graph.data_handle(), graph_view().data_handle() + graph_start,
                   graph_length, raft::resource::get_cuda_stream(handle));
        raft::copy(d_map.data_handle(), map.data_handle() + start,
                   length, raft::resource::get_cuda_stream(handle));

        uint32_t result_buffer_size = param.beam + graph_degree();
        result_buffer_size = roundUp32(result_buffer_size);
        uint32_t shared_mem_size = dim() * sizeof(uint8_t) +
                                   result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                                   hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                                   1 * sizeof(uint32_t) + 3 * sizeof(uint32_t) +
                                   1 * sizeof(uint32_t) + 1 * sizeof(uint32_t);
        assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

        cudaEventRecord(time_start);
        for (uint32_t qid = 0; qid < query_num; qid += max_queries) {
            uint32_t number_for_search = std::min(max_queries, query_num - qid);

            uint32_t threads_per_block = set_block_size(handle, graph_degree(), param.search_block_size,
                                                        shared_mem_size, number_for_search);
            uint32_t blocks_per_grim = number_for_search;
            auto d_final_result_ids_ptr = d_final_result_ids.data_handle() + qid * topk;
            auto d_final_result_distances_ptr = d_final_result_dists.data_handle() + qid * topk;
            auto d_segment_ids_ptr = d_segment_ids.data_handle() + qid * centroid_num();
            auto d_query_ptr = d_query_view.data_handle() + qid * dim();
            cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

            sub_kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(
                    d_final_result_ids_ptr,
                    d_final_result_distances_ptr,
                    topk,
                    d_data.data_handle(),
                    d_query_ptr,
                    d_segment_ids_ptr,
                    d_graph.data_handle(),
                    d_map.data_handle(),
                    graph_degree(),
                    centroid_num(),
                    segment_id,
                    param.beam,
                    param.max_iterations,
                    param.min_iterations,
                    param.hash_bit,
                    param.hash_reset_interval,
                    start_point,
                    dim(),
                    metric);
            RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        }
        cudaEventRecord(time_stop);
        cudaEventSynchronize(time_stop);
        cudaEventElapsedTime(&milliseconds, time_start, time_stop);
        search_time += milliseconds / 1000.0f;
        std::cout << "query " << segment_id << " time: " << milliseconds / 1000.0f << " s" << std::endl;
    }
    std::cout << "total query time: " << search_time << " s" << "   " << "QPS: " << (float) query_num / search_time
              << std::endl;

    std::ofstream result;
    result.setf(std::ios::fixed);
    result.open(result_file, std::ios::app);
    result << search_time << "," << (float) query_num / search_time << ",";
    result.close();

    raft::copy(h_result_id_view.data_handle(), d_final_result_ids.data_handle(), query_num * topk,
               raft::resource::get_cuda_stream(handle));
    raft::copy(h_result_dist_view.data_handle(), d_final_result_dists.data_handle(), query_num * topk,
               raft::resource::get_cuda_stream(handle));
}