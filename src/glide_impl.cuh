#include <utility>
#include <algorithm>
#include <cuvs/neighbors/cagra.hpp>

#include "glide.cuh"
#include "graph_kernel.cuh"
#include "search_kernel.cuh"
#include "kernel_config.cuh"
#include "utils.cuh"

#include <omp.h>
#include <random>
#include <thrust/execution_policy.h>

GLIDE::GLIDE(raft::device_resources &handle, uint32_t graph_degree,
             raft::host_matrix_view<float> h_data,
             raft::host_matrix_view<float> h_centroids, Metric metric)
        : handle(handle),
          metric(metric),
          d_graph{raft::make_device_matrix<uint32_t, uint32_t, raft::row_major>(handle, num(), graph_degree)},
          d_data{raft::make_device_matrix<float, uint32_t, raft::row_major>(handle, h_data.extent(0),
                                                                            h_data.extent(1))},
          d_centroids{raft::make_device_matrix<float, uint32_t, raft::row_major>(handle, h_centroids.extent(0),
                                                                                 h_centroids.extent(1))} {
    raft::copy(d_data.data_handle(), h_data.data_handle(),
               h_data.size(), raft::resource::get_stream_from_stream_pool(handle));
    raft::copy(d_centroids.data_handle(), h_centroids.data_handle(),
               h_centroids.size(), raft::resource::get_stream_from_stream_pool(handle));
}

void
GLIDE::load(raft::host_matrix_view<uint32_t> h_graph,
            raft::host_vector_view<uint32_t> h_start_points) {
    d_start_points.emplace(raft::make_device_vector<uint32_t, uint32_t>(handle, h_start_points.size()));

    raft::copy(d_graph.data_handle(), h_graph.data_handle(),
               h_graph.size(), raft::resource::get_cuda_stream(handle));
    raft::copy(d_start_points->data_handle(), h_start_points.data_handle(),
               h_start_points.size(), raft::resource::get_cuda_stream(handle));
}

void GLIDE::start_point_select(raft::host_vector_view<uint32_t> h_segment_start_view,
                               raft::host_vector_view<uint32_t> h_segment_length_view,
                               raft::host_vector_view<uint32_t> h_map_view,
                               raft::host_vector_view<uint32_t> h_reorder_start_points,
                               raft::host_vector_view<uint32_t> h_start_points,
                               float &build_time) {
    uint32_t start_point_num = h_segment_start_view.extent(0);
    thread_local std::mt19937 generator(std::random_device{}());

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    cudaEventRecord(start_time);
    for (uint32_t i = 0; i < start_point_num; i++) {
        std::uniform_int_distribution<uint32_t> dist(0, h_segment_length_view(i) - 1);
        uint32_t segment_start = h_segment_start_view(i);
        uint32_t id = dist(generator);

        h_reorder_start_points(i) = id;
        if (i != start_point_num - 1) {
            h_start_points(i) = h_map_view(segment_start + id);
        }
    }
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

void
GLIDE::subgraph_build_and_merge(SearchParameter &param, float relaxant_factor,
                                raft::device_matrix_view<float> d_reorder_data_view,
                                raft::device_vector_view<uint32_t> d_map_view,
                                raft::device_vector_view<uint32_t> d_segment_start_view,
                                raft::device_vector_view<uint32_t> d_segment_length_view,
                                raft::device_matrix_view<uint32_t> d_knn_graph_view,
                                float &build_time) {
    uint32_t knn_degree = d_knn_graph_view.extent(1);
    uint32_t graph_degree = d_graph.extent(1);
    uint32_t segment_num = d_segment_start_view.extent(0);
    uint32_t reorder_num = d_reorder_data_view.extent(0);
    uint32_t dim = d_reorder_data_view.extent(1);

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    uint32_t max_size = deviceProp.maxGridSize[1];
    uint32_t max_points = std::min(num(), max_size);

    uint32_t result_buffer_size = param.beam + knn_degree;
    result_buffer_size = roundUp32(result_buffer_size);
    uint32_t bitmap_size = ceildiv<uint32_t>(param.beam, 32);
    uint32_t max_graph_degree = roundUp32(graph_degree);
    uint32_t shared_mem_size = dim * sizeof(float) + result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                               max_graph_degree * sizeof(uint32_t) +
                               hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 3 * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 2 * sizeof(uint32_t) +
                               bitmap_size * sizeof(uint32_t);
    assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

    auto kernel_1 = build_kernel_config::choose_kernel(param.beam, knn_degree);

    cudaEventRecord(start_time);
    for (uint32_t pid = 0; pid < num(); pid += max_points) {
        cudaStream_t stream = raft::resource::get_cuda_stream(handle);
        uint32_t number_for_build = std::min(max_points, num() - pid);

        uint32_t threads_per_block = set_block_size(handle, knn_degree, param.search_block_size,
                                                    shared_mem_size, number_for_build);
        uint32_t blocks_per_grim = number_for_build;
        kernel_1<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_reorder_data_view.data_handle(),
                                                                                  d_graph.data_handle(),
                                                                                  start_point_view().data_handle(),
                                                                                  d_segment_start_view.data_handle(),
                                                                                  d_segment_length_view.data_handle(),
                                                                                  knn_degree,
                                                                                  graph_degree,
                                                                                  max_graph_degree,
                                                                                  segment_num,
                                                                                  dim,
                                                                                  d_map_view.data_handle(),
                                                                                  d_knn_graph_view.data_handle(),
                                                                                  param.beam,
                                                                                  param.hash_bit,
                                                                                  param.hash_reset_interval,
                                                                                  param.max_iterations,
                                                                                  param.min_iterations,
                                                                                  num(),
                                                                                  pid,
                                                                                  relaxant_factor,
                                                                                  metric);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;

    auto kernel_2 = build_boundary_kernel_config::choose_kernel(param.beam, knn_degree);
    uint32_t segment_id = segment_num - 1;
    uint32_t segment_start = num();
    uint32_t boundary_num = reorder_num - num();
    max_points = std::min(boundary_num, max_size);
    shared_mem_size += max_graph_degree * sizeof(uint32_t);

    cudaEventRecord(start_time);
    for (uint32_t pid = 0; pid < boundary_num; pid += max_points) {
        cudaStream_t stream = raft::resource::get_cuda_stream(handle);
        uint32_t number_for_build = std::min(max_points, boundary_num - pid);

        uint32_t threads_per_block = set_block_size(handle, knn_degree, param.search_block_size,
                                                    shared_mem_size, number_for_build);
        uint32_t blocks_per_grim = number_for_build;

        kernel_2<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_reorder_data_view.data_handle(),
                                                                                  d_graph.data_handle(),
                                                                                  start_point_view().data_handle(),
                                                                                  knn_degree,
                                                                                  graph_degree,
                                                                                  max_graph_degree,
                                                                                  dim,
                                                                                  bitmap_size,
                                                                                  segment_id,
                                                                                  segment_start,
                                                                                  d_map_view.data_handle(),
                                                                                  d_knn_graph_view.data_handle(),
                                                                                  param.beam,
                                                                                  param.hash_bit,
                                                                                  param.hash_reset_interval,
                                                                                  param.max_iterations,
                                                                                  param.min_iterations,
                                                                                  reorder_num,
                                                                                  segment_start + pid,
                                                                                  relaxant_factor,
                                                                                  metric);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

void
GLIDE::reverse_graph(float &build_time) {
    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    auto d_reverse_graph = raft::make_device_matrix<uint32_t>(handle, num(), graph_degree());
    auto d_reverse_graph_degree = raft::make_device_vector<uint32_t>(handle, num());
    auto d_destination_nodes = raft::make_device_vector<uint32_t>(handle, num());

    thrust::fill(thrust::device.on(raft::resource::get_stream_from_stream_pool(handle)),
                 d_reverse_graph.data_handle(),
                 d_reverse_graph.data_handle() + num() * graph_degree(),
                 0xffffffffu);
    thrust::fill(thrust::device.on(raft::resource::get_stream_from_stream_pool(handle)),
                 d_reverse_graph_degree.data_handle(),
                 d_reverse_graph_degree.data_handle() + num(),
                 0);

    cudaEventRecord(start_time);
    for (uint32_t degree_id = 0; degree_id < graph_degree(); degree_id++) {
        cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

        uint32_t threads_per_block = 512;
        uint32_t blocks_per_grid = (num() + threads_per_block - 1) / threads_per_block;

        reverse_graph_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_destination_nodes.data_handle(),
                                                                                d_reverse_graph.data_handle(),
                                                                                d_reverse_graph_degree.data_handle(),
                                                                                d_graph.data_handle(),
                                                                                num(),
                                                                                graph_degree(),
                                                                                degree_id);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
//    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;

    uint32_t threads_per_block = 32;
    uint32_t blocks_per_grid = num();
    cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);
    uint32_t max_graph_degree = roundUp32(graph_degree());
    uint32_t shared_mem_size = max_graph_degree * sizeof(uint32_t);

    cudaEventRecord(start_time);
    insert_reverse_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(
            d_graph.data_handle(),
            d_reverse_graph.data_handle(),
            d_reverse_graph_degree.data_handle(),
            num(),
            graph_degree(),
            max_graph_degree);
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

void GLIDE::refine(SearchParameter &param, float relaxant_factor, float &build_time) {
    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    uint32_t max_size = deviceProp.maxGridSize[1];
    uint32_t max_points = std::min(num(), max_size);

    cudaEvent_t start_time, stop_time;
    float milliseconds = 0;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    uint32_t result_buffer_size = param.beam + graph_degree();
    result_buffer_size = roundUp32(result_buffer_size);
    uint32_t bitmap_size = ceildiv<uint32_t>(param.beam, 32);
    uint32_t max_graph_degree = roundUp32(graph_degree());
    uint32_t shared_mem_size = dim() * sizeof(float) +
                               result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                               max_graph_degree * sizeof(uint32_t) +
                               max_graph_degree * sizeof(uint32_t) +
                               hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 3 * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 4 * sizeof(uint32_t) +
                               bitmap_size * sizeof(uint32_t);
    assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

    auto kernel = refine_kernel_config::choose_kernel(param.beam, graph_degree(), centroid_num());

    cudaEventRecord(start_time);
    for (uint32_t pid = 0; pid < num(); pid += max_points) {
        cudaStream_t stream = raft::resource::get_cuda_stream(handle);
        uint32_t number_for_build = std::min(max_points, num() - pid);

        uint32_t threads_per_block = set_block_size(handle, graph_degree(), param.search_block_size,
                                                    shared_mem_size, number_for_build);
        uint32_t blocks_per_grim = number_for_build;
        kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(data_view().data_handle(),
                                                                                d_graph.data_handle(),
                                                                                start_point_view().data_handle(),
                                                                                centroid_view().data_handle(),
                                                                                graph_degree(),
                                                                                max_graph_degree,
                                                                                dim(),
                                                                                param.beam,
                                                                                param.hash_bit,
                                                                                param.hash_reset_interval,
                                                                                param.max_iterations,
                                                                                param.min_iterations,
                                                                                num(),
                                                                                centroid_num(),
                                                                                relaxant_factor,
                                                                                pid,
                                                                                metric);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&milliseconds, start_time, stop_time);
    build_time += milliseconds / 1000.0f;
}

void
GLIDE::build(IndexParameter &build_param, SearchParameter &search_param_knn, SearchParameter &search_param_refine,
             std::optional<raft::device_matrix<uint32_t>> &d_knn_graph,
             std::optional<raft::device_vector<uint32_t>> &d_segment_start,
             std::optional<raft::device_vector<uint32_t>> &d_segment_length,
             std::optional<raft::device_vector<uint32_t>> &d_map,
             raft::host_vector_view<uint32_t> h_segment_start_view,
             raft::host_vector_view<uint32_t> h_segment_length_view,
             raft::host_vector_view<uint32_t> h_map_view,
             std::optional<raft::device_matrix<float>> &d_reorder_data,
             std::string result_file) {
    float build_time = 0.0f;

    auto h_start_point = raft::make_host_vector<uint32_t, uint32_t>(handle, d_segment_start->size() - 1);
    auto h_reorder_start_point = raft::make_host_vector<uint32_t, uint32_t>(handle, d_segment_start->size());

    start_point_select(h_segment_start_view, h_segment_length_view, h_map_view,
                       h_reorder_start_point.view(), h_start_point.view(), build_time);

    d_start_points.emplace(raft::make_device_vector<uint32_t, uint32_t>(handle, h_reorder_start_point.size()));
    raft::copy(d_start_points->data_handle(), h_reorder_start_point.data_handle(),
               h_reorder_start_point.size(), raft::resource::get_cuda_stream(handle));

    subgraph_build_and_merge(search_param_knn, build_param.relaxant_factor, d_reorder_data->view(),
                             d_map->view(), d_segment_start->view(), d_segment_length->view(),
                             d_knn_graph->view(), build_time);
    d_knn_graph.reset();
    d_segment_start.reset();
    d_segment_length.reset();
    d_map.reset();
    d_reorder_data.reset();

    d_start_points.reset();
    d_start_points.emplace(raft::make_device_vector<uint32_t, uint32_t>(handle, h_start_point.size()));
    raft::copy(d_start_points->data_handle(), h_start_point.data_handle(),
               h_start_point.size(), raft::resource::get_cuda_stream(handle));

    reverse_graph(build_time);

    if (search_param_refine.beam != 0) {
        refine(search_param_refine, build_param.relaxant_factor, build_time);
    }

    std::cout << "index build time: " << build_time << " s" << std::endl;
    std::ofstream result_out(result_file, std::ios::app);
    result_out << build_time << ",";
    result_out.close();
}

void
GLIDE::search(SearchParameter &param,
              raft::device_matrix_view<float> d_query_view,
              raft::host_matrix<uint32_t> &result_ids,
              raft::host_matrix<float> &result_dists,
              std::string result_file) {
    uint32_t query_number = d_query_view.extent(0);
    uint32_t top_k = param.topk;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaDeviceProp deviceProp = raft::resource::get_device_properties(handle);
    uint32_t max_grid_size = deviceProp.maxGridSize[1];
    uint32_t max_queries = std::min(query_number, max_grid_size);

    uint32_t result_buffer_size = param.beam + graph_degree();
    result_buffer_size = roundUp32(result_buffer_size);
    uint32_t shared_mem_size = dim() * sizeof(float) +
                               result_buffer_size * (sizeof(uint32_t) + sizeof(float)) +
                               hashmap::get_size(param.hash_bit) * sizeof(uint32_t) +
                               1 * sizeof(uint32_t) + 3 * sizeof(uint32_t) + 1 * sizeof(uint32_t);
    assert(shared_mem_size <= deviceProp.sharedMemPerBlock);

    auto d_result_ids = raft::make_device_matrix<uint32_t>(handle, query_number, top_k);
    auto d_result_dists = raft::make_device_matrix<float>(handle, query_number, top_k);

    auto kernel = search_kernel_config::choose_kernel(param.beam, graph_degree(), centroid_num());

    RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       shared_mem_size));

    cudaEventRecord(start);
    for (uint32_t qid = 0; qid < query_number; qid += max_queries) {
        uint32_t number_for_search = std::min(max_queries, query_number - qid);

        uint32_t threads_per_block = set_block_size(handle, graph_degree(), param.search_block_size,
                                                    shared_mem_size, number_for_search);
        uint32_t blocks_per_grim = number_for_search;
        std::cout << "threads per block: " << threads_per_block << "   " << "blocks_per_grim: " << blocks_per_grim
                  << std::endl;

        auto d_result_ids_ptr = d_result_ids.data_handle() + qid * top_k;
        auto d_result_dists_ptr = d_result_dists.data_handle() + qid * top_k;
        auto d_query_ptr = d_query_view.data_handle() + qid * dim();
        cudaStream_t stream = raft::resource::get_stream_from_stream_pool(handle);

        kernel<<<blocks_per_grim, threads_per_block, shared_mem_size, stream>>>(d_result_ids_ptr,
                                                                                d_result_dists_ptr,
                                                                                top_k,
                                                                                data_view().data_handle(),
                                                                                d_query_ptr,
                                                                                graph_view().data_handle(),
                                                                                graph_degree(),
                                                                                param.beam,
                                                                                param.max_iterations,
                                                                                param.min_iterations,
                                                                                param.hash_bit,
                                                                                param.hash_reset_interval,
                                                                                start_point_view().data_handle(),
                                                                                dim(),
                                                                                centroid_num(),
                                                                                centroid_view().data_handle(),
                                                                                metric);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    std::cout << "query time: " << seconds << " s" << "   " << "QPS: " << (float) query_number / seconds << std::endl;

    std::ofstream result;
    result.setf(std::ios::fixed);
    result.open(result_file, std::ios::app);
    result << seconds << "," << (float) query_number / seconds << ",";
    result.close();

    raft::copy(result_ids.data_handle(), d_result_ids.data_handle(), query_number * top_k,
               raft::resource::get_cuda_stream(handle));
    raft::copy(result_dists.data_handle(), d_result_dists.data_handle(), query_number * top_k,
               raft::resource::get_cuda_stream(handle));
    nvtxRangePop();
}