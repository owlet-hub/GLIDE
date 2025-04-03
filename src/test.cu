#include "load.cuh"
//#include "nsg_impl.cuh"
#include "glide_impl.cuh"
#include "partition.cuh"
#include "params.cuh"
#include "nn_descent.cuh"
#include <algorithm>
#include <fstream>

uint32_t calculate_recall(raft::host_matrix_view<uint32_t> neighbors, const std::string truth_path, uint32_t topk,
                          std::string result_file) {

    std::vector<std::vector<uint32_t>> true_neighbors;
    load_truth(truth_path, true_neighbors);

    uint32_t num_queries = neighbors.extent(0);
    float total_recall = 0.0;
    float max_recall = 0.0;
    float min_recall = 1.0;
    uint32_t point = 0;

    std::ofstream out;
    out.open("/data/gpuGraph/spgg/recall.txt", std::ios::ate);

    for (uint32_t query_id = 0; query_id < num_queries; query_id++) {
        uint32_t correct_count = 0;
        for (uint32_t k = 0; k < topk; ++k) {
            if (std::find(true_neighbors[query_id].begin(), true_neighbors[query_id].begin() + topk,
                          neighbors(query_id, k)) != true_neighbors[query_id].begin() + topk) {
                correct_count++;
            }
        }
        float recall = static_cast<float>(correct_count) / topk;

        out << query_id << ": " << recall << std::endl;
        total_recall += recall;
        if (recall > max_recall) {
            max_recall = recall;
        }
        if (recall < min_recall) {
            min_recall = recall;
        }

        if (recall == 1.0) {
            point++;
        }
    }
    out.close();
    std::cout << "max_recall: " << max_recall << std::endl;
    std::cout << "min_recall: " << min_recall << std::endl;
    std::cout << "total_recall: " << total_recall / num_queries << std::endl;

    std::ofstream result;
    result.open(result_file, std::ios::app);
    result << total_recall / num_queries << std::endl;
    result.close();

    //return total_recall / num_queries;
    return point;
}

int main(int argc, char **argv) {
//    if (argc != 18) {
//        std::cout << argv[0]
//                  << "data_file query_file truth_file centroid_num boundary_fact sample_fact graph_degree knn_degree relaxant_factor build_beam refine_beam refine topk search_beam search_width search_block_size"
//                  << std::endl;
//        exit(-1);
//    }

    cudaSetDevice(1);
    raft::device_resources handle;
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(32);
    raft::resource::set_cuda_stream_pool(handle, stream_pool);

    std::string data_file(argv[1]);
    std::string query_file(argv[2]);
    std::string truth_file(argv[3]);
    std::string result_file(argv[4]);
    std::string metric(argv[17]);
    uint32_t boundary_partition_num = 1;

    PartitionParameter partition_param;
    partition_param.centroid_num = std::stoi(argv[5]);
    partition_param.boundary_factor = std::stof(argv[6]);
    partition_param.sample_factor = std::stof(argv[7]);

    if(metric == "Euclidean") {
        partition_param.metric = Metric::Euclidean;
    } else if(metric == "Cosine") {
        partition_param.metric = Metric::Cosine;
    }

    IndexParameter build_param;
    build_param.graph_degree = std::stoi(argv[8]);
    build_param.knn_degree = std::stoi(argv[9]);
    build_param.relaxant_factor = std::stof(argv[10]);
    if(metric == "Euclidean") {
        build_param.metric = Metric::Euclidean;
    } else if(metric == "Cosine") {
        build_param.metric = Metric::Cosine;
    }

    NNDescentParameter nnd_param(build_param.knn_degree);

    SearchParameter search_param_knn;
    search_param_knn.beam = std::stoi(argv[11]);
    adjust_search_params(search_param_knn.min_iterations, search_param_knn.max_iterations,
                         search_param_knn.beam, search_param_knn.search_width);
    search_param_knn.hash_bit = calculate_hash_bitlen(search_param_knn.beam, search_param_knn.search_width,
                                                      build_param.knn_degree, search_param_knn.hash_max_fill_rate,
                                                      search_param_knn.hashmap_min_bitlen);
    search_param_knn.hash_reset_interval = calculate_hash_reset_interval(search_param_knn.beam,
                                                                         search_param_knn.search_width,
                                                                         build_param.knn_degree,
                                                                         search_param_knn.hash_max_fill_rate,
                                                                         search_param_knn.hash_bit);

    SearchParameter search_param_refine;
    search_param_refine.beam = std::stoi(argv[12]);
    adjust_search_params(search_param_refine.min_iterations, search_param_refine.max_iterations,
                         search_param_refine.beam, search_param_refine.search_width);
    search_param_refine.hash_bit = calculate_hash_bitlen(search_param_refine.beam,
                                                         search_param_refine.search_width,
                                                         build_param.knn_degree, search_param_refine.hash_max_fill_rate,
                                                         search_param_refine.hashmap_min_bitlen);
    search_param_refine.hash_reset_interval = calculate_hash_reset_interval(search_param_refine.beam,
                                                                            search_param_refine.search_width,
                                                                            build_param.knn_degree,
                                                                            search_param_refine.hash_max_fill_rate,
                                                                            search_param_refine.hash_bit);

//    SearchParameter search_param;
//    search_param.topk = std::stoi(argv[13]);
//    search_param.beam = std::stoi(argv[14]);
//    search_param.search_width = std::stoi(argv[15]);
//    search_param.search_block_size = std::stoi(argv[16]);
//
//    adjust_search_params(search_param.min_iterations, search_param.max_iterations,
//                         search_param.beam, search_param.search_width);
//    search_param.hash_bit = calculate_hash_bitlen(search_param.beam, search_param.search_width,
//                                                   build_param.graph_degree, search_param.hash_max_fill_rate,
//                                                   search_param.hashmap_min_bitlen);
//    search_param.hash_reset_interval = calculate_hash_reset_interval(search_param.beam,
//                                                                      search_param.search_width,
//                                                                      build_param.graph_degree,
//                                                                      search_param.hash_max_fill_rate,
//                                                                      search_param.hash_bit);

    auto op_data = load_data(data_file);
    uint32_t number = op_data.extent(0);
    uint32_t dim = op_data.extent(1);
    uint32_t dataset_block_dim = set_dataset_block_dim(dim);
    
    std::optional<raft::device_matrix<float>> d_reorder_data;
    std::optional<raft::device_vector<uint32_t>> d_mapping;
    std::optional<raft::device_vector<uint32_t>> d_segment_start;
    std::optional<raft::device_vector<uint32_t>> d_segment_length;
    std::optional<raft::device_matrix<float, int>> d_centroids;

    d_segment_start.emplace(raft::make_device_vector<uint32_t, uint32_t>(handle, partition_param.centroid_num +
                                                                                 boundary_partition_num));
    d_segment_length.emplace(raft::make_device_vector<uint32_t, uint32_t>(handle, partition_param.centroid_num +
                                                                                  boundary_partition_num));
    d_centroids.emplace(raft::make_device_matrix<float, int>(handle, partition_param.centroid_num, op_data.extent(1)));

    thrust::fill(thrust::device, d_segment_length->data_handle(),
                 d_segment_length->data_handle() + d_segment_length->size(), 0);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::optional<raft::device_matrix<float>> d_data;
    d_data.emplace(raft::make_device_matrix<float>(handle, op_data.extent(0), op_data.extent(1)));
    raft::copy(d_data->data_handle(), op_data.data_handle(), op_data.size(), raft::resource::get_cuda_stream(handle));

    std::ofstream result;
    result.open(result_file, std::ios::app);
    result <<"boundary_factor,index_time,degree,beam,search_time,QPS,recall"<<std::endl;
    result<<partition_param.boundary_factor<<",";
    result.close();

///preprocess
    cudaEventRecord(start);
    preprocess(handle, partition_param, d_data->view(), d_reorder_data, d_mapping, d_segment_start->view(),
               d_segment_length->view(), d_centroids->view());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float preprocess_time = milliseconds / 1000.0f;
    std::cout << "preprocess time: " << preprocess_time << " s" << std::endl;
    d_data.reset();

    auto centroids = raft::make_host_matrix<float>(partition_param.centroid_num, dim);
    raft::copy(centroids.data_handle(), d_centroids->data_handle(), centroids.size(),
               raft::resource::get_stream_from_stream_pool(handle));
    auto segment_start = raft::make_host_vector<uint32_t>(d_segment_start->size());
    raft::copy(segment_start.data_handle(), d_segment_start->data_handle(), segment_start.size(),
               raft::resource::get_stream_from_stream_pool(handle));
    auto segment_length = raft::make_host_vector<uint32_t>(d_segment_length->size());
    raft::copy(segment_length.data_handle(), d_segment_length->data_handle(), segment_length.size(),
               raft::resource::get_stream_from_stream_pool(handle));
    auto mapping = raft::make_host_vector<uint32_t>(d_mapping->size());
    raft::copy(mapping.data_handle(), d_mapping->data_handle(), mapping.size(),
               raft::resource::get_stream_from_stream_pool(handle));
    auto reorder_data = raft::make_host_matrix<float>(d_reorder_data->extent(0), d_reorder_data->extent(1));
    raft::copy(reorder_data.data_handle(), d_reorder_data->data_handle(), reorder_data.size(),
               raft::resource::get_stream_from_stream_pool(handle));

    build_param.segment_start_view = segment_start.view();
    build_param.segment_length_view = segment_length.view();
    build_param.mapping_view = mapping.view();

    for (uint32_t i = 0; i < segment_start.size(); i++) {
        std::cout << segment_start(i) << " " << segment_start(i) + segment_length(i) << " " << segment_length(i)
                  << std::endl;
    }

    d_centroids.reset();

///knn graph
    std::optional<raft::device_matrix<uint32_t>> d_knn_graph;
    d_knn_graph.emplace(raft::make_device_matrix<uint32_t>(handle, reorder_data.extent(0), build_param.knn_degree));

    cudaEventRecord(start);
    auto knn_index = build_nnd(handle, nnd_param, build_param, d_reorder_data->view());
//    auto knn_index = build_knn(handle, nnd_param, build_param, d_reorder_data->view());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float knn_time = milliseconds / 1000.0f;
    std::cout << "knn time: " << knn_time << " s" << std::endl;
    raft::copy(d_knn_graph->data_handle(), reinterpret_cast<uint32_t *>(knn_index.data_handle()), d_knn_graph->size(),
               raft::resource::get_stream_from_stream_pool(handle));

///build index
    NSG index(handle, number, build_param.graph_degree, centroids.view(), build_param.metric);
//    NSG index(handle, reorder_data.extent(0), build_param.graph_degree, centroids.view(), build_param.metric);

    cudaEventRecord(start);
    index.build(build_param, search_param_knn, search_param_refine, dataset_block_dim, d_knn_graph,
                d_segment_start, d_segment_length, d_mapping, d_reorder_data,
                op_data.view());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float index_time = milliseconds / 1000.0f;
    std::cout << "index build time: " << index_time << " s" << std::endl;

    auto d_query = load_query(query_file, handle);
//    auto result_ids = raft::make_host_matrix<uint32_t>(d_query.extent(0), search_param.topk);
//    auto result_distances = raft::make_host_matrix<float>(d_query.extent(0), search_param.topk);

//    result.open(result_file, std::ios::app);
//    result <<build_param.graph_degree<< ","<<search_param.beam<< ",";
//    result.close();

//    index.search(search_param, d_query.view(), dataset_block_dim, result_ids, result_distances, result_file);

//    uint32_t point = calculate_recall(result_ids.view(), truth_file,
//                                      search_param.topk, result_file);
//    std::cout << point << std::endl;

    uint32_t search_time = std::stoi(argv[18]);

    for(int i=0;i<search_time;i++){
        SearchParameter search_param;
        search_param.topk = std::stoi(argv[13]);
        search_param.beam = std::stoi(argv[14])+10*i;
        search_param.search_width = std::stoi(argv[15]);
        search_param.search_block_size = std::stoi(argv[16]);

        result.open(result_file, std::ios::app);
        result <<build_param.graph_degree<< ","<<search_param.beam<< ",";
        result.close();

        adjust_search_params(search_param.min_iterations, search_param.max_iterations,
                             search_param.beam, search_param.search_width);
        search_param.hash_bit = calculate_hash_bitlen(search_param.beam, search_param.search_width,
                                                      build_param.graph_degree, search_param.hash_max_fill_rate,
                                                      search_param.hashmap_min_bitlen);
        search_param.hash_reset_interval = calculate_hash_reset_interval(search_param.beam,
                                                                         search_param.search_width,
                                                                         build_param.graph_degree,
                                                                         search_param.hash_max_fill_rate,
                                                                         search_param.hash_bit);

        std::cout<<search_param.beam<<" "<<search_param.search_width<<" "<<search_param.hash_bit<<" "<<
        search_param.hash_max_fill_rate<<" "<<search_param.hash_reset_interval<<" "<<search_param.hashmap_min_bitlen<<" "
        <<search_param.max_iterations<<" "<<search_param.min_iterations<<" "<<search_param.search_block_size<<std::endl;

        auto result_ids = raft::make_host_matrix<uint32_t>(d_query.extent(0), search_param.topk);
        auto result_distances = raft::make_host_matrix<float>(d_query.extent(0), search_param.topk);

        index.search(search_param, d_query.view(), dataset_block_dim, result_ids, result_distances, result_file);

        uint32_t point = calculate_recall(result_ids.view(), truth_file,
                                          search_param.topk, result_file);
        std::cout << point << std::endl;
    }
}