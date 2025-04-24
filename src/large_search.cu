#include "load.cuh"
#include "glide_large_impl.cuh"
#include <fstream>

uint32_t calculate_recall(raft::host_matrix_view<uint32_t> neighbors,
                          raft::host_matrix_view<uint32_t> truth,
                          uint32_t top_k) {
    uint32_t query_num = neighbors.extent(0);
    float total_recall = 0.0;

    for (uint32_t query_id = 0; query_id < query_num; query_id++) {
        uint32_t correct_count = 0;
        for (uint32_t k = 0; k < top_k; k++) {
            uint32_t neighbor_id = neighbors(query_id, k);
            for (uint32_t t = 0; t < top_k; t++) {
                if (truth(query_id, t) == neighbor_id) {
                    correct_count++;
                    break;
                }
            }
        }
        float recall = static_cast<float>(correct_count) / top_k;
        total_recall += recall;
    }
    std::cout << "total_recall: " << total_recall / query_num << std::endl;
    return total_recall / query_num;
}

int main(int argc, char **argv) {
    if (argc != 11) {
        std::cout << argv[0]
                  << "preprocess_file query_file truth_file graph_base_file result_file metric topk search_beam min_segment_num boundary_factor"
                  << std::endl;
        exit(-1);
    }

    cudaSetDevice(1);
    raft::device_resources handle;
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(32);
    raft::resource::set_cuda_stream_pool(handle, stream_pool);

    std::string preprocess_file(argv[1]);
    std::string segment_file = preprocess_file + ".segment";
    std::string reorder_file = preprocess_file + ".reorder";
    std::string map_file = preprocess_file + ".map";
    std::string centroid_file = preprocess_file + ".centroids";
    std::string query_file(argv[2]);
    std::string truth_file(argv[3]);
    std::string graph_base_file(argv[4]);
    std::string result_file(argv[5]);
    std::string graph_file = graph_base_file + ".graph";
    std::string start_point_file = graph_base_file + ".sp";
    std::string metric(argv[6]);
    Metric index_metric;
    if (metric == "Euclidean") {
        index_metric = Metric::Euclidean;
    } else if (metric == "Cosine") {
        index_metric = Metric::Cosine;
    }

    SearchParameter search_param;
    search_param.topk = std::stoi(argv[7]);
    search_param.beam = std::stoi(argv[8]);
    uint32_t min_segment_num = std::stoi(argv[9]);
    float boundary_factor = std::stof(argv[10]);

    std::ofstream result_out(result_file, std::ios::app);
    result_out << search_param.beam << ",";
    result_out.close();

    GLIDE_large index(handle, index_metric,
                      reorder_file, map_file, centroid_file,
                      segment_file, start_point_file, graph_file);

    adjust_search_params(search_param.min_iterations, search_param.max_iterations, search_param.beam);
    search_param.hash_bit = calculate_hash_bitlen(search_param.beam, index.graph_degree(),
                                                  search_param.hash_max_fill_rate,
                                                  search_param.hashmap_min_bitlen);
    search_param.hash_reset_interval = calculate_hash_reset_interval(search_param.beam,
                                                                     index.graph_degree(),
                                                                     search_param.hash_max_fill_rate,
                                                                     search_param.hash_bit);

    auto query = load_data<uint8_t, uint32_t>(query_file);
    auto d_query = raft::make_device_matrix<uint8_t, uint32_t>(handle, query.extent(0), query.extent(1));
    raft::copy(d_query.data_handle(), query.data_handle(),
               query.size(), raft::resource::get_stream_from_stream_pool(handle));
    auto result_ids = raft::make_host_matrix<uint32_t, uint32_t>(d_query.extent(0), search_param.topk);
    auto result_distances = raft::make_host_matrix<float, uint32_t>(d_query.extent(0), search_param.topk);

    index.search(search_param, min_segment_num, boundary_factor, d_query.view(), result_ids.view(),
                 result_distances.view(), result_file);

    auto truth = load_data<uint32_t, uint32_t>(truth_file);
    float recall = calculate_recall(result_ids.view(), truth.view(), search_param.topk);

    std::ofstream result;
    result.open(result_file, std::ios::app);
    result << recall << std::endl;
    result.close();
}