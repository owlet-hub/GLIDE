#include "load.cuh"
#include "glide_large_impl.cuh"
#include <fstream>

/**
 * @brief Main function for building the GLIDE graph index for large-scale data
 *
 * @param argc Number of command line arguments
 * @param argv Command line arguments:
 *             [0] Program name
 *             [1] Input preprocess file base path
 *             [2] Input KNN graph file path
 *             [3] Output Graph file base path
 *             [4] Result file path
 *             [5] Distance metric ("Euclidean" or "Cosine")
 *             [6] Graph degree (uint32_t)
 *             [7] KNN degree (uint32_t)
 *             [8] Relaxation factor for pruning (float)
 *             [9] Search beam size for build (uint32_t)
 *             [10] Search beam size for refinement (uint32_t)
 *
 * @return int Program exit status (0 for success, non-zero for failure)
 */
int main(int argc, char **argv) {
    if (argc != 11) {
        std::cout << argv[0]
                  << "preprocess_file knn_file graph_base_file result_file metric graph_degree knn_degree relaxant_factor beam refine_beam"
                  << std::endl;
        exit(-1);
    }

    cudaSetDevice(1);
    raft::device_resources handle;
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(32);
    raft::resource::set_cuda_stream_pool(handle, stream_pool);

    std::string preprocess_file(argv[1]);
    std::string knn_file(argv[2]);
    std::string graph_base_file(argv[3]);
    std::string result_file(argv[4]);
    std::string centroid_file = preprocess_file + ".centroid";
    std::string segment_file = preprocess_file + ".segment";
    std::string map_file = preprocess_file + ".map";
    std::string reorder_file = preprocess_file + ".reorder";
    std::string graph_file = graph_base_file + ".graph";
    std::string start_point_file = graph_base_file + ".sp";
    std::string metric(argv[5]);

    IndexParameter build_param;
    build_param.graph_degree = std::stoi(argv[6]);
    build_param.knn_degree = std::stoi(argv[7]);
    build_param.relaxant_factor = std::stof(argv[8]);
    if (metric == "Euclidean") {
        build_param.metric = Metric::Euclidean;
    } else if (metric == "Cosine") {
        build_param.metric = Metric::Cosine;
    }

    SearchParameter search_param_knn;
    search_param_knn.beam = std::stoi(argv[9]);
    adjust_search_params(search_param_knn.min_iterations, search_param_knn.max_iterations, search_param_knn.beam);
    search_param_knn.hash_bit = calculate_hash_bitlen(search_param_knn.beam, build_param.knn_degree,
                                                      search_param_knn.hash_max_fill_rate,
                                                      search_param_knn.hashmap_min_bitlen);
    search_param_knn.hash_reset_interval = calculate_hash_reset_interval(search_param_knn.beam,
                                                                         build_param.knn_degree,
                                                                         search_param_knn.hash_max_fill_rate,
                                                                         search_param_knn.hash_bit);

    SearchParameter search_param_refine;
    search_param_refine.beam = std::stoi(argv[10]);
    adjust_search_params(search_param_refine.min_iterations, search_param_refine.max_iterations,
                         search_param_refine.beam);
    search_param_refine.hash_bit = calculate_hash_bitlen(search_param_refine.beam, build_param.knn_degree,
                                                         search_param_refine.hash_max_fill_rate,
                                                         search_param_refine.hashmap_min_bitlen);
    search_param_refine.hash_reset_interval = calculate_hash_reset_interval(search_param_refine.beam,
                                                                            build_param.knn_degree,
                                                                            search_param_refine.hash_max_fill_rate,
                                                                            search_param_refine.hash_bit);

    auto knn_graph = load_matrix_data<uint32_t, uint64_t>(knn_file);

    GLIDE_large<uint8_t, uint64_t> index(handle, build_param.graph_degree, build_param.metric,
                                         reorder_file, map_file, centroid_file, segment_file);

    std::ofstream result_out;
    result_out.open(result_file, std::ios::app);
    result_out << build_param.graph_degree << ",";
    result_out.close();

    index.build(build_param, search_param_knn, search_param_refine, knn_graph.view(), result_file);

    save_matrix_data<uint32_t, uint64_t>(graph_file, index.graph_view());
    save_vector_data(start_point_file, index.start_point_view());
}