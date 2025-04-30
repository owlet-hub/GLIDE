#include "load.cuh"
#include "glide_large_impl.cuh"
#include <fstream>

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

    auto knn_graph = load_data<uint32_t, uint64_t>(knn_file);

    GLIDE_large index(handle, build_param.graph_degree, build_param.metric,
                      reorder_file, map_file, centroid_file, segment_file);

    std::ofstream result_out;
    result_out.open(result_file, std::ios::app);
    result_out << build_param.graph_degree << ",";
    result_out.close();

    index.build(build_param, search_param_knn, search_param_refine, knn_graph.view(), result_file);

    std::ofstream out(graph_file, std::ios::binary);
    uint32_t num = index.num();
    uint32_t degree = index.graph_degree();
    out.write(reinterpret_cast<const char *>(&num), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&degree), sizeof(uint32_t));
    for (uint32_t i = 0; i < num; i++) {
        uint64_t start_pos = static_cast<uint64_t>(i) * degree;
        out.write(reinterpret_cast<const char *>(index.graph_view().data_handle() + start_pos),
                  degree * sizeof(uint32_t));
    }
    out.close();

    std::ofstream start_point_out(start_point_file, std::ios::binary);
    uint32_t start_point_num = index.start_point_num();
    start_point_out.write(reinterpret_cast<const char *>(&start_point_num), sizeof(uint32_t));
    start_point_out.write(reinterpret_cast<const char *>(index.start_point_view().data_handle()),
                          start_point_num * sizeof(uint32_t));
}