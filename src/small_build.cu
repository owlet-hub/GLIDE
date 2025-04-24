#include "load.cuh"
#include "glide_impl.cuh"
#include <fstream>

int main(int argc, char **argv) {
    if (argc != 12) {
        std::cout << argv[0]
                  << "data_file preprocess_file knn_file graph_base_file result_file metric graph_degree knn_degree relaxant_factor beam refine_beam"
                  << std::endl;
        exit(-1);
    }

    cudaSetDevice(1);
    raft::device_resources handle;
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(32);
    raft::resource::set_cuda_stream_pool(handle, stream_pool);

    std::string data_file(argv[1]);
    std::string preprocess_file(argv[2]);
    std::string knn_file(argv[3]);
    std::string graph_base_file(argv[4]);
    std::string result_file(argv[5]);
    std::string centroid_file = preprocess_file + ".centroids";
    std::string segment_file = preprocess_file + ".segment";
    std::string map_file = preprocess_file + ".map";
    std::string reorder_file = preprocess_file + ".reorder";
    std::string graph_file = graph_base_file + ".graph";
    std::string start_point_file = graph_base_file + ".sp";
    std::string metric(argv[6]);

    IndexParameter build_param;
    build_param.graph_degree = std::stoi(argv[7]);
    build_param.knn_degree = std::stoi(argv[8]);
    build_param.relaxant_factor = std::stof(argv[9]);
    if (metric == "Euclidean") {
        build_param.metric = Metric::Euclidean;
    } else if (metric == "Cosine") {
        build_param.metric = Metric::Cosine;
    }

    SearchParameter search_param_knn;
    search_param_knn.beam = std::stoi(argv[10]);
    adjust_search_params(search_param_knn.min_iterations, search_param_knn.max_iterations, search_param_knn.beam);
    search_param_knn.hash_bit = calculate_hash_bitlen(search_param_knn.beam, build_param.knn_degree,
                                                      search_param_knn.hash_max_fill_rate,
                                                      search_param_knn.hashmap_min_bitlen);
    search_param_knn.hash_reset_interval = calculate_hash_reset_interval(search_param_knn.beam,
                                                                         build_param.knn_degree,
                                                                         search_param_knn.hash_max_fill_rate,
                                                                         search_param_knn.hash_bit);

    SearchParameter search_param_refine;
    search_param_refine.beam = std::stoi(argv[11]);
    adjust_search_params(search_param_refine.min_iterations, search_param_refine.max_iterations,
                         search_param_refine.beam);
    search_param_refine.hash_bit = calculate_hash_bitlen(search_param_refine.beam, build_param.knn_degree,
                                                         search_param_refine.hash_max_fill_rate,
                                                         search_param_refine.hashmap_min_bitlen);
    search_param_refine.hash_reset_interval = calculate_hash_reset_interval(search_param_refine.beam,
                                                                            build_param.knn_degree,
                                                                            search_param_refine.hash_max_fill_rate,
                                                                            search_param_refine.hash_bit);

    auto h_data = load_data<float, uint32_t>(data_file);
    auto h_centroids = load_data<float, uint32_t>(centroid_file);
    auto h_segment_start = load_segment_start(segment_file);
    auto h_segment_length = load_segment_length(segment_file);
    auto h_map = load_map(map_file);
    auto h_reorder_data = load_data<float, uint32_t>(reorder_file);
    auto h_knn_graph = load_data<uint32_t, uint32_t>(knn_file);

    std::optional<raft::device_matrix<float>> d_reorder_data;
    std::optional<raft::device_vector<uint32_t>> d_map;
    std::optional<raft::device_vector<uint32_t>> d_segment_start;
    std::optional<raft::device_vector<uint32_t>> d_segment_length;
    std::optional<raft::device_matrix<uint32_t>> d_knn_graph;

    d_reorder_data.emplace(raft::make_device_matrix<float>(handle, h_reorder_data.extent(0), h_reorder_data.extent(1)));
    d_map.emplace(raft::make_device_vector<uint32_t, uint32_t>(handle, h_map.size()));
    d_segment_start.emplace(raft::make_device_vector<uint32_t, uint32_t>(handle, h_segment_start.size()));
    d_segment_length.emplace(raft::make_device_vector<uint32_t, uint32_t>(handle, h_segment_length.size()));
    d_knn_graph.emplace(raft::make_device_matrix<uint32_t>(handle, h_knn_graph.extent(0), h_knn_graph.extent(1)));

    raft::copy(d_reorder_data->data_handle(), h_reorder_data.data_handle(),
               h_reorder_data.size(), raft::resource::get_cuda_stream(handle));
    raft::copy(d_map->data_handle(), h_map.data_handle(),
               h_map.size(), raft::resource::get_cuda_stream(handle));
    raft::copy(d_segment_start->data_handle(), h_segment_start.data_handle(),
               h_segment_start.size(), raft::resource::get_cuda_stream(handle));
    raft::copy(d_segment_length->data_handle(), h_segment_length.data_handle(),
               h_segment_length.size(), raft::resource::get_cuda_stream(handle));
    raft::copy(d_knn_graph->data_handle(), h_knn_graph.data_handle(),
               h_knn_graph.size(), raft::resource::get_cuda_stream(handle));

    GLIDE index(handle, build_param.graph_degree, h_data.view(), h_centroids.view(), build_param.metric);

    std::ofstream result_out;
    result_out.open(result_file, std::ios::app);
    result_out << build_param.graph_degree << ",";
    result_out.close();

    index.build(build_param, search_param_knn, search_param_refine, d_knn_graph, d_segment_start, d_segment_length,
                d_map, h_segment_start.view(), h_segment_length.view(), h_map.view(), d_reorder_data,
                result_file);

    uint32_t num = index.num();
    uint32_t degree = index.graph_degree();
    auto graph = raft::make_host_matrix<uint32_t>(num, degree);
    raft::copy(graph.data_handle(), index.graph_view().data_handle(), graph.size(),
               raft::resource::get_cuda_stream(handle));
    auto start_point = raft::make_host_vector<uint32_t>(index.start_point_num());
    raft::copy(start_point.data_handle(), index.start_point_view().data_handle(), start_point.size(),
               raft::resource::get_cuda_stream(handle));

    std::ofstream out(graph_file, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&num), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&degree), sizeof(uint32_t));
    for (int i = 0; i < num; i++) {
        uint32_t start_pos = i * degree;
        out.write(reinterpret_cast<const char *>(graph.data_handle() + start_pos), degree * sizeof(uint32_t));
    }
    out.close();

    std::ofstream start_point_out(start_point_file, std::ios::binary);
    uint32_t start_point_size = index.start_point_num();
    start_point_out.write(reinterpret_cast<const char *>(&start_point_size), sizeof(uint32_t));
    start_point_out.write(reinterpret_cast<const char *>(start_point.data_handle()),
                          start_point_size * sizeof(uint32_t));
    start_point_out.close();
}