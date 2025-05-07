#include "load.cuh"
#include "params.cuh"
#include "partition.cuh"
#include <fstream>

/**
 * @brief Main function for data preprocessing pipeline for small-scale data
 *
 * @param argc Number of command line arguments
 * @param argv Command line arguments:
 *             [0] Program name
 *             [1] Input data file path
 *             [2] Output preprocess file base path
 *             [3] Result file path
 *             [4] Number of centroids (uint32_t)
 *             [5] Boundary factor for boundary point identification (float)
 *             [6] Sample factor for sampling-based cluster (float)
 *             [7] Distance metric ("Euclidean" or "Cosine")
 *
 * @return int Program exit status (0 for success, non-zero for failure)
 */
int main(int argc, char **argv) {
    if (argc != 8) {
        std::cout << argv[0]
                  << "data_file preprocess_file result_file centroid_num boundary_factor sample_factor metric"
                  << std::endl;
        exit(-1);
    }

    cudaSetDevice(1);
    raft::device_resources handle;
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(32);
    raft::resource::set_cuda_stream_pool(handle, stream_pool);

    std::string data_file(argv[1]);
    std::string preprocess_file(argv[2]);
    std::string result_file(argv[3]);
    std::string centroid_file = preprocess_file + ".centroid";
    std::string segment_file = preprocess_file + ".segment";
    std::string map_file = preprocess_file + ".map";
    std::string reorder_file = preprocess_file + ".reorder";

    PartitionParameter partition_param;
    partition_param.centroid_num = std::stoi(argv[4]);
    partition_param.boundary_factor = std::stof(argv[5]);
    partition_param.sample_factor = std::stof(argv[6]);
    std::string metric(argv[7]);
    if (metric == "Euclidean") {
        partition_param.metric = Metric::Euclidean;
    } else if (metric == "Cosine") {
        partition_param.metric = Metric::Cosine;
    }

    auto h_data = load_matrix_data<float, uint32_t>(data_file);
    uint32_t num = h_data.extent(0);
    uint32_t dim = h_data.extent(1);

    std::optional<raft::device_matrix<float>> d_reorder_data;
    std::optional<raft::device_vector<uint32_t>> d_map;
    auto h_segment_start = raft::make_host_vector<uint32_t>(partition_param.centroid_num + 1);
    auto h_segment_length = raft::make_host_vector<uint32_t>(partition_param.centroid_num + 1);
    auto d_centroids = raft::make_device_matrix<float, int>(handle, partition_param.centroid_num, dim);


    auto d_data = raft::make_device_matrix<float>(handle, num, dim);
    raft::copy(d_data.data_handle(), h_data.data_handle(),
               num * dim, raft::resource::get_cuda_stream(handle));

    preprocess<float, uint32_t>(handle, partition_param, d_data.view(), d_reorder_data, d_map, h_segment_start.view(),
                                h_segment_length.view(), d_centroids.view(), result_file);

    auto h_centroids = raft::make_host_matrix<float>(partition_param.centroid_num, dim);
    raft::copy(h_centroids.data_handle(), d_centroids.data_handle(),
               h_centroids.size(), raft::resource::get_stream_from_stream_pool(handle));
    auto h_map = raft::make_host_vector<uint32_t>(d_map->size());
    raft::copy(h_map.data_handle(), d_map->data_handle(),
               h_map.size(), raft::resource::get_stream_from_stream_pool(handle));
    auto h_reorder_data = raft::make_host_matrix<float>(d_reorder_data->extent(0), d_reorder_data->extent(1));
    raft::copy(h_reorder_data.data_handle(), d_reorder_data->data_handle(),
               h_reorder_data.size(), raft::resource::get_stream_from_stream_pool(handle));

    save_matrix_data<float, uint32_t>(centroid_file, h_centroids.view());
    save_segment(segment_file, h_segment_start.view(), h_segment_length.view());
    save_vector_data(map_file, h_map.view());
    save_matrix_data<float, uint32_t>(reorder_file, h_reorder_data.view());
}