#include "load.cuh"
#include "partition.cuh"
#include "params.cuh"
#include <fstream>

/**
 * @brief Main function for data preprocessing pipeline for large-scale data
 *
 * @param argc Number of command line arguments
 * @param argv Command line arguments:
 *             [0] Program name
 *             [1] Input data file path
 *             [2] Output preprocess file base path
 *             [3] Result file path
 *             [4] Number of centroids (uint32_t)
 *             [5] Sample factor for sampling-based cluster (float)
 *             [6] Distance metric ("Euclidean" or "Cosine")
 *
 * @return int Program exit status (0 for success, non-zero for failure)
 */
int main(int argc, char **argv) {
    if (argc != 7) {
        std::cout << argv[0]
                  << "data_file preprocess_file result_file centroid_num sample_factor metric"
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
    partition_param.sample_factor = std::stof(argv[5]);
    std::string metric(argv[6]);
    if (metric == "Euclidean") {
        partition_param.metric = Metric::Euclidean;
    } else if (metric == "Cosine") {
        partition_param.metric = Metric::Cosine;
    }

    auto h_data = load_matrix_data<uint8_t, uint64_t>(data_file);
    uint32_t num = h_data.extent(0);
    uint32_t dim = h_data.extent(1);
    uint32_t batch_num = 10;
    uint32_t batch_size = num / batch_num;

    auto h_reorder_data = raft::make_host_matrix<uint8_t, uint64_t>(num, dim);
    auto h_map = raft::make_host_vector<uint32_t>(num);
    auto h_segment_start = raft::make_host_vector<uint32_t>(partition_param.centroid_num);
    auto h_segment_length = raft::make_host_vector<uint32_t>(partition_param.centroid_num);
    auto d_centroids = raft::make_device_matrix<float, int>(handle, partition_param.centroid_num, dim);

    auto d_data = raft::make_device_matrix<uint8_t, uint64_t>(handle, batch_size, dim);
    preprocess_for_large<uint8_t, uint64_t>(handle, partition_param, h_data.view(), d_data.view(),
                                            h_reorder_data.view(), h_map.view(), h_segment_start.view(),
                                            h_segment_length.view(), d_centroids.view(), batch_num, result_file);

    auto centroids = raft::make_host_matrix<float>(partition_param.centroid_num, dim);
    raft::copy(centroids.data_handle(), d_centroids.data_handle(), centroids.size(),
               raft::resource::get_stream_from_stream_pool(handle));

    save_matrix_data<float, uint32_t>(centroid_file, centroids.view());
    save_segment(segment_file, h_segment_start.view(), h_segment_length.view());
    save_vector_data(map_file, h_map.view());
    save_matrix_data<uint8_t, uint64_t>(reorder_file, h_reorder_data.view());
}