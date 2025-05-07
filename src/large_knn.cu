#include "load.cuh"
#include "nn_descent.cuh"
#include <fstream>

/**
 * @brief Main function for KNN graph construction pipeline for large-scale data
 *
 * @param argc Number of command line arguments
 * @param argv Command line arguments:
 *             [0] Program name
 *             [1] Input preprocess file base path
 *             [2] Output KNN graph file path
 *             [3] Result file path
 *             [4] KNN degree (uint32_t)
 *
 * @return int Program exit status (0 for success, non-zero for failure)
 */
int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << argv[0]
                  << "preprocess_file knn_file result_file knn_degree"
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
    std::string knn_file(argv[2]);
    std::string result_file(argv[3]);
    uint32_t knn_degree = std::stoi(argv[4]);

    auto h_reorder_data = load_matrix_data<uint8_t, uint64_t>(reorder_file);
    auto h_segment_start = load_segment_start(segment_file);
    auto h_segment_length = load_segment_length(segment_file);


    auto knn_index = build_knn_for_large<uint8_t, uint64_t>(handle, knn_degree, h_segment_start.view(),
                                                            h_segment_length.view(), h_reorder_data.view(),
                                                            result_file);

    save_matrix_data<int, uint64_t>(knn_file, knn_index.view());
}