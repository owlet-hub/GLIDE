#include "load.cuh"
#include "nn_descent.cuh"
#include <fstream>

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

    auto h_reorder_data = load_data<float, uint32_t>(reorder_file);
    auto h_segment_start = load_segment_start(segment_file);
    auto h_segment_length = load_segment_length(segment_file);

    NNDescentParameter nnd_param(knn_degree);

    std::optional<raft::device_matrix<float>> d_reorder_data;
    d_reorder_data.emplace(raft::make_device_matrix<float>(handle, h_reorder_data.extent(0), h_reorder_data.extent(1)));
    raft::copy(d_reorder_data->data_handle(), h_reorder_data.data_handle(),
               h_reorder_data.size(), raft::resource::get_cuda_stream(handle));

    auto knn_index = build_nnd(handle, nnd_param, d_reorder_data,
                               h_segment_start.view(), h_segment_length.view(), result_file);

    uint32_t num = knn_index.extent(0);
    uint32_t degree = knn_index.extent(1);
    std::ofstream out(knn_file, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&num), sizeof(num));
    out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
    for (uint32_t i = 0; i < num; i++) {
        uint64_t start_pos = i * degree;
        out.write(reinterpret_cast<const char *>(knn_index.data_handle() + start_pos), degree * sizeof(uint32_t));
    }
    out.close();
}