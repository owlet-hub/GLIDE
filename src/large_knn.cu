#include "load.cuh"
#include "nn_descent.cuh"
#include <fstream>

int main(int argc, char **argv) {
//    if (argc != 5) {
//        std::cout << argv[0]
//                  << "preprocess_file knn_file result_file knn_degree"
//                  << std::endl;
//        exit(-1);
//    }

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

    auto reorder_data = load_data_uint8(reorder_file);
    auto segment_start = load_segment_start(segment_file);
    auto segment_length = load_segment_length(segment_file);


    auto knn_index = build_knn_for_large_dataset(handle, knn_degree, segment_start.view(),
                                                 segment_length.view(), reorder_data.view(), result_file);

    uint32_t number = knn_index.extent(0);
    uint32_t degree = knn_index.extent(1);
    std::ofstream out(knn_file, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&number), sizeof(number));
    out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
    for(uint32_t i=0;i<number;i++){
        uint64_t start_pos = static_cast<uint64_t>(i)*degree;
        out.write(reinterpret_cast<const char *>(knn_index.data_handle()+start_pos), degree * sizeof(uint32_t));
    }
    out.close();
}