#include "load.cuh"
#include "partition.cuh"
#include "params.cuh"
#include <fstream>

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
    std::string centroid_file = preprocess_file + ".centroids";
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

    auto h_data = load_data<uint8_t, uint64_t>(data_file);
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

    preprocess_for_large(handle, partition_param, h_data.view(), d_data.view(), h_reorder_data.view(),
                         h_map.view(), h_segment_start.view(), h_segment_length.view(), d_centroids.view(),
                         batch_num, result_file);

    for (uint32_t i = 0; i < h_segment_start.size(); i++) {
        std::cout << h_segment_start(i) << " " << h_segment_start(i) + h_segment_length(i) << " " << h_segment_length(i)
                  << std::endl;
    }

    auto h_centroids = raft::make_host_matrix<float>(partition_param.centroid_num, dim);
    raft::copy(h_centroids.data_handle(), d_centroids.data_handle(),
               h_centroids.size(), raft::resource::get_stream_from_stream_pool(handle));

    std::ofstream centroid_out(centroid_file, std::ios::binary);
    centroid_out.write(reinterpret_cast<const char *>(&partition_param.centroid_num), sizeof(uint32_t));
    centroid_out.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
    for (uint32_t i = 0; i < partition_param.centroid_num; i++) {
        uint32_t start_pos = i * dim;
        centroid_out.write(reinterpret_cast<const char *>(h_centroids.data_handle() + start_pos), dim * sizeof(float));
    }
    centroid_out.close();

    std::ofstream segment_out(segment_file, std::ios::binary);
    uint32_t segment_num = h_segment_start.size();
    segment_out.write(reinterpret_cast<const char *>(&segment_num), sizeof(uint32_t));
    segment_out.write(reinterpret_cast<const char *>(h_segment_start.data_handle()), segment_num * sizeof(uint32_t));
    segment_out.write(reinterpret_cast<const char *>(h_segment_length.data_handle()), segment_num * sizeof(uint32_t));
    segment_out.close();

    std::ofstream map_out(map_file, std::ios::binary);
    map_out.write(reinterpret_cast<const char *>(&num), sizeof(uint32_t));
    map_out.write(reinterpret_cast<const char *>(h_map.data_handle()), num * sizeof(uint32_t));
    map_out.close();

    std::ofstream reorder_out(reorder_file, std::ios::binary);
    reorder_out.write(reinterpret_cast<const char *>(&num), sizeof(uint32_t));
    reorder_out.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
    for (uint32_t i = 0; i < num; i++) {
        uint64_t start_pos = static_cast<uint64_t>(i) * dim;
        reorder_out.write(reinterpret_cast<const char *>(h_reorder_data.data_handle() + start_pos),
                          dim * sizeof(uint8_t));
    }
    reorder_out.close();
}