#include <fstream>
#include <vector>
#include <iostream>
#include "raft/core/host_mdarray.hpp"
#include "raft/core/handle.hpp"
#include "raft/core/device_mdarray.hpp"

template<typename Data_t, typename Index_t>
raft::host_matrix<Data_t, Index_t>
load_data(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    std::cout << "Start load data from: " << path << std::endl;

    uint32_t num;
    uint32_t dim;
    file.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));

    auto data = raft::make_host_matrix<Data_t, Index_t>(num, dim);

    for (uint32_t i = 0; i < num; i++) {
        Index_t start = static_cast<Index_t>(i) * dim;
        file.read(reinterpret_cast<char *>(data.data_handle() + start), dim * sizeof(Data_t));
    }
    file.close();

    return data;
}

raft::host_matrix<float>
load_data(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    uint32_t num_vectors;
    uint32_t num_dim;

    std::cout << "Start load data from: " << path << std::endl;

    file.read(reinterpret_cast<char *>(&num_vectors), sizeof(num_vectors));
    file.read(reinterpret_cast<char *>(&num_dim), sizeof(num_dim));

    auto data = raft::make_host_matrix<float>(num_vectors, num_dim);

    for (uint32_t i = 0; i < num_vectors; i++) {
        uint32_t start = i * num_dim;
        file.read(reinterpret_cast<char *>(data.data_handle()+start), num_dim * sizeof(float));
    }
    file.close();

    return data;
}

raft::device_matrix<float>
load_query(const std::string &path, raft::device_resources &handle) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    uint32_t num_queries;
    uint32_t num_dim;

    std::cout << "Start load data from: " << path << std::endl;

    file.read(reinterpret_cast<char *>(&num_queries), sizeof(num_queries));
    file.read(reinterpret_cast<char *>(&num_dim), sizeof(num_dim));

    std::vector<std::vector<float>> query;
    query.resize(num_queries);
    for (uint32_t i = 0; i < num_queries; i++) {
        query[i].resize(num_dim);
        file.read(reinterpret_cast<char *>(query[i].data()), num_dim * sizeof(float));
    }
    file.close();

    auto d_query = raft::make_device_matrix<float>(handle, num_queries, num_dim);
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    for (uint32_t i = 0; i < num_queries; i++) {
        uint32_t start = i * num_dim;
        raft::copy(d_query.data_handle() + start, query[i].data(), num_dim, stream);
    }
    raft::resource::sync_stream(handle, stream);

    return d_query;
}

//raft::host_matrix<uint32_t>
//load_truth(const std::string &path) {
//    uint32_t num_queries;
//    uint32_t K;
//
//    std::ifstream file(path, std::ios::binary);
//    if (!file) {
//        std::cerr << "Error opening file: " << path << std::endl;
//    }
//
//    std::cout << "Start load data from: " << path << std::endl;
//
//    file.read(reinterpret_cast<char *>(&num_queries), sizeof(num_queries));
//    file.read(reinterpret_cast<char *>(&K), sizeof(K));
//
//    auto truth = raft::make_host_matrix<uint32_t>(num_queries, K);
//
//    for (uint32_t i = 0; i < num_queries; i++) {
//        uint32_t start = i * K;
//        file.read(reinterpret_cast<char *>(truth.data_handle()+start), K * sizeof(uint32_t));
//    }
//    file.close();
//
//    return truth;
//}

raft::host_vector<uint32_t>
load_segment_start(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    std::cout << "Start load data from: " << path << std::endl;

    uint32_t segment_num;
    file.read(reinterpret_cast<char *>(&segment_num), sizeof(uint32_t));

    auto segment_start = raft::make_host_vector<uint32_t>(segment_num);

    file.read(reinterpret_cast<char *>(segment_start.data_handle()), segment_num * sizeof(uint32_t));
    file.close();

    return segment_start;
}

raft::host_vector<uint32_t>
load_segment_length(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    std::cout << "Start load data from: " << path << std::endl;

    uint32_t segment_num;
    file.read(reinterpret_cast<char *>(&segment_num), sizeof(uint32_t));

    auto segment_length = raft::make_host_vector<uint32_t>(segment_num);
    auto temp = raft::make_host_vector<uint32_t>(segment_num);

    file.read(reinterpret_cast<char *>(temp.data_handle()), segment_num * sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(segment_length.data_handle()), segment_num * sizeof(uint32_t));
    file.close();

    return segment_length;
}

raft::host_vector<uint32_t>
load_map(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    std::cout << "Start load data from: " << path << std::endl;

    uint32_t num;
    file.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));

    auto map = raft::make_host_vector<uint32_t>(num);

    file.read(reinterpret_cast<char *>(map.data_handle()), num * sizeof(uint32_t));
    file.close();

    return map;
}

template<typename ID_t, typename Index_t>
raft::host_matrix<ID_t, Index_t>
load_graph(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    std::cout << "Start load data from: " << path << std::endl;

    uint32_t num, degree;
    file.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&degree), sizeof(uint32_t));

    auto knn_graph = raft::make_host_matrix<ID_t, Index_t>(num, degree);

    for (uint32_t i = 0; i < num; i++) {
        Index_t start = static_cast<Index_t>(i) * degree;
        file.read(reinterpret_cast<char *>(knn_graph.data_handle() + start), degree * sizeof(ID_t));
    }
    file.close();

    return knn_graph;
}

//raft::host_matrix<uint32_t>
//load_graph(const std::string &path) {
//    std::ifstream file(path, std::ios::binary);
//    if (!file) {
//        std::cerr << "Error opening file: " << path << std::endl;
//    }
//
//    std::cout << "Start load data from: " << path << std::endl;
//
//    uint32_t number, degree;
//    file.read(reinterpret_cast<char *>(&number), sizeof(uint32_t));
//    file.read(reinterpret_cast<char *>(&degree), sizeof(uint32_t));
//
//    auto knn_graph = raft::make_host_matrix<uint32_t>(number, degree);
//
//    for (uint32_t i = 0; i < number; i++) {
//        uint32_t start = i * degree;
//        file.read(reinterpret_cast<char *>(knn_graph.data_handle()+start), degree * sizeof(uint32_t));
//    }
//    file.close();
//
//    return knn_graph;
//}

raft::host_matrix<uint32_t, uint64_t>
load_graph_large(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }

    std::cout << "Start load data from: " << path << std::endl;

    uint32_t number, degree;
    file.read(reinterpret_cast<char *>(&number), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&degree), sizeof(uint32_t));

    auto knn_graph = raft::make_host_matrix<uint32_t, uint64_t>(number, degree);

    for (uint32_t i = 0; i < number; i++) {
        uint64_t start = static_cast<uint64_t>(i) * degree;
        file.read(reinterpret_cast<char *>(knn_graph.data_handle()+start), degree * sizeof(uint32_t));
    }
    file.close();

    return knn_graph;
}

//raft::host_matrix<float>
//load_centroids(const std::string &path) {
//    std::ifstream file(path, std::ios::binary);
//    if (!file) {
//        std::cerr << "Error opening file: " << path << std::endl;
//    }
//
//    std::cout << "Start load data from: " << path << std::endl;
//    uint32_t centroid_num, dim;
//    file.read(reinterpret_cast<char *>(&centroid_num), sizeof(uint32_t));
//    file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
//
//    auto centroids = raft::make_host_matrix<float>(centroid_num, dim);
//
//    for(uint32_t i=0;i<centroid_num;i++){
//        uint32_t start_pos = i*dim;
//        file.read(reinterpret_cast<char *>(centroids.data_handle()+start_pos), dim*sizeof(float));
//    }
//    file.close();
//
//    return centroids;
//}

raft::host_vector<uint32_t>
load_start_point(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    std::cout << "Start load data from: " << path << std::endl;

    uint32_t start_point_num;
    file.read(reinterpret_cast<char *>(&start_point_num), sizeof(uint32_t));

    auto start_point = raft::make_host_vector<uint32_t>(start_point_num);

    file.read(reinterpret_cast<char *>(start_point.data_handle()), start_point_num * sizeof(uint32_t));
    file.close();

    return start_point;
}

raft::host_matrix<uint8_t, uint64_t>
load_data_uint8(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    uint32_t num_vectors;
    uint32_t num_dim;

    std::cout << "Start load data from: " << path << std::endl;

    file.read(reinterpret_cast<char *>(&num_vectors), sizeof(num_vectors));
    file.read(reinterpret_cast<char *>(&num_dim), sizeof(num_dim));

    auto data = raft::make_host_matrix<uint8_t, uint64_t>(num_vectors, num_dim);

    for (uint32_t i = 0; i < num_vectors; i++) {
        uint64_t start = static_cast<uint64_t>(i) * num_dim;
        file.read(reinterpret_cast<char *>(data.data_handle()+start), num_dim * sizeof(uint8_t));
    }
    file.close();

    return data;
}

raft::device_matrix<uint8_t>
load_query_uint8(const std::string &path, raft::device_resources &handle) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }
    uint32_t num_queries;
    uint32_t num_dim;

    std::cout << "Start load data from: " << path << std::endl;

    file.read(reinterpret_cast<char *>(&num_queries), sizeof(num_queries));
    file.read(reinterpret_cast<char *>(&num_dim), sizeof(num_dim));

    std::vector<std::vector<uint8_t>> query;
    query.resize(num_queries);
    for (uint32_t i = 0; i < num_queries; i++) {
        query[i].resize(num_dim);
        file.read(reinterpret_cast<char *>(query[i].data()), num_dim * sizeof(uint8_t));
    }
    file.close();

    auto d_query = raft::make_device_matrix<uint8_t>(handle, num_queries, num_dim);
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    for (uint32_t i = 0; i < num_queries; i++) {
        uint32_t start = i * num_dim;
        raft::copy(d_query.data_handle() + start, query[i].data(), num_dim, stream);
    }
    raft::resource::sync_stream(handle, stream);

    return d_query;
}