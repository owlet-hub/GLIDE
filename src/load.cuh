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