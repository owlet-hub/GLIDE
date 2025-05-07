#include <fstream>
#include <vector>
#include <iostream>
#include "raft/core/host_mdarray.hpp"
#include "raft/core/handle.hpp"
#include "raft/core/device_mdarray.hpp"

template<typename Data_t, typename Index_t>
raft::host_matrix<Data_t, Index_t>
load_matrix_data(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }

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
load_vector_data(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
    }

    uint32_t num;
    file.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));

    auto data = raft::make_host_vector<uint32_t>(num);

    file.read(reinterpret_cast<char *>(data.data_handle()), num * sizeof(uint32_t));
    file.close();

    return data;
}

template<typename Data_t, typename Index_t>
void
save_matrix_data(const std::string &path, raft::host_matrix_view<const Data_t, Index_t> h_matrix_view) {
    std::ofstream file(path, std::ios::binary);

    uint32_t num = h_matrix_view.extent(0);
    uint32_t dim = h_matrix_view.extent(1);
    file.write(reinterpret_cast<const char *>(&num), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));

    for (uint32_t i = 0; i < num; i++) {
        Index_t start = static_cast<Index_t>(i) * dim;
        file.write(reinterpret_cast<const char *>(h_matrix_view.data_handle() + start), dim * sizeof(Data_t));
    }
    file.close();
}

void
save_segment(const std::string &path,
             raft::host_vector_view<uint32_t> h_segment_start_view,
             raft::host_vector_view<uint32_t> h_segment_length_view) {
    std::ofstream file(path, std::ios::binary);

    uint32_t segment_num = h_segment_start_view.size();
    file.write(reinterpret_cast<const char *>(&segment_num), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(h_segment_start_view.data_handle()), segment_num * sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(h_segment_length_view.data_handle()), segment_num * sizeof(uint32_t));

    file.close();
}

void
save_vector_data(const std::string &path, raft::host_vector_view<const uint32_t> h_vector_view) {
    std::ofstream file(path, std::ios::binary);

    uint32_t num = h_vector_view.size();

    file.write(reinterpret_cast<const char *>(&num), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(h_vector_view.data_handle()), num * sizeof(uint32_t));
    file.close();
}