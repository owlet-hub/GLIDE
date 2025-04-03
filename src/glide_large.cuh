#include "raft/core/device_mdspan.hpp"
#include "raft/core/device_mdarray.hpp"
#include "raft/core/handle.hpp"
#include "raft/core/host_mdarray.hpp"
#include "params.cuh"

struct GLIDE_for_large {
public:
    GLIDE_for_large(raft::device_resources &handle, uint32_t graph_degree, Metric metric,
        std::string reorder_file, std::string map_file, std::string centroid_file, std::string segment_file);

    GLIDE_for_large(raft::device_resources &handle, Metric metric,
        std::string reorder_file, std::string map_file, std::string centroid_file,
        std::string segment_file, std::string start_point_file, std::string graph_file);

    void load(raft::host_matrix_view<uint32_t, uint64_t> h_graph_view,
              raft::host_vector_view<uint32_t> h_start_point_view);

    ~GLIDE_for_large() = default;

    void build(IndexParameter &build_param, SearchParameter &search_param_knn, SearchParameter &search_param_refine,
               raft::host_matrix_view<uint32_t, uint64_t> h_knn_graph_view, std::string &result_file);

    void search(SearchParameter &param, uint32_t min_segment_num, float boundary_factor,
                raft::device_matrix_view<uint8_t> d_query_view,
                raft::host_matrix_view<uint32_t> h_result_id_view,
                raft::host_matrix_view<float> h_result_dist_view,
                std::string &result_file);

    inline uint32_t dim() {
        return (data.data_handle() != nullptr) ? data.extent(1) : 0;
    }

    inline uint32_t num() {
        return (data.data_handle() != nullptr) ? data.extent(0) : 0;
    }

    inline uint32_t graph_degree() {
        return (graph.data_handle() != nullptr) ? graph.extent(1) : 0;
    }

    inline uint32_t segment_num() {
        return (segment_start.data_handle() != nullptr) ? segment_start.extent(0) : 0;
    }

    inline uint32_t start_point_num() {
        return (start_points.data_handle() != nullptr) ? start_points.extent(0) : 0;
    }

    inline uint32_t centroid_num() {
        return (centroids.data_handle() != nullptr) ? centroids.extent(0) : 0;
    }

    inline raft::host_matrix_view<const uint8_t, uint64_t> data_view() {
        return data.view();
    }

    inline raft::host_vector_view<const uint32_t> start_point_view() {
        return start_points.view();
    }

    inline raft::host_matrix_view<const uint32_t, uint64_t> graph_view() {
        return graph.view();
    }

    inline raft::host_vector_view<const uint32_t> segment_start_view() {
        return segment_start.view();
    }

    inline raft::host_vector_view<const uint32_t> segment_length_view() {
        return segment_length.view();
    }

    inline raft::host_matrix_view<const float> centroids_view() {
        return centroids.view();
    }

protected:
    void start_point_select(float &build_time);

    void subgraph_build_and_merge(SearchParameter &param, float relaxant_factor, float &build_time,
                                  raft::host_matrix_view<uint32_t, uint64_t> knn_graph_view);

    void reverse_graph(float &build_time);

    void refine(SearchParameter &param, float relaxant_factor, float &build_time);

private:
    raft::device_resources &handle;

    Metric metric;
    raft::host_matrix<uint8_t, uint64_t> data;
    raft::host_vector<uint32_t> map;
    raft::host_matrix<float> centroids;
    raft::host_vector<uint32_t> segment_start;
    raft::host_vector<uint32_t> segment_length;
    raft::host_vector<uint32_t> start_points;
    raft::host_matrix<uint32_t, uint64_t> graph;
};