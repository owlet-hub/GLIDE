#include "raft/core/device_mdspan.hpp"
#include "raft/core/device_mdarray.hpp"
#include "raft/core/handle.hpp"
#include "raft/core/host_mdarray.hpp"
#include "params.cuh"

struct GLIDE {
public:
    GLIDE(raft::device_resources &handle, uint32_t graph_degree,
          raft::host_matrix_view<float> h_data,
          raft::host_matrix_view<float> h_centroids, Metric metric);

    void load(raft::host_matrix_view<uint32_t> h_graph,
              raft::host_vector_view<uint32_t> h_start_points);

    ~GLIDE() = default;

    void build(IndexParameter &build_param, SearchParameter &search_param_knn, SearchParameter &search_param_refine,
               std::optional<raft::device_matrix<uint32_t>> &d_knn_graph,
               std::optional<raft::device_vector<uint32_t>> &d_segment_start,
               std::optional<raft::device_vector<uint32_t>> &d_segment_length,
               std::optional<raft::device_vector<uint32_t>> &d_map,
               raft::host_vector_view<uint32_t> h_segment_start_view,
               raft::host_vector_view<uint32_t> h_segment_length_view,
               raft::host_vector_view<uint32_t> h_map_view,
               std::optional<raft::device_matrix<float>> &d_reorder_data,
               std::string result_file);

    void search(SearchParameter &param,
                raft::device_matrix_view<float> d_query_view,
                raft::host_matrix_view<uint32_t> h_result_id_view,
                raft::host_matrix_view<float> h_result_dist_view,
                std::string &result_file);

    inline uint32_t dim() {
        return (d_data.data_handle() != nullptr) ? d_data.extent(1) : 0;
    }

    inline uint32_t num() {
        return (d_data.data_handle() != nullptr) ? d_data.extent(0) : 0;
    }

    inline uint32_t graph_degree() {
        return (d_graph.data_handle() != nullptr) ? d_graph.extent(1) : 0;
    }

    inline uint32_t start_point_num() {
        return (d_start_points->data_handle() != nullptr) ? d_start_points->extent(0) : 0;
    }

    inline uint32_t centroid_num() {
        return (d_centroids.data_handle() != nullptr) ? d_centroids.extent(0) : 0;
    }

    inline raft::device_matrix_view<const float> data_view() {
        return d_data.view();
    }

    inline raft::device_vector_view<const uint32_t> start_point_view() {
        return d_start_points->view();
    }

    inline raft::device_matrix_view<const uint32_t> graph_view() {
        return d_graph.view();
    }

    inline raft::device_matrix_view<const float> centroid_view() {
        return d_centroids.view();
    }

protected:
    void start_point_select(raft::host_vector_view<uint32_t> h_segment_start_view,
                            raft::host_vector_view<uint32_t> h_segment_length_view,
                            raft::host_vector_view<uint32_t> h_map_view,
                            raft::host_vector_view<uint32_t> h_reorder_start_points,
                            raft::host_vector_view<uint32_t> h_start_points,
                            float &build_time);

    void refine(SearchParameter &param, float relaxant_factor, float &build_time);

    void subgraph_build_and_merge(SearchParameter &param, float relaxant_factor,
                                  raft::device_matrix_view<float> d_reorder_data,
                                  raft::device_vector_view<uint32_t> d_map_view,
                                  raft::device_vector_view<uint32_t> d_segment_start_view,
                                  raft::device_vector_view<uint32_t> d_segment_length_view,
                                  raft::device_matrix_view<uint32_t> d_knn_graph_view,
                                  float &build_time);

    void reverse_graph(float &build_time);

private:
    raft::device_resources &handle;

    Metric metric;
    raft::device_matrix<float> d_data;
    std::optional<raft::device_vector<uint32_t>> d_start_points;
    raft::device_matrix<uint32_t> d_graph;
    raft::device_matrix<float> d_centroids;
};