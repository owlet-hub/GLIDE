#include "raft/core/device_mdspan.hpp"
#include "raft/core/device_mdarray.hpp"
#include "raft/core/handle.hpp"
#include "raft/core/host_mdarray.hpp"
#include "params.cuh"

/**
 * @class GLIDE
 * @brief Main class for GLIDE graph index implementation for small_scale data
 *
 * @tparam Data_t Data type for points
 * @tparam Index_t Data type for addressing points (needs to be able to represent num ¡Á dim)
 */
template<typename Data_t=float, typename Index_t=uint32_t>
struct GLIDE {
public:
    /**
     * @brief Constructor of GLIDE
     *
     * @param handle RAFT device resources
     * @param graph_degree Degree of the graph
     * @param h_data_view Host matrix view of input data (num ¡Á dim)
     * @param h_centroid_view Host view of centroids (centroid_num ¡Á dim)
     * @param metric Distance metric to use
     */
    GLIDE(raft::device_resources &handle, uint32_t graph_degree,
          raft::host_matrix_view<Data_t, Index_t> h_data_view,
          raft::host_matrix_view<float> h_centroid_view, Metric metric);

    /**
     * @brief Load graph structure and start points
     *
     * @param h_graph_view Host matrix view of graph  (num ¡Á graph_degree)
     * @param h_start_point_view Host vector view of start points
     */
    void load(raft::host_matrix_view<uint32_t, Index_t> h_graph_view,
              raft::host_vector_view<uint32_t> h_start_point_view);

    ~GLIDE() = default;

    /**
     * @brief Build the GLIDE index
     *
     * @param build_param Index building parameter
     * @param search_param_knn Search parameter for build
     * @param search_param_refine Search parameter for refinement
     * @param d_knn_graph Device matrix of KNN graph (num ¡Á knn_degree)
     * @param d_segment_start Device vector of segment start ids
     * @param d_segment_length Device vector of segment lengths
     * @param d_map Device vector of mapping
     * @param h_segment_start_view Host vector view of segment start ids
     * @param h_segment_length_view Host vector view of segment lengths
     * @param h_map_view Host vector view of mapping
     * @param d_reorder_data Device matrix of reordered data (reorder_num ¡Á dim)
     * @param result_file Result file path
     */
    void build(IndexParameter &build_param, SearchParameter &search_param_knn, SearchParameter &search_param_refine,
               std::optional<raft::device_matrix<uint32_t, Index_t>> &d_knn_graph,
               std::optional<raft::device_vector<uint32_t>> &d_segment_start,
               std::optional<raft::device_vector<uint32_t>> &d_segment_length,
               std::optional<raft::device_vector<uint32_t>> &d_map,
               raft::host_vector_view<uint32_t> h_segment_start_view,
               raft::host_vector_view<uint32_t> h_segment_length_view,
               raft::host_vector_view<uint32_t> h_map_view,
               std::optional<raft::device_matrix<Data_t, Index_t>> &d_reorder_data,
               std::string &result_file);

    /**
     * @brief Search on the GLIDE index
     *
     * @param param Search parameters
     * @param d_query_view Device matrix view of query points (query_num ¡Á dim)
     * @param result_ids Host matrix of result ids (query_num ¡Á top_k)
     * @param result_distances Host matrix of result distances (query_num ¡Á top_k)
     * @param result_file Result file path
     */
    void search(SearchParameter &param, raft::device_matrix_view<Data_t> d_query_view,
                raft::host_matrix<uint32_t> &result_ids,
                raft::host_matrix<float> &result_distances, std::string &result_file);

    // Accessor methods with inline documentation
    /** @brief Get data dimensionality */
    inline uint32_t dim() {
        return (d_data.data_handle() != nullptr) ? d_data.extent(1) : 0;
    }

    /** @brief Get number of data points */
    inline uint32_t num() {
        return (d_data.data_handle() != nullptr) ? d_data.extent(0) : 0;
    }

    /** @brief Get graph degree */
    inline uint32_t graph_degree() {
        return (d_graph.data_handle() != nullptr) ? d_graph.extent(1) : 0;
    }

    /** @brief Get number of start points */
    inline uint32_t start_point_num() {
        return (d_start_points->data_handle() != nullptr) ? d_start_points->extent(0) : 0;
    }

    /** @brief Get number of centroids */
    inline uint32_t centroid_num() {
        return (d_centroids.data_handle() != nullptr) ? d_centroids.extent(0) : 0;
    }

    /** @brief Get device matrix view of data */
    inline raft::device_matrix_view<const Data_t, Index_t> data_view() {
        return d_data.view();
    }

    /** @brief Get device vector view of start points */
    inline raft::device_vector_view<const uint32_t> start_point_view() {
        return d_start_points->view();
    }

    /** @brief Get device matrix view of graph */
    inline raft::device_matrix_view<const uint32_t, Index_t> graph_view() {
        return d_graph.view();
    }

    /** @brief Get device matrix view of centroids */
    inline raft::device_matrix_view<const float> centroid_view() {
        return d_centroids.view();
    }

protected:
    /**
     * @brief Select start points
     *
     * @param h_segment_start_view Host vector view of segment start ids
     * @param h_segment_length_view Host vector view of segment lengths
     * @param h_map_view Host vector view of mapping
     * @param h_reorder_start_points Host vector view of reordered start points
     * @param h_start_points Host vector view of start points
     * @param build_time Reference to accumulate build time
     */
    void start_point_select(raft::host_vector_view<uint32_t> h_segment_start_view,
                            raft::host_vector_view<uint32_t> h_segment_length_view,
                            raft::host_vector_view<uint32_t> h_map_view,
                            raft::host_vector_view<uint32_t> h_reorder_start_points,
                            raft::host_vector_view<uint32_t> h_start_points,
                            float &build_time);

    /**
     * @brief Build and merge subgraphs
     *
     * @param param Search parameter for build
     * @param relaxant_factor Relaxation factor for pruning
     * @param d_reorder_data_view Device matrix view of reordered data
     * @param d_map_view Device vector view of mapping
     * @param d_segment_start_view Device vector view of segment start ids
     * @param d_segment_length_view Device vector view of segment lengths
     * @param d_knn_graph_view Device matrix view of KNN graph
     * @param build_time Reference to accumulate build time
     */
    void subgraph_build_and_merge(SearchParameter &param, float relaxant_factor,
                                  raft::device_matrix_view<Data_t, Index_t> d_reorder_data_view,
                                  raft::device_vector_view<uint32_t> d_map_view,
                                  raft::device_vector_view<uint32_t> d_segment_start_view,
                                  raft::device_vector_view<uint32_t> d_segment_length_view,
                                  raft::device_matrix_view<uint32_t, Index_t> d_knn_graph_view,
                                  float &build_time);

    /**
     * @brief Add reverse edges to the graph
     *
     * @param build_time Reference to accumulate build time
     */
    void reverse_graph(float &build_time);

    /**
    * @brief Refine the graph structure
    *
    * @param param Search parameter for refinement
    * @param relaxant_factor Relaxation factor for pruning
    * @param build_time Reference to accumulate build time
    */
    void refine(SearchParameter &param, float relaxant_factor, float &build_time);

private:
    raft::device_resources &handle; /// RAFT device resources

    Metric metric; /// Distance metric
    raft::device_matrix<Data_t, Index_t> d_data; /// Device matrix of data (num ¡Á dim)
    std::optional<raft::device_vector<uint32_t>> d_start_points; /// Device vector of start points
    raft::device_matrix<uint32_t, Index_t> d_graph; /// Device matrix of graph (num ¡Á graph_degree)
    raft::device_matrix<float> d_centroids; /// Device matrix of centroids (centroid_num ¡Á dim)
};