#include "raft/core/device_mdspan.hpp"
#include "raft/core/device_mdarray.hpp"
#include "raft/core/handle.hpp"
#include "raft/core/host_mdarray.hpp"
#include "params.cuh"

/**
 * @class GLIDE
 * @brief Main class for GLIDE graph index implementation for large-scale data
 *
 * @tparam Data_t Data type for points
 * @tparam Index_t Data type for addressing points (needs to be able to represent num × dim)
 */
template<typename Data_t=uint8_t, typename Index_t=uint64_t>
struct GLIDE_large {
public:
    /**
     * @brief Constructor of GLIDE
     *
     * @param handle RAFT device resources
     * @param graph_degree Degree of the graph
     * @param metric Distance metric to use
     * @param reorder_file Reordered data file path
     * @param map_file Mapping file path
     * @param centroid_file Centroids file path
     * @param segment_file Segments file path
     */
    GLIDE_large(raft::device_resources &handle, uint32_t graph_degree, Metric metric,
                std::string &reorder_file, std::string &map_file,
                std::string &centroid_file, std::string &segment_file);

    /**
     * @brief Constructor for GLIDE with graph file
     *
     * @param handle RAFT device resources
     * @param metric Distance metric to use
     * @param reorder_file Reordered data file path
     * @param map_file Mapping file path
     * @param centroid_file Centroids file path
     * @param segment_file Segments file path
     * @param start_point_file Start points file path
     * @param graph_file Graph file path
     */
    GLIDE_large(raft::device_resources &handle, Metric metric,
                std::string &reorder_file, std::string &map_file, std::string &centroid_file,
                std::string &segment_file, std::string &start_point_file, std::string &graph_file);

    /**
     * @brief Load graph structure and start points
     *
     * @param h_graph_view Host matrix view of graph (num × graph_degree)
     * @param h_start_point_view Host vector view of start points
     */
    void load(raft::host_matrix_view<uint32_t, Index_t> h_graph_view,
              raft::host_vector_view<uint32_t> h_start_point_view);

    ~GLIDE_large() = default;

    /**
     * @brief Build the GLIDE index
     *
     * @param build_param Index building parameter
     * @param search_param_knn Search parameter for build
     * @param search_param_refine Search parameter for refinement
     * @param h_knn_graph_view Host matrix view of KNN graph (num × knn_degree)
     * @param result_file Result file path
     */
    void build(IndexParameter &build_param, SearchParameter &search_param_knn, SearchParameter &search_param_refine,
               raft::host_matrix_view<uint32_t, Index_t> h_knn_graph_view, std::string &result_file);

    /**
     * @brief Search on the GLIDE index
     *
     * @param param Search parameters
     * @param min_segment_num Minimum number of sub-graphs to be searched
     * @param boundary_factor Boundary factor for boundary point identification
     * @param d_query_view Device matrix view of query points (query_num × dim)
     * @param result_ids Host matrix of result ids (query_num × top_k)
     * @param result_distances Host matrix of result distances (query_num × top_k)
     * @param result_file Result file path
     */
    void search(SearchParameter &param, uint32_t min_segment_num, float boundary_factor,
                raft::device_matrix_view<Data_t, Index_t> d_query_view,
                raft::host_matrix_view<uint32_t> h_result_ids_view,
                raft::host_matrix_view<float> h_result_distances_view,
                std::string &result_file);

    // Accessor methods with inline documentation
    /** @brief Get data dimensionality */
    uint32_t dim() {
        return (h_data.data_handle() != nullptr) ? h_data.extent(1) : 0;
    }

    /** @brief Get number of data points */
    uint32_t num() {
        return (h_data.data_handle() != nullptr) ? h_data.extent(0) : 0;
    }

    /** @brief Get graph degree */
    uint32_t graph_degree() {
        return (h_graph.data_handle() != nullptr) ? h_graph.extent(1) : 0;
    }

    /** @brief Get number of segments */
    uint32_t segment_num() {
        return (h_segment_start.data_handle() != nullptr) ? h_segment_start.extent(0) : 0;
    }

    /** @brief Get number of start points */
    uint32_t start_point_num() {
        return (h_start_point.data_handle() != nullptr) ? h_start_point.extent(0) : 0;
    }

    /** @brief Get number of centroids */
    uint32_t centroids_num() {
        return (h_centroids.data_handle() != nullptr) ? h_centroids.extent(0) : 0;
    }

    /** @brief Get host matrix view of data */
    raft::host_matrix_view<const Data_t, Index_t> data_view() {
        return h_data.view();
    }

    /** @brief Get host vector view of start points */
    raft::host_vector_view<const uint32_t> start_point_view() {
        return h_start_point.view();
    }

    /** @brief Get host matrix view of graph */
    raft::host_matrix_view<const uint32_t, Index_t> graph_view() {
        return h_graph.view();
    }

    /** @brief Get host vector view of segment start ids */
    raft::host_vector_view<const uint32_t> segment_start_view() {
        return h_segment_start.view();
    }

    /** @brief Get host vector view of segment lengths */
    raft::host_vector_view<const uint32_t> segment_length_view() {
        return h_segment_length.view();
    }

    /** @brief Get host view of mapping */
    raft::host_vector_view<const uint32_t> map_view() {
        return h_map.view();
    }

    /** @brief Get host matrix view of centroids */
    raft::host_matrix_view<const float> centroids_view() {
        return h_centroids.view();
    }

protected:

    /**
     * @brief Select start points
     *
     * @param build_time Reference to accumulate build time
     */
    void start_point_select(float &build_time);

    /**
     * @brief Build and merge subgraphs
     *
     * @param param Search parameter for build
     * @param relaxant_factor Relaxation factor for pruning
     * @param build_time Reference to accumulate build time
     * @param h_knn_graph_view Host matrix view of KNN graph (num × knn_degree)
     */
    void subgraph_build_and_merge(SearchParameter &param, float relaxant_factor, float &build_time,
                                  raft::host_matrix_view<uint32_t, Index_t> h_knn_graph_view);

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
    raft::host_matrix<Data_t, Index_t> h_data; /// Host matrix of data (num × dim)
    raft::host_vector<uint32_t> h_map; /// Host vector of mapping
    raft::host_matrix<float> h_centroids; /// Host matrix of centroids (centroid_num × dim)
    raft::host_vector<uint32_t> h_segment_start; /// Host vector of segment start ids
    raft::host_vector<uint32_t> h_segment_length; /// Host vector of segment lengths
    raft::host_vector<uint32_t> h_start_point; /// Host vector of start points
    raft::host_matrix<uint32_t, Index_t> h_graph; /// Host matrix of graph (num × graph_degree)
};