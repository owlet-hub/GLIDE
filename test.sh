data_file='./datasets/sift_learn.fbin'
query_file='./datasets/sift_query.fbin'
truth_file='./datasets/sift_query_learn_gt100'
result_file='result.csv'
dataset='sift100k'

metric=Euclidean

#preprocess
centroid_num=10
boundary_factor=1.05
sample_factor=0.01
#build
knn_degree=32
graph_degree=64
relaxant_factor=1.05
build_beam=64
refine_beam=128
#search
topk=10
search_beam=50

preprocess_file='./temp/'${dataset}'_centroid_'${centroid_num}'_boundary_'${boundary_factor}'_sample_'${sample_factor}
knn_file='./temp/'${dataset}'_knn_'${knn_degree}
graph_base_file='./temp/'${dataset}'_degree_'${graph_degree}'_relaxant_'${relaxant_factor}'_beam_'${build_beam}'_refinebeam'${refine_beam}

./build/small_preprocess ${data_file} ${preprocess_file} ${result_file} ${centroid_num} ${boundary_factor} ${sample_factor} ${metric}

./build/small_knn ${preprocess_file} ${knn_file} ${result_file} ${knn_degree}

./build/small_build ${data_file} ${preprocess_file} ${knn_file} ${graph_base_file} ${result_file} ${metric} ${graph_degree} ${knn_degree} ${relaxant_factor} ${build_beam} ${refine_beam} 

./build/small_search ${data_file} ${query_file} ${truth_file} ${preprocess_file} ${graph_base_file} ${result_file} ${metric} ${topk} ${search_beam}