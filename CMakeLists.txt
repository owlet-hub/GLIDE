cmake_minimum_required(VERSION 3.26.4)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDACXX /usr/local/cuda/bin/nvcc)

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(GLIDE)

project(GLIDE LANGUAGES CXX CUDA)

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)
include(cmake/thirdparty/get_cuvs.cmake)

find_package(OpenMP REQUIRED)

add_executable(small_preprocess src/small_preprocess.cu)
add_executable(small_knn src/small_knn.cu)
add_executable(small_build src/small_build.cu)
add_executable(small_search src/small_search.cu)
#add_executable(spgg src/test.cu)
add_executable(large_preprocess src/large_preprocess.cu)
add_executable(large_knn src/large_knn.cu)
add_executable(large_build src/large_build.cu)
add_executable(large_search src/large_search.cu)

if(OpenMP_CXX_FOUND)
    target_compile_options(spgg PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(spgg PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(knn PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(knn PUBLIC OpenMP::OpenMP_CXX)
endif()

#target_link_libraries(spgg PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
#target_compile_options(spgg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(small_preprocess PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_compile_options(small_preprocess PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(small_knn PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_compile_options(small_knn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(small_build PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_compile_options(small_build PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(small_search PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_compile_options(small_search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(large_preprocess PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_compile_options(large_preprocess PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(large_knn PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_compile_options(large_knn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(large_build PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_compile_options(large_build PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(large_search PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_compile_options(large_search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)