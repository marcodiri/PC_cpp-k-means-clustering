cmake_minimum_required(VERSION 3.17)
project(PC_cpp_k_means_clustering)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

set(SOURCE_FILES main.cpp cluster/KMeans.cpp cluster/KMeans.h cluster/KMeans_Seq.cpp cluster/KMeans_Seq.h cluster/KMeans_Par.cpp cluster/KMeans_Par.h utils/utils.h utils/utils.cpp benchmark/Benchmark.cpp benchmark/Benchmark.h)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_package(OpenMP)
target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_LIBRARIES})
