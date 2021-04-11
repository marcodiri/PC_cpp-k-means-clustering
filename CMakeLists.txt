cmake_minimum_required(VERSION 3.19)
project(PC_cpp_k_means_clustering)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

include_directories(utils cluster benchmark)

add_subdirectory(OpenMP)
add_subdirectory(CUDA)