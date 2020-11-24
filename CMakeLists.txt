cmake_minimum_required(VERSION 3.17)
project(PC_cpp_k_means_clustering)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp cluster/KMeans.cpp cluster/KMeans.h cluster/KMeans_Seq.cpp cluster/KMeans_Seq.h cluster/KMeans_Par.cpp cluster/KMeans_Par.h)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

