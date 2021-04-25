project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(SOURCE_FILES main.cu KMeans_CUDA.cu KMeans_CUDA.cuh
        ../cluster/KMeans.cpp ../cluster/KMeans.h
        ../cluster/KMeans_Seq.cpp ../cluster/KMeans_Seq.h
        ../utils/utils.h utils.cuh
        ../benchmark/Benchmark.cpp ../benchmark/Benchmark.h)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)