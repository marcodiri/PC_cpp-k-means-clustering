project(OpenMP)

set(SOURCE_FILES main.cpp KMeans_OMP.cpp KMeans_OMP.h
        ../cluster/KMeans.cpp ../cluster/KMeans.h
        ../cluster/KMeans_Seq.cpp ../cluster/KMeans_Seq.h
        ../utils/utils.h ../utils/utils.cpp
        ../benchmark/Benchmark.cpp ../benchmark/Benchmark.h)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_package(OpenMP)
target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_LIBRARIES})
