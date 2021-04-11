#ifndef PC_CPP_K_MEANS_CLUSTERING_KMEANS_CUDA_CUH
#define PC_CPP_K_MEANS_CLUSTERING_KMEANS_CUDA_CUH


#include "KMeans.h"
#include <algorithm>
#include <chrono>
#include "utils.h"
#include "utils.cuh"

class KMeans_CUDA : public KMeans {
public:
    KMeans_CUDA(int nClusters, int maxIter, uint seed);

    const KMeans *fit(const matrix &points) override;
};

__global__ void
assignPointsToCentroids(size_t dimension_n, size_t points_n, size_t cluster_n);

#endif //PC_CPP_K_MEANS_CLUSTERING_KMEANS_CUDA_CUH
