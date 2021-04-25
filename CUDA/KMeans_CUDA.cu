#include "KMeans_CUDA.cuh"

KMeans_CUDA::KMeans_CUDA(int nClusters, int maxIter, uint seed) : KMeans(nClusters, maxIter, seed) {}

// flattened device arrays
__device__ el_type *d_points, *d_centroids, *d_centroidsNew;
__device__ unsigned int *d_pointsPerCluster, *d_labels;

const KMeans *KMeans_CUDA::fit(const matrix &points) {
    const auto &points_n = points[0].size();
    const auto &dimension_n = points.size();
    const auto &cluster_n = getNClusters();

    // 1. select random observation as starting centroids
    matrix centroids{dimension_n};

    std::mt19937 mt(getSeed());
    std::uniform_real_distribution<double> dist(0, points_n);

    for (int i=0; i<cluster_n; ++i) {
        auto rnd_i = static_cast<uint>(dist(mt));
        for (int ii=0; ii<dimension_n; ++ii) {
            centroids[ii].push_back(points[ii][rnd_i]);
        }
    }

    // points
    el_type *dev_points;
    {
        // allocate device memory
        size_t sz = points_n * sizeof(el_type);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(dev_points), dimension_n * sz));
        // copy arrays from host to device (flattened)
        CUDA_CHECK_RETURN(cudaMemcpy(dev_points, points[0].data(), sz, cudaMemcpyHostToDevice));
        // copy second dimension array with offset
        CUDA_CHECK_RETURN(cudaMemcpy(dev_points + points_n, points[1].data(), sz, cudaMemcpyHostToDevice));
        // copy host struct to device
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_points, &dev_points, sizeof(el_type *)));
    }

    // centroids
    el_type *dev_centroids;
    {
        size_t sz = cluster_n * sizeof(el_type);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(dev_centroids), dimension_n * sz));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_centroids, centroids[0].data(), sz, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_centroids + cluster_n, centroids[1].data(), sz, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_centroids, &dev_centroids, sizeof(el_type *)));
    }

    // centroidsNew
    el_type *dev_centroidsNew;
    {
        size_t sz = cluster_n * sizeof(el_type);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(dev_centroidsNew), dimension_n * sz));
        // init to 0
        CUDA_CHECK_RETURN(cudaMemset(dev_centroidsNew, 0, dimension_n * sz));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_centroidsNew, &dev_centroidsNew, sizeof(el_type *)));
    }

    // pointsPerCluster
    unsigned int *dev_pointsPerCluster;
    {
        size_t sz = cluster_n * sizeof(unsigned int);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_pointsPerCluster, sz));
        CUDA_CHECK_RETURN(cudaMemset(dev_pointsPerCluster, 0, sz));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_pointsPerCluster, &dev_pointsPerCluster, sizeof(unsigned int *)));
    }

    // labels
    unsigned int *dev_labels;
    {
        size_t sz = points_n * sizeof(unsigned int);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_labels, sz));
        CUDA_CHECK_RETURN(cudaMemset(dev_labels, 0, sz));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_labels, &dev_labels, sizeof(unsigned int *)));
    }

    {
        unsigned int threadsPerBlock = 128;
        unsigned int blocks = std::ceil((float) points_n / threadsPerBlock);

        size_t cN_sz = dimension_n * cluster_n;
        size_t cN_sz_bytes = cN_sz * sizeof(el_type);
        size_t pPC_sz_bytes = cluster_n * sizeof(unsigned int);
        int maxIter = getMaxIter(), nIter = 0;
        while (nIter < maxIter) {
            ++nIter;

            assignPointsToCentroids<<<blocks, threadsPerBlock, dimension_n*cluster_n*sizeof(el_type)>>>(dimension_n, points_n, cluster_n);
            cudaDeviceSynchronize();

            calculateNewCentroids<<<1, cluster_n>>>(dimension_n, points_n, cluster_n);
            cudaDeviceSynchronize();

            CUDA_CHECK_RETURN(cudaMemset(dev_centroidsNew, 0, cN_sz_bytes));
            CUDA_CHECK_RETURN(cudaMemset(dev_pointsPerCluster, 0, pPC_sz_bytes));
        }

        // copy data back to host
        unsigned int h_labels[points_n];
        CUDA_CHECK_RETURN(cudaMemcpy(&h_labels, dev_labels, points_n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        el_type h_centroids[dimension_n*cluster_n];
        CUDA_CHECK_RETURN(cudaMemcpy(&h_centroids, dev_centroids, dimension_n*cluster_n * sizeof(el_type), cudaMemcpyDeviceToHost));
        for (size_t d_i = 0; d_i < dimension_n; d_i++) {
            for (size_t i = 0; i < cluster_n; i++) {
                centroids[d_i][i] = h_centroids[d_i * cluster_n + i];
            }
        }

        clusterCenters = centroids;
        this->labels.insert(this->labels.begin(), h_labels, h_labels+points_n);
        this->nIter = nIter;
    }

    // free memory
    CUDA_CHECK_RETURN(cudaFree(dev_points));
    CUDA_CHECK_RETURN(cudaFree(dev_centroids));
    CUDA_CHECK_RETURN(cudaFree(dev_centroidsNew));
    CUDA_CHECK_RETURN(cudaFree(dev_pointsPerCluster));
    CUDA_CHECK_RETURN(cudaFree(dev_labels));

    return this;
}

__global__ void
assignPointsToCentroids(size_t dimension_n, size_t points_n, size_t cluster_n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load centroids in shared memory
    extern __shared__ el_type s_centroids[];
    for (int i = threadIdx.x; i < cluster_n; i += blockDim.x) {
        for (int j = 0; j < dimension_n; j++) {
            s_centroids[cluster_n * j + i] = d_centroids[cluster_n * j + i];
        }
    }
    __syncthreads();

    if (idx < points_n) {
        // 2. compute the distance between centroids and observation
        unsigned int label = 0; // index of the centroid closer to p
        {
            double min = -1; // minimum distance between p and a centroid
            for (size_t j = 0; j < cluster_n; j++) {
                // euclidean distance (norm2) between centroid and point
                // sqrt is unnecessary since it's a monotone function
                el_type d = 0;
                for (size_t d_i = 0; d_i < dimension_n; ++d_i) {
                    // d_i * points_n is the coordinate offset since array is flat
                    el_type sub = d_points[d_i * points_n + idx] - s_centroids[d_i * cluster_n + j];
                    d += sub * sub;
                }
                if (d < min || min == -1) {
                    min = d;
                    label = j;
                }
            }
        }
        // update our point label (no sync because every thread writes its index)
        d_labels[idx] = label;

        // 3. assign each observation to a centroid based on their distance
        // atomically increment the global counter array
        atomicAdd(&(d_pointsPerCluster[label]), 1);

        // 4. compute new centroids (mean of observations in a cluster)
        // atomically sum thread point at centroid index
        // the actual mean will be done in another kernel since we need to sync
        // every thread of the grid to get final pointsPerCluster values
        for (size_t d_i = 0; d_i < dimension_n; d_i++) {
            atomicAdd(&(d_centroidsNew[d_i * cluster_n + label]),
                            d_points[d_i * points_n + idx]);
        }
    }
}

__global__ void
calculateNewCentroids(size_t dimension_n, size_t points_n, size_t cluster_n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // calculate new centroids
    if(idx < cluster_n){
        for (size_t d_i=0; d_i<dimension_n; ++d_i) {
            d_centroids[d_i * cluster_n + idx] = d_centroidsNew[d_i * cluster_n + idx] / d_pointsPerCluster[idx];
        }
    }
}
