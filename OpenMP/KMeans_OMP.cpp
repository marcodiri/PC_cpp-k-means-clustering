#include "KMeans_OMP.h"

KMeans_OMP::KMeans_OMP(int nClusters, int maxIter, uint seed) : KMeans(nClusters, maxIter, seed) {}

const KMeans *KMeans_OMP::fit(const matrix &points) {
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

    // declare reduction functions to apply to non-trivial classes
#pragma omp declare reduction(vec_uint_plus : std::vector<unsigned int> : \
omp_out = vectPlus<>().operator()(omp_in, omp_out)) \
initializer(omp_priv = omp_orig)
#pragma omp declare reduction(matrix_plus : matrix : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), vectPlus<>())) \
initializer(omp_priv = omp_orig)

    decltype(labels) labels(points_n);
    int nIter = 0;
    bool centroidsChanged = true;
    while (centroidsChanged && nIter < getMaxIter()) {
        ++nIter;
        centroidsChanged = false;
        std::vector<unsigned int> pointsPerCluster(cluster_n);
        for (auto &p : pointsPerCluster) p = 0;
        matrix centroidsNew{dimension_n};
        for (auto &el : centroidsNew)
            el.resize(cluster_n, 0); // init new centroids to 0 vectors

        // privatize structures and apply reduction later
        // NOTE: labels doesn't need reduction nor synchronization
        // because every thread writes on different indices
#pragma omp parallel for default(none) shared(points, centroids, labels, points_n, cluster_n, dimension_n) \
reduction(vec_uint_plus : pointsPerCluster) reduction(matrix_plus : centroidsNew)
        // 2. compute the distance between centroids and observation
        for (size_t i=0; i<points_n; i++) {
            double min = -1; // minimum distance between p and a centroid
            unsigned int centroid_i = 0; // index of the centroid closer to p
            for (size_t j=0; j < cluster_n; j++) {
                // euclidean distance (norm2) between centroid and point
                el_type d = 0;
                for (size_t d_i=0; d_i < dimension_n; ++d_i) {
                    el_type sub = points[d_i][i] - centroids[d_i][j];
                    d += sub * sub;
                }
                if (d < min || min == -1) {
                    min = d;
                    centroid_i = j;
                }
            }
            // 3. assign each observation to a centroid based on their distance
            labels[i] = centroid_i;
            pointsPerCluster[centroid_i]++;

            // update new centroids partial sum
            for (size_t d_i=0; d_i<dimension_n; ++d_i)
                centroidsNew[d_i][centroid_i] += points[d_i][i];
        }

        // 4. compute new centroids (mean of observations in a cluster)
        for (size_t d_i=0; d_i<dimension_n; d_i++) {
            for (size_t i=0; i<cluster_n; i++) {
                centroidsNew[d_i][i] /= pointsPerCluster[i];
                // check whether the new centroids are the same as the old ones
                if (!centroidsChanged && !compare(centroidsNew[d_i][i], centroids[d_i][i], 3))
                    centroidsChanged = true;
            }
        }
        centroids = centroidsNew;
    }

    clusterCenters = centroids;
    this->labels = labels;
    this->nIter = nIter;

    return this;
}
