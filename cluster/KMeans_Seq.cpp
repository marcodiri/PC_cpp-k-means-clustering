#include "KMeans_Seq.h"

KMeans_Seq::KMeans_Seq(int nClusters, int maxIter, uint seed) : KMeans(nClusters, maxIter,seed) {}

const KMeans *KMeans_Seq::fit(const matrix &points) {
    // 1. select random observation as starting centroids
    matrix centroids;

    std::mt19937 mt(getSeed());
    std::uniform_real_distribution<double> dist(0, points.size());

    for (int i=0; i<getNClusters(); ++i) {
        centroids.push_back(points[static_cast<uint>(dist(mt))]);
    }

    decltype(labels) labels(points.size());
    int nIter = 0;
    bool centroidsChanged = true;
    while (centroidsChanged && nIter < getMaxIter()) {
        ++nIter;
        centroidsChanged = false;
        std::vector<unsigned int> pointsPerCluster(getNClusters());
        for (auto &p : pointsPerCluster) p = 0;
        matrix centroidsNew(centroids.size());
        for (auto &el : centroidsNew)
            el.resize(points[0].size(), 0); // init new centroids to 0 vectors
        // 2. compute the distance between centroids and observation
        for (size_t i=0; i<points.size(); i++) {
            const auto &p = points[i];
            double min = -1; // minimum distance between p and a centroid
            unsigned int centroid_i = 0; // index of the centroid closer to p
            for (size_t j=0; j < centroids.size(); j++) {
                const auto &c = centroids[j];
                // euclidean distance between c and p
                auto d = norm2(c.begin(), c.end(), p.begin());
                if (d < min || min == -1) {
                    min = d;
                    centroid_i = j;
                }
            }
            // 3. assign each observation to a centroid based on their distance
            labels[i] = centroid_i;
            pointsPerCluster[centroid_i]++;

            // update new centroids partial sum
            centroidsNew[centroid_i] = vectPlus<>().operator()(centroidsNew[centroid_i], points[i]);
        }

        // 4. compute new centroids (mean of observations in a cluster)
        for (auto i = centroidsNew.size(); i--;) {
            for (auto j = centroidsNew[i].size(); j--;) {
                centroidsNew[i][j] /= pointsPerCluster[i];
                // check whether the new centroids are the same as the old ones
                if (!compare(centroidsNew[i][j], centroids[i][j], 5)) centroidsChanged = true;
            }
        }
        centroids = centroidsNew;
    }

    clusterCenters = centroids;
    this->labels = labels;
    this->nIter = nIter;

    return this;
}
