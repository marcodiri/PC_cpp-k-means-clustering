#ifndef K_MEANS_CLUSTERING_KMEANS_PAR_H
#define K_MEANS_CLUSTERING_KMEANS_PAR_H


#include "KMeans.h"
#include <algorithm>
#include "utils.h"

class KMeans_OMP : public KMeans {
public:
    KMeans_OMP(int nClusters, int maxIter, uint seed);

    const KMeans *fit(const matrix &points) override;
};


#endif //K_MEANS_CLUSTERING_KMEANS_PAR_H
