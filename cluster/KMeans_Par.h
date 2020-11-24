#ifndef K_MEANS_CLUSTERING_KMEANS_PAR_H
#define K_MEANS_CLUSTERING_KMEANS_PAR_H


#include "KMeans.h"

class KMeans_Par : public KMeans {
public:
    KMeans_Par(int nClusters, int maxIter);

    const KMeans *fit(const matrix &points) override;
};


#endif //K_MEANS_CLUSTERING_KMEANS_PAR_H
