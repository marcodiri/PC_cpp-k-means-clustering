#ifndef K_MEANS_CLUSTERING_KMEANS_SEQ_H
#define K_MEANS_CLUSTERING_KMEANS_SEQ_H


#include "KMeans.h"
#include <algorithm>
#include "../utils/utils.h"

class KMeans_Seq : public KMeans {
public:
    KMeans_Seq(int nClusters, int maxIter, uint seed);

    const KMeans *fit(const matrix &points) override;
};


#endif //K_MEANS_CLUSTERING_KMEANS_SEQ_H
