#include "KMeans_Par.h"

KMeans_Par::KMeans_Par(int nClusters, int maxIter) : KMeans(nClusters, maxIter) {}

const KMeans *KMeans_Par::fit(const matrix &points) {
    // ... change state
    return this;
}
