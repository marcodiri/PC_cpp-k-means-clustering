#include "KMeans_Seq.h"

KMeans_Seq::KMeans_Seq(int nClusters, int maxIter) : KMeans(nClusters, maxIter) {}

const KMeans *KMeans_Seq::fit(const matrix &points) {
    // ... change state
    return this;
}
