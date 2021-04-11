#include "KMeans.h"

KMeans::KMeans(int nClusters, int maxIter, uint seed) :
        nClusters(nClusters), maxIter(maxIter), seed(seed), nIter(0) {}

KMeans::~KMeans() = default;

const matrix &KMeans::getClusterCenters() const {
    return clusterCenters;
}

const std::vector<unsigned int> &KMeans::getLabels() const {
    return labels;
}

int KMeans::getNIter() const {
    return nIter;
}

int KMeans::getNClusters() const {
    return nClusters;
}

int KMeans::getMaxIter() const {
    return maxIter;
}

unsigned int KMeans::getSeed() const {
    return seed;
}

void KMeans::setSeed(unsigned int seed) {
    KMeans::seed = seed;
}

bool KMeans::toFile(const std::string& filename) const {
    std::string filepath = filename+"_labels.txt";
    std::ofstream outFile(filepath);
    if (!outFile) {
        std::cerr << "couldn't create file "
        << filepath << std::endl;
        return false;
    }
    for (const auto &l : getLabels()) outFile << l << std::endl;
    outFile.close();

    filepath = filename+"_centroids.txt";
    outFile.open(filepath);
    if (!outFile) {
        std::cerr << "couldn't create file "
                  << filepath << std::endl;
        return false;
    }
    outFile << std::scientific
    << std::setprecision(std::numeric_limits<double>::digits10 + 2);
    const auto centroids = getClusterCenters();
    for (int c_i=0; c_i < getNClusters(); ++c_i) {
        for (const auto & centroid : centroids)
            outFile << centroid[c_i] << " ";
        outFile << std::endl;
    }
    outFile.close();

    return true;
}
