#ifndef K_MEANS_CLUSTERING_KMEANS_H
#define K_MEANS_CLUSTERING_KMEANS_H


#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>

// using float instead of double because my GPU has compute capability 3.0
// so atomicAdd(double,double) is not available (yes, we could implement it
// manually but not being hardware supported it terribly slows down the kernel)
using el_type = float;
using matrix = std::vector<std::vector<el_type>>;

class KMeans {
/**
 * Abstract class representing KMeans algorithm to solve
 * the clustering problem.
 * Each derived class should implement the fit method
 * to define the state of the finished clustering.
 */
public:
    KMeans(int nClusters, int maxIter, uint seed);

    virtual ~KMeans();

    const matrix &getClusterCenters() const;

    const std::vector<unsigned int> &getLabels() const;

    int getNIter() const;

    int getNClusters() const;

    int getMaxIter() const;

    unsigned int getSeed() const;

    void setSeed(unsigned int seed);

    bool toFile(const std::string&) const;

    virtual const KMeans* fit(const matrix &points)=0;

protected:
    matrix clusterCenters;
    std::vector<unsigned int> labels;
    int nIter;

private:
    int nClusters;
    int maxIter;
    unsigned int seed;
};


#endif //K_MEANS_CLUSTERING_KMEANS_H
