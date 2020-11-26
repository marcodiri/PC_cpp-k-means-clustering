#ifndef PC_CPP_K_MEANS_CLUSTERING_BENCHMARK_H
#define PC_CPP_K_MEANS_CLUSTERING_BENCHMARK_H

#include "../cluster/KMeans.h"
#include "../cluster/KMeans_Seq.h"
#include "../cluster/KMeans_Par.h"
#include <iostream>
#include <memory>
#include <chrono>

using namespace std::chrono;

enum BTYPE {SEQUENTIAL, PARALLEL, VERSUS};

class Benchmark {
private:
    Benchmark()=default;
    explicit Benchmark(KMeans * kMeans);

    void test(const matrix &dataset) const;
    auto getWCT() const;

public:
    static void benchmark(const BTYPE &t, const matrix &dataset, const int &nCluster, const int &maxIter);
    static const std::unique_ptr<KMeans>* getTester();
    static void clearTesters();

private:
     mutable system_clock::time_point start;
     mutable system_clock::time_point stop;
     std::unique_ptr<KMeans> tester;
     static std::vector<std::unique_ptr<Benchmark>> benchmarks;
};

#endif //PC_CPP_K_MEANS_CLUSTERING_BENCHMARK_H
