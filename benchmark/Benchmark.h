#ifndef PC_CPP_K_MEANS_CLUSTERING_BENCHMARK_H
#define PC_CPP_K_MEANS_CLUSTERING_BENCHMARK_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>
#include <cstdarg>
#include <KMeans.h>

using namespace std::chrono;

enum BTYPE {SEQUENTIAL, PARALLEL, VERSUS};

class Benchmark {
private:
    explicit Benchmark(KMeans * tester);

    void test(const matrix &dataset) const;
    auto getWCT() const;

public:
    static void setTester(const std::vector<KMeans *> &testers);
    static void benchmark(const BTYPE &KMeans, const matrix &dataset);
    static const KMeans* getTester();
    static void clearTesters();

private:
     mutable system_clock::time_point start;
     mutable system_clock::time_point stop;
     KMeans *tester;
     static std::vector<std::unique_ptr<Benchmark>> benchmarks;
};

#endif //PC_CPP_K_MEANS_CLUSTERING_BENCHMARK_H
