#include "Benchmark.h"

std::vector<std::unique_ptr<Benchmark>> Benchmark::benchmarks;

Benchmark::Benchmark(KMeans *kMeans) : tester(kMeans) {}

void Benchmark::test(const matrix &dataset) const {
    start = high_resolution_clock::now();
    tester->fit(dataset);
    stop = high_resolution_clock::now();
}

auto Benchmark::getWCT() const {
    return duration_cast<std::chrono::microseconds>(stop - start).count();
}

void Benchmark::benchmark(const BTYPE &t, const matrix &dataset, const int &nCluster, const int &maxIter) {
#ifdef DEBUG
    uint seed = 42;
#else
    auto seed = t == BTYPE::VERSUS ? 42 : std::random_device{}();
#endif
    switch (t) {
        case BTYPE::SEQUENTIAL: {
            benchmarks.emplace_back(new Benchmark(new KMeans_Seq(nCluster, maxIter, seed)));
            break;
        }
        case BTYPE::PARALLEL: {
            benchmarks.emplace_back(new Benchmark(new KMeans_Par(nCluster, maxIter, seed)));
            break;
        }
        default: {
            benchmarks.emplace_back(new Benchmark(new KMeans_Seq(nCluster, maxIter, seed)));
            benchmarks.emplace_back(new Benchmark(new KMeans_Par(nCluster, maxIter, seed)));
        }
    }
    for (int i=0; i<benchmarks.size(); i++) {
        const auto &el = benchmarks[i];
        el->test(dataset);
        std::cout << "Completed "
        << (t == BTYPE::PARALLEL || i ? "parallel" : "sequential")
        << " kmeans in "
        << el->tester->getNIter()
        << " iterations after "
        << el->getWCT() << " us"
        << std::endl;
    }

    if (t == BTYPE::VERSUS) {
        std::cout << std::fixed << std::setprecision(2)
        << "SPEEDUP: " <<
        static_cast<double>(benchmarks[BTYPE::SEQUENTIAL]->getWCT()) / benchmarks[BTYPE::PARALLEL]->getWCT()
        << std::endl;
    }
}

const std::unique_ptr<KMeans>* Benchmark::getTester() {
    if (benchmarks.empty())
        return nullptr;
    else
        return &(benchmarks.front()->tester);
}

void Benchmark::clearTesters() {
    benchmarks.clear();
}
