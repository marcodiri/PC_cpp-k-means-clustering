#include "Benchmark.h"

std::vector<std::unique_ptr<Benchmark>> Benchmark::benchmarks;

Benchmark::Benchmark(KMeans * tester) : tester(tester) {}

void Benchmark::test(const matrix &dataset) const {
    start = high_resolution_clock::now();
    tester->fit(dataset);
    stop = high_resolution_clock::now();
}

auto Benchmark::getWCT() const {
    return duration_cast<std::chrono::microseconds>(stop - start).count();
}

void Benchmark::benchmark(const BTYPE &t, const matrix &dataset) {
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
                  static_cast<double>(benchmarks.front()->getWCT()) / benchmarks.back()->getWCT()
                  << std::endl;
    }
}

void Benchmark::setTester(const std::vector<KMeans *> &testers) {
    for (auto t : testers)
        benchmarks.emplace_back(new Benchmark(t));
}

const KMeans* Benchmark::getTester() {
    if (benchmarks.empty())
        return nullptr;
    else
        return benchmarks.back()->tester;
}

void Benchmark::clearTesters() {
    benchmarks.clear();
}
