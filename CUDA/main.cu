#include <iostream>
#include <stdexcept>
#include <string>
#include <iterator>
#include "KMeans_Seq.h"
#include "KMeans_CUDA.cuh"
#include "Benchmark.h"

#define N_MANDATORY_ARGS 2

int main(int argc, char **argv) {
    // read args
    int bmarkType = 0, nClusters = 10, maxIter = 300;
    int* args[] = {&bmarkType, &nClusters, &maxIter};
    std::size_t pos;
    std::string arg;
    try {
        if (argc < 3 or argc > 6) {
            throw std::invalid_argument("Invalid arguments number");
        }
        for (int i=0; i<argc-N_MANDATORY_ARGS-1; i++) {
            arg = argv[i+N_MANDATORY_ARGS+1];
            int argi = std::stoi(arg, &pos);
            if (i == 0) {
                if (argi > 2) argi = 2;
                if (argi < 0) argi = 0;
            }
            if (argi <= 0 && i > 0) {
                throw std::invalid_argument("argument in position " +
                                            std::to_string(i+N_MANDATORY_ARGS+1) +
                                            " must be greater than 0");
            }
            if (pos < arg.size()) {
                throw std::invalid_argument("Trailing characters after number: "+arg);
            }
            *(args[i]) = argi;
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << ex.what() << std::endl;
        std::cout << "Usage: PC_cpp_k_means_clustering \"path/to/dataset\" "
                     "\"path/to/output/folder\" "
                     "[0, 1 or 2: 0 sequential, 1 parallel, 2 versus] "
                     "[nClusters > 0] "
                     "[maxIter > 0]" << std::endl;
        std::cout << "\"path/to/output/folder\" must exist. "
                     "Typing the empty string \"\" will not save the results."
                  << std::endl;
        return 1;
    } catch (std::out_of_range const &ex) {
        std::cerr << "Number out of range: " << arg << std::endl;
        return 1;
    }

#ifdef DEBUG
    uint seed = 42;
#else
    auto seed = bmarkType==2 ? 42 : std::random_device{}();
#endif

    std::cout << "Called script with parameters: ";
    for (const auto *val : args)
        std::cout << *val << " ";
    std::cout << std::endl;

    // read dataset into vector
    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "ERROR: couldn't read data file" << std::endl;
        return 1;
    }

    matrix dataset{2};
    {
        std::string temp;
        while (std::getline(infile, temp)) {
            std::istringstream buffer(temp);
            std::vector<el_type> line{std::istream_iterator<el_type>(buffer),
                                      std::istream_iterator<el_type>()};
            dataset[0].push_back(line[0]);
            dataset[1].push_back(line[1]);
        }
        infile.close();
    }


    /*************************** BENCHMARK ***************************/

    // trigger the lazy creation of the CUDA context, otherwise the first
    // cudaMalloc() will absorb the context creation cost which is not good
    // for benchmark purposes
    cudaFree(nullptr);

    std::vector<KMeans *> testers;
    switch (bmarkType) {
        case 0:
            testers.emplace_back(new KMeans_Seq(nClusters, maxIter, seed));
            break;
        case 1:
            testers.emplace_back(new KMeans_CUDA(nClusters, maxIter, seed));
            break;
        case 2:
        default:
            testers.emplace_back(new KMeans_Seq(nClusters, maxIter, seed));
            testers.emplace_back(new KMeans_CUDA(nClusters, maxIter, seed));
            break;
    }
    Benchmark::setTester(testers);
    Benchmark::benchmark(static_cast<BTYPE>(bmarkType), dataset);

    // save results to file if path provided
    std::string savepath(argv[2]);
    if (!savepath.empty()) {
        auto tester = Benchmark::getTester();
        if (tester != nullptr)
            tester->toFile(savepath + "/kmeans");
    }

    Benchmark::clearTesters();

    for (auto t : testers)
        delete t;

    return 0;
}
