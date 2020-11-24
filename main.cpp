#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iterator>
#include "cluster/KMeans.h"
#include "cluster/KMeans_Seq.h"
#include "cluster/KMeans_Par.h"

int main(int argc, char **argv) {
    if (argc < 2 or argc > 5) {
        std::cerr << "Invalid arguments number" << std::endl;
        std::cout << "Usage: PC_cpp_k_means_clustering \"path/to/dataset\" "
                     "[0 or 1: 0 sequential, 1 parallel] "
                     "[nClusters] "
                     "[maxIter]" << std::endl;
        return 1;
    }

    // read args
    int isParallel = true, nClusters = 10, maxIter = 300;
    int* args[] = {&isParallel, &nClusters, &maxIter};
    std::size_t pos;
    std::string arg;
    try {
        for (int i=2; i<argc; i++) {
            arg = argv[i];
            *(args[i-2]) = std::stoi(arg, &pos);
            if (*(args[i-2]) <= 0 && i-2 > 0) {
                std::cerr << "argument in position " << i-1 << " must be greater than 0" << std::endl;
                return 1;
            }
            if (pos < arg.size()) {
                std::cerr << "Trailing characters after number: " << arg << std::endl;
                return 1;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid number: " << arg << std::endl;
        return 1;
    } catch (std::out_of_range const &ex) {
        std::cerr << "Number out of range: " << arg << std::endl;
        return 1;
    }

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

    matrix dataset;
    std::string temp;
    while (std::getline(infile, temp)) {
        std::istringstream buffer(temp);
        std::vector<double> line{std::istream_iterator<double>(buffer),
                                 std::istream_iterator<double>()};
        dataset.push_back(line);
    }
    infile.close();

    KMeans* kmeans;
    if (isParallel)
        kmeans = new KMeans_Par(nClusters, maxIter);
    else
        kmeans = new KMeans_Seq(nClusters, maxIter);

    kmeans->fit(dataset);
    kmeans->toFile("../data/kmeans");

    std::cout << "Completed after "
    << kmeans->getNIter()
    << " iterations"
    << std::endl;

    delete kmeans;

    return 0;
}
