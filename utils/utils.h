#ifndef PC_CPP_K_MEANS_CLUSTERING_UTILS_H
#define PC_CPP_K_MEANS_CLUSTERING_UTILS_H

#include <cmath>
#include <algorithm>
#include <functional>

template<typename InputIt1>
double norm2(InputIt1 first, InputIt1 last) {
    double acc = 0;
    while (first!=last) {
        acc += *first * *first;
        ++first;
    }
    return sqrt(acc);
}

template<typename InputIt1, typename InputIt2>
double norm2(InputIt1 first1, InputIt1 last1, InputIt2 first2) {
    double acc = 0;
    while (first1 != last1) {
        double sub = *first1 - *first2;
        acc += sub * sub;
        ++first1; ++first2;
    }
    return sqrt(acc);
}

// sum between std::vectors
template<typename T = void>
struct vectPlus;

template<typename T>
struct vectPlus : public std::binary_function<T, T, T> {
    T operator()(const T &v1, const T &v2) const {
        T sumV(v1.size());
        std::transform(v1.begin(), v1.end(),
                       v2.begin(), sumV.begin(),
                       std::plus<>());
        return sumV;
    }
};

template<> struct vectPlus<void> {
    template<typename T>
    auto operator()(const T &v1, const T &v2) {
        return vectPlus<T>().operator()(v1, v2);
    }
};

bool compare(const double& value1, const double& value2, const int& precision);

#endif //PC_CPP_K_MEANS_CLUSTERING_UTILS_H
