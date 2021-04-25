#ifndef PC_CPP_K_MEANS_CLUSTERING_UTILS_H
#define PC_CPP_K_MEANS_CLUSTERING_UTILS_H

#include <cmath>
#include <algorithm>
#include <functional>

// sum between std::vectors
template<typename T = void>
struct vectPlus;

template<typename T>
struct vectPlus : public std::binary_function<T, T, T> {
    T &operator()(const T &v1, T &v2) const {
        std::transform(v1.begin(), v1.end(),
                       v2.begin(), v2.begin(),
                       std::plus<>());
        return v2;
    }
};

template<> struct vectPlus<void> {
    template<typename T>
    auto &operator()(const T &v1, T &v2) {
        return vectPlus<T>().operator()(v1, v2);
    }
};

#endif //PC_CPP_K_MEANS_CLUSTERING_UTILS_H
