#ifndef PC_CPP_K_MEANS_CLUSTERING_UTILS_CUH
#define PC_CPP_K_MEANS_CLUSTERING_UTILS_CUH

static void CheckCudaErrorAux(const char *, unsigned, const char *,
                              cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
                              const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

#endif //PC_CPP_K_MEANS_CLUSTERING_UTILS_CUH
