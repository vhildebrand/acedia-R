#include <Rcpp.h>
#include "cuda_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
bool gpu_available() {
    return cuda_utils::isGpuAvailable();
}

// [[Rcpp::export]]
std::string gpu_info() {
    return cuda_utils::getGpuInfo();
}

// [[Rcpp::export]]
double gpu_memory_available() {
    size_t mem = cuda_utils::getAvailableGpuMemory();
    return static_cast<double>(mem);
} 