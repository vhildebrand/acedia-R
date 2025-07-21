#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <string>

namespace cuda_utils {

/**
 * @brief Check if CUDA is available and working
 * @return true if CUDA is available and at least one GPU is present
 */
inline bool isGpuAvailable() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    // Try to query the first device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    
    return (err == cudaSuccess);
}

/**
 * @brief Get detailed GPU information
 * @return string containing GPU information or error message
 */
inline std::string getGpuInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess) {
        return "CUDA Error: " + std::string(cudaGetErrorString(err));
    }
    
    if (deviceCount == 0) {
        return "No CUDA-capable devices found";
    }
    
    std::string info = "Found " + std::to_string(deviceCount) + " CUDA device(s):\n";
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        
        if (err == cudaSuccess) {
            info += "  Device " + std::to_string(i) + ": " + std::string(prop.name);
            info += " (Compute " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
            info += " - " + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB\n";
        }
    }
    
    return info;
}

/**
 * @brief Enhanced CUDA error checking with context information
 */
inline void cudaSafeCall(cudaError_t err, const char* file, int line, const char* func) {
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA Error at " + std::string(file) + ":" + std::to_string(line) + 
                               " in " + std::string(func) + "(): " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
}

#define CUDA_SAFE_CALL(call) cuda_utils::cudaSafeCall((call), __FILE__, __LINE__, __FUNCTION__)

/**
 * @brief CPU fallback implementation for vector addition
 */
inline void addVectorsCpu(double* result, const double* a, const double* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

/**
 * @brief Check available GPU memory
 * @return Available memory in bytes, or 0 if GPU not available
 */
inline size_t getAvailableGpuMemory() {
    if (!isGpuAvailable()) {
        return 0;
    }
    
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (err != cudaSuccess) {
        return 0;
    }
    
    return free_mem;
}

} // namespace cuda_utils

#endif // CUDA_UTILS_H 