#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdexcept>
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

/**
 * @brief Lightweight tensor descriptor for strided operations
 * 
 * This POD struct can be efficiently copied to GPU constant memory
 * and used by both host and device code for tensor operations.
 */
struct TensorDescriptor {
    void* data;           // Base data pointer
    int ndims;           // Number of dimensions (max 8)
    int shape[8];        // Dimension sizes
    int strides[8];      // Strides in elements (not bytes)
    size_t offset;       // Base offset in elements
    size_t total_size;   // Total number of elements
    
    // Default constructor
    __host__ __device__ TensorDescriptor() : data(nullptr), ndims(0), offset(0), total_size(0) {
        for (int i = 0; i < 8; ++i) {
            shape[i] = 0;
            strides[i] = 0;
        }
    }
    
    // Check if tensor is contiguous (unit strides in expected order)
    __host__ __device__ bool is_contiguous() const {
        if (ndims == 0) return true;
        
        int expected_stride = 1;
        for (int i = ndims - 1; i >= 0; --i) {
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape[i];
        }
        return true;
    }
    
    // Compute linear offset for given coordinates
    __host__ __device__ size_t compute_offset(const int* coords) const {
        size_t linear_offset = offset;
        for (int i = 0; i < ndims; ++i) {
            linear_offset += coords[i] * strides[i];
        }
        return linear_offset;
    }
    
    // Convert linear index to coordinates (used in kernels)
    __host__ __device__ void linear_to_coords(size_t linear_idx, int* coords) const {
        size_t remaining = linear_idx;
        for (int i = ndims - 1; i >= 0; --i) {
            coords[i] = remaining % shape[i];
            remaining /= shape[i];
        }
    }
};

} // namespace cuda_utils

#endif // CUDA_UTILS_H 