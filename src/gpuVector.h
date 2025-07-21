#ifndef GPUVECTOR_H
#define GPUVECTOR_H

#include <cuda_runtime.h>
#include <stdexcept>

#ifndef __CUDA_ARCH__
#include <memory>
#include <vector>
#endif

/**
 * @class gpuVector
 * @brief A RAII wrapper for GPU memory management of double vectors
 * 
 * This class handles GPU memory allocation/deallocation automatically
 * and provides safe data transfer between host and device.
 */
class gpuVector {
private:
    double* d_ptr;     // Device pointer to GPU memory
    size_t size_;      // Number of elements
    size_t capacity_;  // Allocated capacity (for future expansion)
    
    // Helper function to safely call CUDA functions
    void cudaSafeCall(cudaError_t err, const char* msg) const {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    }

public:
    // Constructor - allocates GPU memory
    explicit gpuVector(size_t n) : size_(n), capacity_(n) {
        if (n == 0) {
            d_ptr = nullptr;
            return;
        }
        
        cudaError_t err = cudaMalloc(&d_ptr, capacity_ * sizeof(double));
        cudaSafeCall(err, "Failed to allocate GPU memory in gpuVector constructor");
    }
    
    // Constructor from host data - allocates and copies
    gpuVector(const double* host_data, size_t n) : size_(n), capacity_(n) {
        if (n == 0) {
            d_ptr = nullptr;
            return;
        }
        
        cudaError_t err = cudaMalloc(&d_ptr, capacity_ * sizeof(double));
        cudaSafeCall(err, "Failed to allocate GPU memory in gpuVector constructor");
        
        err = cudaMemcpy(d_ptr, host_data, size_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaSafeCall(err, "Failed to copy data to GPU in gpuVector constructor");
    }
    
    // Destructor - frees GPU memory
    ~gpuVector() {
        if (d_ptr != nullptr) {
            cudaFree(d_ptr);
        }
    }
    
    // Copy constructor
    gpuVector(const gpuVector& other) : size_(other.size_), capacity_(other.capacity_) {
        if (capacity_ == 0) {
            d_ptr = nullptr;
            return;
        }
        
        cudaError_t err = cudaMalloc(&d_ptr, capacity_ * sizeof(double));
        cudaSafeCall(err, "Failed to allocate GPU memory in gpuVector copy constructor");
        
        err = cudaMemcpy(d_ptr, other.d_ptr, size_ * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaSafeCall(err, "Failed to copy GPU data in gpuVector copy constructor");
    }
    
    // Move constructor
    gpuVector(gpuVector&& other) noexcept 
        : d_ptr(other.d_ptr), size_(other.size_), capacity_(other.capacity_) {
        other.d_ptr = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    // Copy assignment operator
    gpuVector& operator=(const gpuVector& other) {
        if (this != &other) {
            // Free current memory
            if (d_ptr != nullptr) {
                cudaFree(d_ptr);
                d_ptr = nullptr;
            }
            
            size_ = other.size_;
            capacity_ = other.capacity_;
            
            if (capacity_ > 0) {
                cudaError_t err = cudaMalloc(&d_ptr, capacity_ * sizeof(double));
                cudaSafeCall(err, "Failed to allocate GPU memory in gpuVector copy assignment");
                
                err = cudaMemcpy(d_ptr, other.d_ptr, size_ * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaSafeCall(err, "Failed to copy GPU data in gpuVector copy assignment");
            }
        }
        return *this;
    }
    
    // Move assignment operator
    gpuVector& operator=(gpuVector&& other) noexcept {
        if (this != &other) {
            // Free current memory
            if (d_ptr != nullptr) {
                cudaFree(d_ptr);
            }
            
            // Steal other's resources
            d_ptr = other.d_ptr;
            size_ = other.size_;
            capacity_ = other.capacity_;
            
            // Reset other
            other.d_ptr = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Accessors
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    double* data() { return d_ptr; }
    const double* data() const { return d_ptr; }
    bool empty() const { return size_ == 0; }
    
    // Data transfer methods
    void copyToHost(double* host_ptr) const {
        if (size_ == 0) return;
        
        cudaError_t err = cudaMemcpy(host_ptr, d_ptr, size_ * sizeof(double), cudaMemcpyDeviceToHost);
        cudaSafeCall(err, "Failed to copy data from GPU to host");
    }
    
    void copyFromHost(const double* host_ptr) {
        if (size_ == 0) return;
        
        cudaError_t err = cudaMemcpy(d_ptr, host_ptr, size_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaSafeCall(err, "Failed to copy data from host to GPU");
    }
    
#ifndef __CUDA_ARCH__
    // Convert to host vector (only available in host code)
    std::vector<double> toHost() const {
        if (size_ == 0) return std::vector<double>();
        
        std::vector<double> result(size_);
        copyToHost(result.data());
        return result;
    }
#endif
};

#endif // GPUVECTOR_H 