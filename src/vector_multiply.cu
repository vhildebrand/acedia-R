// vector_multiply.cu
// CUDA kernels and host functions for vector multiplication operations

#include <cuda_runtime.h>
#include <stdio.h>
#include "gpuVector.h"

// CUDA Kernel for element-wise multiplication
__global__ void multiplyKernel(double *c, const double *a, const double *b, int n) {
    // Global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one element of vector
    // Check bounds to handle cases where we have fewer threads than elements
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

// CUDA Kernel for scalar multiplication
__global__ void scalarMultiplyKernel(double *result, const double *vec, double scalar, int n) {
    // Global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one element of vector
    if (i < n) {
        result[i] = vec[i] * scalar;
    }
}

// CUDA Kernel for dot product (reduction)
__global__ void dotProductKernel(double *result, const double *a, const double *b, int n) {
    extern __shared__ double shared_data[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_data[tid] = 0.0;
    
    // Load data into shared memory and perform multiplication
    if (i < n) {
        shared_data[tid] = a[i] * b[i];
    }
    
    __syncthreads();
    
    // Perform tree-based reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the block result to global memory
    if (tid == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

// Host function for element-wise multiplication with C-style linkage
extern "C" void multiplyVectorsOnGpu(double *h_c, const double *h_a, const double *h_b, int n) {
    double *d_a, *d_b, *d_c;
    size_t size = n * sizeof(double);

    // Allocate memory on GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    multiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

cleanup:
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Host function for scalar multiplication with C-style linkage
extern "C" void scalarMultiplyVectorOnGpu(double *h_result, const double *h_vec, double scalar, int n) {
    double *d_vec, *d_result;
    size_t size = n * sizeof(double);

    // Allocate memory on GPU
    cudaMalloc(&d_vec, size);
    cudaMalloc(&d_result, size);

    // Copy data from host to device
    cudaMemcpy(d_vec, h_vec, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    scalarMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_vec, scalar, n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy result back to host
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

cleanup:
    // Free device memory
    cudaFree(d_vec);
    cudaFree(d_result);
}

// Host function for dot product with C-style linkage
extern "C" double dotProductOnGpu(const double *h_a, const double *h_b, int n) {
    double *d_a, *d_b, *d_block_results, *h_block_results;
    double final_result = 0.0;  // Declare early to avoid goto issues
    size_t size = n * sizeof(double);

    // Allocate memory on GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    
    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory for block results
    cudaMalloc(&d_block_results, blocksPerGrid * sizeof(double));
    h_block_results = (double*)malloc(blocksPerGrid * sizeof(double));

    // Launch kernel with shared memory
    size_t sharedMemSize = threadsPerBlock * sizeof(double);
    dotProductKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_block_results, d_a, d_b, n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Dot product kernel launch failed: %s\n", cudaGetErrorString(err));
        free(h_block_results);
        goto cleanup;
    }

    // Copy block results back to host
    cudaMemcpy(h_block_results, d_block_results, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    // Sum the block results on CPU
    for (int i = 0; i < blocksPerGrid; i++) {
        final_result += h_block_results[i];
    }

    free(h_block_results);

cleanup:
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_block_results);
    
    return final_result;
}

// gpuVector-based wrapper functions for element-wise multiplication
extern "C" void multiplyVectorsOnGpu_gpuVector(gpuVector& result, const gpuVector& a, const gpuVector& b) {
    // Validate input sizes
    if (a.size() != b.size() || result.size() != a.size()) {
        fprintf(stderr, "gpuVector size mismatch in multiplication\n");
        return;
    }
    
    if (a.empty()) {
        return; // Nothing to do for empty vectors
    }
    
    int n = static_cast<int>(a.size());
    
    // Launch kernel using existing device memory
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    multiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(result.data(), a.data(), b.data(), n);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

// gpuVector-based wrapper function for scalar multiplication
extern "C" void scalarMultiplyVectorOnGpu_gpuVector(gpuVector& result, const gpuVector& vec, double scalar) {
    // Validate input sizes
    if (result.size() != vec.size()) {
        fprintf(stderr, "gpuVector size mismatch in scalar multiplication\n");
        return;
    }
    
    if (vec.empty()) {
        return; // Nothing to do for empty vectors
    }
    
    int n = static_cast<int>(vec.size());
    
    // Launch kernel using existing device memory
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    scalarMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(result.data(), vec.data(), scalar, n);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

// gpuVector-based wrapper function for dot product
extern "C" double dotProductOnGpu_gpuVector(const gpuVector& a, const gpuVector& b) {
    // Validate input sizes
    if (a.size() != b.size()) {
        fprintf(stderr, "gpuVector size mismatch in dot product\n");
        return 0.0;
    }
    
    if (a.empty()) {
        return 0.0; // Dot product of empty vectors is 0
    }
    
    int n = static_cast<int>(a.size());
    double *d_block_results, *h_block_results;
    
    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory for block results
    cudaMalloc(&d_block_results, blocksPerGrid * sizeof(double));
    h_block_results = (double*)malloc(blocksPerGrid * sizeof(double));

    // Launch kernel with shared memory using existing device memory
    size_t sharedMemSize = threadsPerBlock * sizeof(double);
    dotProductKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_block_results, a.data(), b.data(), n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Dot product kernel launch failed: %s\n", cudaGetErrorString(err));
        free(h_block_results);
        cudaFree(d_block_results);
        return 0.0;
    }

    // Wait for kernel completion
    cudaDeviceSynchronize();

    // Copy block results back to host
    cudaMemcpy(h_block_results, d_block_results, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    // Sum the block results on CPU
    double final_result = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        final_result += h_block_results[i];
    }

    free(h_block_results);
    cudaFree(d_block_results);
    
    return final_result;
} 