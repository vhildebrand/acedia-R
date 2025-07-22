// tensor_ops.cu
// Operation functors and launch helper functions for tensor operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "kernel_utils.cuh"
#include "tensor_kernels.cu"

// Function objects for operations
struct AddOp {
    template<typename T>
    __device__ T operator()(const T& a, const T& b) const { return a + b; }
};

struct MulOp {
    template<typename T>
    __device__ T operator()(const T& a, const T& b) const { return a * b; }
};

struct SubOp {
    template<typename T>
    __device__ T operator()(const T& a, const T& b) const { return a - b; }
};

struct DivOp {
    template<typename T>
    __device__ T operator()(const T& a, const T& b) const { return a / b; }
};

// Unary operation functors
struct ExpOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return exp(a); }
};

struct LogOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return log(a); }
};

struct SqrtOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return sqrt(a); }
};

// Host interface functions using templates
template<typename T>
void launch_fill(T* data, T value, size_t n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    fill_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(data, value, n);
}

template<typename T, typename Op>
void launch_elementwise_binary(T* result, const T* a, const T* b, size_t n, Op op, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(result, a, b, n, op);
}

template<typename T, typename U, typename Op>
void launch_elementwise_scalar(T* result, const T* input, U scalar, size_t n, Op op, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(result, input, scalar, n, op);
}

template<typename T, typename Op>
void launch_elementwise_unary(T* result, const T* input, size_t n, Op op, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(result, input, n, op);
}

template<typename To, typename From>
void launch_type_conversion(To* output, const From* input, size_t n, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    type_conversion_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(output, input, n);
}

template<typename T, typename AccumType, typename Op>
AccumType launch_reduction(const T* input, size_t n, Op op, AccumType init_val, cudaStream_t stream = 0) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    AccumType* d_block_results;
    cudaMalloc(&d_block_results, blocksPerGrid * sizeof(AccumType));
    
    size_t sharedMemSize = threadsPerBlock * sizeof(AccumType);
    reduction_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(
        d_block_results, input, n, op, init_val
    );
    
    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return init_val;
    }
    
    // Wait for kernel completion
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA synchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        return init_val;
    }
    
    // Recursively reduce if needed
    AccumType* h_block_results = (AccumType*)malloc(blocksPerGrid * sizeof(AccumType));
    cudaMemcpy(h_block_results, d_block_results, blocksPerGrid * sizeof(AccumType), cudaMemcpyDeviceToHost);
    
    AccumType final_result = init_val;
    for (int i = 0; i < blocksPerGrid; i++) {
        final_result = op(final_result, h_block_results[i]);
    }
    
    free(h_block_results);
    cudaFree(d_block_results);
    
    return final_result;
}

template<typename T>
void launch_matmul(T* C, const T* A, const T* B, size_t M, size_t N, size_t K, cudaStream_t stream = 0) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Use tiled version for better performance
    matmul_tiled_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(C, A, B, M, N, K);
}

// Launch helper for broadcast operations
template<typename T, typename Op>
void launch_broadcast_binary(
    T* result, const T* a, const T* b,
    const int* a_strides, const int* b_strides, const int* result_strides,
    const int* shape, int ndims, size_t total_elements, Op op
) {
    // Copy stride and shape data to device
    int *d_a_strides, *d_b_strides, *d_result_strides, *d_shape;
    cudaMalloc(&d_a_strides, ndims * sizeof(int));
    cudaMalloc(&d_b_strides, ndims * sizeof(int));
    cudaMalloc(&d_result_strides, ndims * sizeof(int));
    cudaMalloc(&d_shape, ndims * sizeof(int));
    
    cudaMemcpy(d_a_strides, a_strides, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_strides, result_strides, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, ndims * sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    broadcast_binary_kernel<T, Op><<<grid_size, block_size>>>(
        result, a, b, d_a_strides, d_b_strides, d_result_strides, d_shape, ndims, total_elements
    );
    
    cudaFree(d_a_strides);
    cudaFree(d_b_strides);
    cudaFree(d_result_strides);
    cudaFree(d_shape);
}

template<typename T>
void launch_strided_copy(T* dest, const T* src, const std::vector<int>& strides, const std::vector<int>& shape, size_t total_elements) {
    // Copy stride and shape data to device
    int* d_strides;
    int* d_shape;
    int ndims = shape.size();
    
    cudaMalloc(&d_strides, ndims * sizeof(int));
    cudaMalloc(&d_shape, ndims * sizeof(int));
    
    cudaMemcpy(d_strides, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_copy_kernel<T><<<grid_size, block_size>>>(
        dest, src, d_strides, d_shape, ndims, total_elements
    );
    
    cudaFree(d_strides);
    cudaFree(d_shape);
} 