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

// Function objects for operations - with both __host__ and __device__ support
struct AddOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const { return a + b; }
};

struct MulOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const { return a * b; }
};

struct SubOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const { return a - b; }
};

struct DivOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const { return a / b; }
};

// Unary operation functors - device only since they use CUDA math functions
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

// Reduction operation functors - with both __host__ and __device__ support
struct MaxOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const { 
#ifdef __CUDA_ARCH__
        return fmax(a, b); // Device version
#else
        return (a > b) ? a : b; // Host version
#endif
    }
};

struct MinOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const { 
#ifdef __CUDA_ARCH__
        return fmin(a, b); // Device version
#else
        return (a < b) ? a : b; // Host version
#endif
    }
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
    // Handle edge cases
    if (n == 0) return init_val;
    if (n == 1) {
        // For single element, copy from device to host safely
        T single_element;
        cudaError_t cudaStatus = cudaMemcpy(&single_element, input, sizeof(T), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            printf("CUDA memcpy failed for single element: %s\n", cudaGetErrorString(cudaStatus));
            return init_val;
        }
        return convert_type<AccumType>(single_element);
    }
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Limit the number of blocks to prevent excessive memory usage
    blocksPerGrid = std::min(blocksPerGrid, 65535);
    
    AccumType* d_block_results = nullptr;
    cudaError_t cudaStatus = cudaMalloc(&d_block_results, blocksPerGrid * sizeof(AccumType));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return init_val;
    }
    
    size_t sharedMemSize = threadsPerBlock * sizeof(AccumType);
    reduction_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(
        d_block_results, input, n, op, init_val
    );
    
    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_block_results);
        return init_val;
    }
    
    // Wait for kernel completion
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA synchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_block_results);
        return init_val;
    }
    
    // Copy results back and reduce on host
    AccumType* h_block_results = (AccumType*)malloc(blocksPerGrid * sizeof(AccumType));
    if (!h_block_results) {
        printf("Host malloc failed\n");
        cudaFree(d_block_results);
        return init_val;
    }
    
    cudaStatus = cudaMemcpy(h_block_results, d_block_results, blocksPerGrid * sizeof(AccumType), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("CUDA memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        free(h_block_results);
        cudaFree(d_block_results);
        return init_val;
    }
    
    // Host-side final reduction - now this will work with __host__ __device__ functors!
    AccumType final_result = h_block_results[0];
    for (int i = 1; i < blocksPerGrid; i++) {
        final_result = op(final_result, h_block_results[i]);
    }
    
    // Cleanup
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

// Launch concat operation
template<typename T>
void launch_concat(T* result, const T** inputs, const int* input_sizes, int num_tensors,
                   const std::vector<int>& result_strides, const std::vector<std::vector<int>>& input_strides_list,
                   const std::vector<int>& shape, int concat_axis, size_t total_elements) {
    
    // Prepare device memory for strides
    int* d_result_strides;
    int* d_input_strides_list;
    int* d_input_sizes;
    int* d_shape;
    T** d_inputs;
    
    int ndims = shape.size();
    
    cudaMalloc(&d_result_strides, ndims * sizeof(int));
    cudaMalloc(&d_input_strides_list, num_tensors * ndims * sizeof(int));
    cudaMalloc(&d_input_sizes, num_tensors * sizeof(int));
    cudaMalloc(&d_shape, ndims * sizeof(int));
    cudaMalloc(&d_inputs, num_tensors * sizeof(T*));
    
    cudaMemcpy(d_result_strides, result_strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_sizes, input_sizes, num_tensors * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, inputs, num_tensors * sizeof(T*), cudaMemcpyHostToDevice);
    
    // Flatten input strides
    std::vector<int> flattened_strides;
    for (const auto& strides : input_strides_list) {
        flattened_strides.insert(flattened_strides.end(), strides.begin(), strides.end());
    }
    cudaMemcpy(d_input_strides_list, flattened_strides.data(), num_tensors * ndims * sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    concat_kernel<T><<<grid_size, block_size>>>(
        result, d_inputs, d_input_sizes, num_tensors,
        d_result_strides, d_input_strides_list, d_shape, ndims, concat_axis, total_elements
    );
    
    cudaFree(d_result_strides);
    cudaFree(d_input_strides_list);
    cudaFree(d_input_sizes);
    cudaFree(d_shape);
    cudaFree(d_inputs);
}

// Launch stack operation
template<typename T>
void launch_stack(T* result, const T** inputs, int num_tensors,
                  const std::vector<int>& input_strides, const std::vector<int>& result_shape,
                  int stack_axis, size_t total_elements) {
    
    int* d_input_strides;
    int* d_result_shape;
    T** d_inputs;
    
    int ndims = result_shape.size();
    
    cudaMalloc(&d_input_strides, (ndims-1) * sizeof(int));
    cudaMalloc(&d_result_shape, ndims * sizeof(int));
    cudaMalloc(&d_inputs, num_tensors * sizeof(T*));
    
    cudaMemcpy(d_input_strides, input_strides.data(), (ndims-1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_shape, result_shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, inputs, num_tensors * sizeof(T*), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    stack_kernel<T><<<grid_size, block_size>>>(
        result, d_inputs, num_tensors, d_input_strides, d_result_shape, ndims, stack_axis, total_elements
    );
    
    cudaFree(d_input_strides);
    cudaFree(d_result_shape);
    cudaFree(d_inputs);
}

// Launch repeat operation
template<typename T>
void launch_repeat(T* result, const T* input,
                   const std::vector<int>& input_strides, const std::vector<int>& repeat_counts,
                   const std::vector<int>& input_shape, const std::vector<int>& result_shape,
                   size_t total_elements) {
    
    int* d_input_strides;
    int* d_repeat_counts;
    int* d_input_shape;
    int* d_result_shape;
    
    int ndims = input_shape.size();
    
    cudaMalloc(&d_input_strides, ndims * sizeof(int));
    cudaMalloc(&d_repeat_counts, ndims * sizeof(int));
    cudaMalloc(&d_input_shape, ndims * sizeof(int));
    cudaMalloc(&d_result_shape, ndims * sizeof(int));
    
    cudaMemcpy(d_input_strides, input_strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_repeat_counts, repeat_counts.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_shape, input_shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_shape, result_shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    repeat_kernel<T><<<grid_size, block_size>>>(
        result, input, d_input_strides, d_repeat_counts, d_input_shape, d_result_shape, ndims, total_elements
    );
    
    cudaFree(d_input_strides);
    cudaFree(d_repeat_counts);
    cudaFree(d_input_shape);
    cudaFree(d_result_shape);
}

// Launch pad operation
template<typename T>
void launch_pad(T* result, const T* input,
                const std::vector<int>& input_strides, const std::vector<int>& input_shape,
                const std::vector<int>& pad_before, const std::vector<int>& pad_after,
                const std::vector<int>& result_shape, T pad_value, int pad_mode, size_t total_elements) {
    
    int* d_input_strides;
    int* d_input_shape;
    int* d_pad_before;
    int* d_pad_after;
    int* d_result_shape;
    
    int ndims = input_shape.size();
    
    cudaMalloc(&d_input_strides, ndims * sizeof(int));
    cudaMalloc(&d_input_shape, ndims * sizeof(int));
    cudaMalloc(&d_pad_before, ndims * sizeof(int));
    cudaMalloc(&d_pad_after, ndims * sizeof(int));
    cudaMalloc(&d_result_shape, ndims * sizeof(int));
    
    cudaMemcpy(d_input_strides, input_strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_shape, input_shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pad_before, pad_before.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pad_after, pad_after.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_shape, result_shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    pad_kernel<T><<<grid_size, block_size>>>(
        result, input, d_input_strides, d_input_shape, d_pad_before, d_pad_after, 
        d_result_shape, ndims, pad_value, pad_mode, total_elements
    );
    
    cudaFree(d_input_strides);
    cudaFree(d_input_shape);
    cudaFree(d_pad_before);
    cudaFree(d_pad_after);
    cudaFree(d_result_shape);
} 