// tensor_ops.cu
// Operation functors and launch helper functions for tensor operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "kernel_utils.cuh"
#include "tensor_kernels.cuh"
#include "../cuda_utils.h"
#include <cublas_v2.h>

// Operation functors are now defined in tensor_kernels.cuh

// Helper macros for common kernel launch patterns
#define LAUNCH_BINARY_KERNEL(result, a, b, n, op) \
    do { \
        int threadsPerBlock = 256; \
        int blocksPerGrid = ((n) + threadsPerBlock - 1) / threadsPerBlock; \
        elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>((result), (a), (b), (n), (op)); \
        cudaDeviceSynchronize(); \
    } while(0)

#define LAUNCH_UNARY_KERNEL(result, input, n, op) \
    do { \
        int threadsPerBlock = 256; \
        int blocksPerGrid = ((n) + threadsPerBlock - 1) / threadsPerBlock; \
        elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>((result), (input), (n), (op)); \
        cudaDeviceSynchronize(); \
    } while(0)

#define LAUNCH_SCALAR_KERNEL(result, input, scalar, n, op) \
    do { \
        int threadsPerBlock = 256; \
        int blocksPerGrid = ((n) + threadsPerBlock - 1) / threadsPerBlock; \
        elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>((result), (input), (scalar), (n), (op)); \
        cudaDeviceSynchronize(); \
    } while(0)

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

// Launch outer product: result[i,j] = a[i] * b[j]
template<typename T>
void launch_outer_product(T* result, const T* a, const T* b, size_t M, size_t N, cudaStream_t stream = 0) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    outer_product_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(result, a, b, M, N);
}

// Launch matrix-vector multiplication: result[i] = sum_j(A[i,j] * v[j])
template<typename T>
void launch_matvec(T* result, const T* A, const T* v, size_t M, size_t N, cudaStream_t stream = 0) {
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    
    // Temporarily use simple version for debugging
    matvec_kernel<<<gridSize, blockSize, 0, stream>>>(result, A, v, M, N);
}

// Launch vector-matrix multiplication: result[j] = sum_i(v[i] * A[i,j])
template<typename T>
void launch_vecmat(T* result, const T* v, const T* A, size_t M, size_t N, cudaStream_t stream = 0) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    vecmat_kernel<<<gridSize, blockSize, 0, stream>>>(result, v, A, M, N);
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

// ===================== Product and Variance Support ===================== //
// SquareOp is now defined in tensor_kernels.cuh

// ===================== Softmax / Argmax Kernels ===================== //

template<typename T>
__global__ void shift_exp_kernel(T* out, const T* in, T shift, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = exp(in[tid] - shift);
    }
}

template<typename T>
__global__ void div_scalar_kernel(T* data, T scalar, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] /= scalar;
    }
}

// ===================== Strided Operation Kernels ===================== //

// Strided kernels for non-contiguous tensor operations
template<typename T, typename Op>
__global__ void strided_unary_kernel(
    cuda_utils::TensorDescriptor out_desc,
    cuda_utils::TensorDescriptor in_desc,
    size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear index to coordinates
    int coords[8];
    out_desc.linear_to_coords(idx, coords);
    
    // Compute offsets using strides
    size_t in_offset = in_desc.compute_offset(coords);
    size_t out_offset = out_desc.compute_offset(coords);
    
    // Perform operation
    T* out_ptr = static_cast<T*>(out_desc.data);
    const T* in_ptr = static_cast<const T*>(in_desc.data);
    
    out_ptr[out_offset] = Op{}(in_ptr[in_offset]);
}

template<typename T, typename Op>
__global__ void strided_binary_kernel(
    cuda_utils::TensorDescriptor out_desc,
    cuda_utils::TensorDescriptor a_desc,
    cuda_utils::TensorDescriptor b_desc,
    size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear index to coordinates
    int coords[8];
    out_desc.linear_to_coords(idx, coords);
    
    // Compute offsets using strides
    size_t a_offset = a_desc.compute_offset(coords);
    size_t b_offset = b_desc.compute_offset(coords);
    size_t out_offset = out_desc.compute_offset(coords);
    
    // Perform operation
    T* out_ptr = static_cast<T*>(out_desc.data);
    const T* a_ptr = static_cast<const T*>(a_desc.data);
    const T* b_ptr = static_cast<const T*>(b_desc.data);
    
    out_ptr[out_offset] = Op{}(a_ptr[a_offset], b_ptr[b_offset]);
}

// Strided scalar operation kernel
template<typename T, typename U, typename Op>
__global__ void strided_scalar_kernel(
    cuda_utils::TensorDescriptor out_desc,
    cuda_utils::TensorDescriptor in_desc,
    U scalar,
    size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear index to coordinates
    int coords[8];
    out_desc.linear_to_coords(idx, coords);
    
    // Compute offsets using strides
    size_t in_offset = in_desc.compute_offset(coords);
    size_t out_offset = out_desc.compute_offset(coords);
    
    // Perform operation
    T* out_ptr = static_cast<T*>(out_desc.data);
    const T* in_ptr = static_cast<const T*>(in_desc.data);
    
    out_ptr[out_offset] = Op{}(in_ptr[in_offset], static_cast<T>(scalar));
}

// BEGIN NEW WRAPPER FUNCTIONS (Sprint 1 enhancements)
extern "C" {

// ===================== Unary Elementwise Math ===================== //

void tensor_exp_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, ExpOp());
    cudaDeviceSynchronize();
}

void tensor_exp_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, ExpOp());
    cudaDeviceSynchronize();
}

void tensor_log_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, LogOp());
    cudaDeviceSynchronize();
}

void tensor_log_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, LogOp());
    cudaDeviceSynchronize();
}

void tensor_sqrt_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, SqrtOp());
    cudaDeviceSynchronize();
}

void tensor_sqrt_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, SqrtOp());
    cudaDeviceSynchronize();
}

// New activation functions
void tensor_tanh_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, TanhOp());
    cudaDeviceSynchronize();
}

void tensor_tanh_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, TanhOp());
    cudaDeviceSynchronize();
}

void tensor_sigmoid_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, SigmoidOp());
    cudaDeviceSynchronize();
}

void tensor_sigmoid_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, SigmoidOp());
    cudaDeviceSynchronize();
}

void tensor_relu_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, ReluOp());
    cudaDeviceSynchronize();
}

void tensor_relu_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, ReluOp());
    cudaDeviceSynchronize();
}

void tensor_sin_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, SinOp());
    cudaDeviceSynchronize();
}

void tensor_sin_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, SinOp());
    cudaDeviceSynchronize();
}

void tensor_cos_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, CosOp());
    cudaDeviceSynchronize();
}

void tensor_cos_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, CosOp());
    cudaDeviceSynchronize();
}

// New math functions for Phase 3.1
void tensor_floor_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, FloorOp());
    cudaDeviceSynchronize();
}

void tensor_floor_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, FloorOp());
    cudaDeviceSynchronize();
}

void tensor_ceil_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, CeilOp());
    cudaDeviceSynchronize();
}

void tensor_ceil_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, CeilOp());
    cudaDeviceSynchronize();
}

void tensor_round_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, RoundOp());
    cudaDeviceSynchronize();
}

void tensor_round_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, RoundOp());
    cudaDeviceSynchronize();
}

void tensor_erf_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, ErfOp());
    cudaDeviceSynchronize();
}

void tensor_erf_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, ErfOp());
    cudaDeviceSynchronize();
}

// Scalar power operation
void tensor_pow_scalar_float32(float* result, const float* input, float exponent, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, exponent, n, PowScalarOp());
    cudaDeviceSynchronize();
}

void tensor_pow_scalar_float64(double* result, const double* input, double exponent, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, exponent, n, PowScalarOp());
    cudaDeviceSynchronize();
}

void tensor_abs_float32(float* result, const float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, AbsOp());
    cudaDeviceSynchronize();
}

void tensor_abs_float64(double* result, const double* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, n, AbsOp());
    cudaDeviceSynchronize();
}

// ===================== Reductions ===================== //

#include <float.h>
#include <cfloat>

// Sum (FLOAT32 & FLOAT64) - using AddOp from above
float tensor_sum_float32(const float* input, size_t n) {
    return launch_reduction<float, float>(input, n, AddOp(), 0.0f);
}

double tensor_sum_float64(const double* input, size_t n) {
    return launch_reduction<double, double>(input, n, AddOp(), 0.0);
}

// Placeholder for half precision & int64 sums (not yet optimized)
float tensor_sum_float16(const half* /*input*/, size_t /*n*/) {
    // TODO: Implement half precision reduction
    return 0.0f;
}

int64_t tensor_sum_int64(const int64_t* /*input*/, size_t /*n*/) {
    // TODO: Implement int64 reduction
    return 0;
}

// Max
float tensor_max_float32(const float* input, size_t n) {
    return launch_reduction<float, float>(input, n, MaxOp(), -FLT_MAX);
}

double tensor_max_float64(const double* input, size_t n) {
    return launch_reduction<double, double>(input, n, MaxOp(), -DBL_MAX);
}

// Min
float tensor_min_float32(const float* input, size_t n) {
    return launch_reduction<float, float>(input, n, MinOp(), FLT_MAX);
}

double tensor_min_float64(const double* input, size_t n) {
    return launch_reduction<double, double>(input, n, MinOp(), DBL_MAX);
}

// ===================== Comparison Functors ===================== //

void tensor_gt_float32(float* result, const float* a, const float* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, GreaterOp());
    cudaDeviceSynchronize();
}

void tensor_gt_float64(double* result, const double* a, const double* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, GreaterOp());
    cudaDeviceSynchronize();
}

void tensor_lt_float32(float* result, const float* a, const float* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, LessOp());
    cudaDeviceSynchronize();
}

void tensor_lt_float64(double* result, const double* a, const double* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, LessOp());
    cudaDeviceSynchronize();
}

void tensor_eq_float32(float* result, const float* a, const float* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, EqualOp());
    cudaDeviceSynchronize();
}

void tensor_eq_float64(double* result, const double* a, const double* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, EqualOp());
    cudaDeviceSynchronize();
}

// Product reductions
float tensor_prod_float32(const float* input, size_t n) {
    return launch_reduction<float, float>(input, n, MulOp(), 1.0f);
}

double tensor_prod_float64(const double* input, size_t n) {
    return launch_reduction<double, double>(input, n, MulOp(), 1.0);
}

// Variance (population) - returns double for precision

double tensor_var_float32(const float* input, size_t n) {
    if (n <= 1) return 0.0;          // Variance undefined for n <= 1, return 0
    // Allocate temp buffer for squares
    float* d_squares = nullptr;
    cudaMalloc(&d_squares, n * sizeof(float));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_squares, input, n, SquareOp());
    cudaDeviceSynchronize();
    
    float sum = launch_reduction<float, float>(input, n, AddOp(), 0.0f);
    float sum_sq = launch_reduction<float, float>(d_squares, n, AddOp(), 0.0f);
    cudaFree(d_squares);
    double mean = static_cast<double>(sum) / static_cast<double>(n);
    // Sample variance with Bessel's correction
    double numerator = static_cast<double>(sum_sq) - static_cast<double>(n) * mean * mean;
    return numerator / static_cast<double>(n - 1);
}

double tensor_var_float64(const double* input, size_t n) {
    if (n <= 1) return 0.0;
    // Allocate temp buffer for squares
    double* d_squares = nullptr;
    cudaMalloc(&d_squares, n * sizeof(double));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_unary_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_squares, input, n, SquareOp());
    cudaDeviceSynchronize();
    
    double sum = launch_reduction<double, double>(input, n, AddOp(), 0.0);
    double sum_sq = launch_reduction<double, double>(d_squares, n, AddOp(), 0.0);
    cudaFree(d_squares);
    double mean = sum / static_cast<double>(n);
    double numerator = sum_sq - static_cast<double>(n) * mean * mean;
    return numerator / static_cast<double>(n - 1);
}

// ----------------- Softmax wrappers -----------------

void tensor_softmax_float32(float* output, const float* input, size_t n) {
    // 1. max
    float max_val = launch_reduction<float,float>(input, n, MaxOp(), -FLT_MAX);
    // 2. exp(x-max)
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    shift_exp_kernel<float><<<blocks, threads>>>(output, input, max_val, n);
    // 3. sum
    float sum_val = launch_reduction<float,float>(output, n, AddOp(), 0.0f);
    // 4. divide
    div_scalar_kernel<float><<<blocks, threads>>>(output, sum_val, n);
}

void tensor_softmax_float64(double* output, const double* input, size_t n) {
    double max_val = launch_reduction<double,double>(input, n, MaxOp(), -DBL_MAX);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    shift_exp_kernel<double><<<blocks, threads>>>(output, input, max_val, n);
    double sum_val = launch_reduction<double,double>(output, n, AddOp(), 0.0);
    div_scalar_kernel<double><<<blocks, threads>>>(output, sum_val, n);
}

// ----------------- Argmax wrappers -----------------

int64_t tensor_argmax_float32(const float* input, size_t n) {
    std::vector<float> host(n);
    cudaMemcpy(host.data(), input, n*sizeof(float), cudaMemcpyDeviceToHost);
    auto it = std::max_element(host.begin(), host.end());
    return static_cast<int64_t>(std::distance(host.begin(), it));
}

int64_t tensor_argmax_float64(const double* input, size_t n) {
    std::vector<double> host(n);
    cudaMemcpy(host.data(), input, n*sizeof(double), cudaMemcpyDeviceToHost);
    auto it = std::max_element(host.begin(), host.end());
    return static_cast<int64_t>(std::distance(host.begin(), it));
}

// ===================== C Wrappers for TensorMutation.cpp ===================== //

void launch_concat_float32(float* result, const float** inputs, const int* input_sizes, int num_tensors,
                           const int* result_strides, const int* input_strides_list, 
                           const int* shape, int ndims, int concat_axis, size_t total_elements) {
    std::vector<int> result_strides_vec(result_strides, result_strides + ndims);
    std::vector<std::vector<int>> input_strides_list_vec(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
        input_strides_list_vec[i] = std::vector<int>(input_strides_list + i * ndims, 
                                                     input_strides_list + (i + 1) * ndims);
    }
    std::vector<int> shape_vec(shape, shape + ndims);
    launch_concat<float>(result, inputs, input_sizes, num_tensors,
                         result_strides_vec, input_strides_list_vec, shape_vec, concat_axis, total_elements);
}

void launch_concat_float64(double* result, const double** inputs, const int* input_sizes, int num_tensors,
                           const int* result_strides, const int* input_strides_list, 
                           const int* shape, int ndims, int concat_axis, size_t total_elements) {
    std::vector<int> result_strides_vec(result_strides, result_strides + ndims);
    std::vector<std::vector<int>> input_strides_list_vec(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
        input_strides_list_vec[i] = std::vector<int>(input_strides_list + i * ndims, 
                                                     input_strides_list + (i + 1) * ndims);
    }
    std::vector<int> shape_vec(shape, shape + ndims);
    launch_concat<double>(result, inputs, input_sizes, num_tensors,
                          result_strides_vec, input_strides_list_vec, shape_vec, concat_axis, total_elements);
}

void launch_stack_float32(float* result, const float** inputs, int num_tensors,
                          const int* input_strides, const int* result_shape, int ndims,
                          int stack_axis, size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + (ndims - 1));
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    launch_stack<float>(result, inputs, num_tensors, input_strides_vec, result_shape_vec, stack_axis, total_elements);
}

void launch_stack_float64(double* result, const double** inputs, int num_tensors,
                          const int* input_strides, const int* result_shape, int ndims,
                          int stack_axis, size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + (ndims - 1));
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    launch_stack<double>(result, inputs, num_tensors, input_strides_vec, result_shape_vec, stack_axis, total_elements);
}

void launch_repeat_float32(float* result, const float* input,
                           const int* input_strides, const int* repeat_counts,
                           const int* input_shape, const int* result_shape, int ndims,
                           size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims);
    std::vector<int> repeat_counts_vec(repeat_counts, repeat_counts + ndims);
    std::vector<int> input_shape_vec(input_shape, input_shape + ndims);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    launch_repeat<float>(result, input, input_strides_vec, repeat_counts_vec, input_shape_vec, result_shape_vec, total_elements);
}

void launch_repeat_float64(double* result, const double* input,
                           const int* input_strides, const int* repeat_counts,
                           const int* input_shape, const int* result_shape, int ndims,
                           size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims);
    std::vector<int> repeat_counts_vec(repeat_counts, repeat_counts + ndims);
    std::vector<int> input_shape_vec(input_shape, input_shape + ndims);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    launch_repeat<double>(result, input, input_strides_vec, repeat_counts_vec, input_shape_vec, result_shape_vec, total_elements);
}

void launch_pad_float32(float* result, const float* input,
                        const int* input_strides, const int* input_shape,
                        const int* pad_before, const int* pad_after,
                        const int* result_shape, int ndims, float pad_value, int pad_mode,
                        size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims);
    std::vector<int> input_shape_vec(input_shape, input_shape + ndims);
    std::vector<int> pad_before_vec(pad_before, pad_before + ndims);
    std::vector<int> pad_after_vec(pad_after, pad_after + ndims);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    launch_pad<float>(result, input, input_strides_vec, input_shape_vec, pad_before_vec, pad_after_vec, 
                      result_shape_vec, pad_value, pad_mode, total_elements);
}

void launch_pad_float64(double* result, const double* input,
                        const int* input_strides, const int* input_shape,
                        const int* pad_before, const int* pad_after,
                        const int* result_shape, int ndims, double pad_value, int pad_mode,
                        size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims);
    std::vector<int> input_shape_vec(input_shape, input_shape + ndims);
    std::vector<int> pad_before_vec(pad_before, pad_before + ndims);
    std::vector<int> pad_after_vec(pad_after, pad_after + ndims);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    launch_pad<double>(result, input, input_strides_vec, input_shape_vec, pad_before_vec, pad_after_vec, 
                       result_shape_vec, pad_value, pad_mode, total_elements);
}

void tensor_div_float16(half* result, const half* a, const half* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, DivOp());
    cudaDeviceSynchronize();
}

// ===================== Missing Arithmetic Functions ===================== //

// Multiplication operations
void tensor_mul_float16(half* result, const half* a, const half* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, MulOp());
    cudaDeviceSynchronize();
}

void tensor_mul_float32(float* result, const float* a, const float* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, MulOp());
    cudaDeviceSynchronize();
}

void tensor_mul_float64(double* result, const double* a, const double* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, MulOp());
    cudaDeviceSynchronize();
}

// Subtraction operations
void tensor_sub_float16(half* result, const half* a, const half* b, size_t n) {
    LAUNCH_BINARY_KERNEL(result, a, b, n, SubOp());
}

void tensor_sub_float32(float* result, const float* a, const float* b, size_t n) {
    LAUNCH_BINARY_KERNEL(result, a, b, n, SubOp());
}

void tensor_sub_float64(double* result, const double* a, const double* b, size_t n) {
    LAUNCH_BINARY_KERNEL(result, a, b, n, SubOp());
}

// Division operations
void tensor_div_float32(float* result, const float* a, const float* b, size_t n) {
    LAUNCH_BINARY_KERNEL(result, a, b, n, DivOp());
}

void tensor_div_float64(double* result, const double* a, const double* b, size_t n) {
    LAUNCH_BINARY_KERNEL(result, a, b, n, DivOp());
}

// Scalar operations
void tensor_scalar_mul_float16(half* result, const half* input, float scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, scalar, n, MulOp());
    cudaDeviceSynchronize();
}

void tensor_scalar_mul_float32(float* result, const float* input, float scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, scalar, n, MulOp());
    cudaDeviceSynchronize();
}

void tensor_scalar_mul_float64(double* result, const double* input, double scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, scalar, n, MulOp());
    cudaDeviceSynchronize();
}

void tensor_scalar_add_float16(half* result, const half* input, float scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, scalar, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_scalar_add_float32(float* result, const float* input, float scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, scalar, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_scalar_add_float64(double* result, const double* input, double scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, scalar, n, AddOp());
    cudaDeviceSynchronize();
}

// ===================== Additional Missing Functions ===================== //

// Addition operations
void tensor_add_float16(half* result, const half* a, const half* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_add_float32(float* result, const float* a, const float* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_add_float64(double* result, const double* a, const double* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_add_int8(int8_t* result, const int8_t* a, const int8_t* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_add_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_add_int64(int64_t* result, const int64_t* a, const int64_t* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, AddOp());
    cudaDeviceSynchronize();
}

// Broadcast operations
void tensor_add_broadcast_float32(float* result, const float* a, const float* b, const int* a_strides, 
                                  const int* b_strides, const int* result_strides, const int* shape, 
                                  int ndims, size_t total_elements) {
    launch_broadcast_binary<float>(result, a, b, a_strides, b_strides, result_strides, shape, ndims, total_elements, AddOp());
}

void tensor_add_broadcast_float64(double* result, const double* a, const double* b, const int* a_strides, 
                                  const int* b_strides, const int* result_strides, const int* shape, 
                                  int ndims, size_t total_elements) {
    launch_broadcast_binary<double>(result, a, b, a_strides, b_strides, result_strides, shape, ndims, total_elements, AddOp());
}

void tensor_mul_broadcast_float32(float* result, const float* a, const float* b, const int* a_strides, 
                                  const int* b_strides, const int* result_strides, const int* shape, 
                                  int ndims, size_t total_elements) {
    launch_broadcast_binary<float>(result, a, b, a_strides, b_strides, result_strides, shape, ndims, total_elements, MulOp());
}

void tensor_mul_broadcast_float64(double* result, const double* a, const double* b, const int* a_strides, 
                                  const int* b_strides, const int* result_strides, const int* shape, 
                                  int ndims, size_t total_elements) {
    launch_broadcast_binary<double>(result, a, b, a_strides, b_strides, result_strides, shape, ndims, total_elements, MulOp());
}

// Matrix multiplication operations
void tensor_matmul_float16(half* C, const half* A, const half* B, size_t M, size_t N, size_t K) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    cublasStatus_t stat = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        &alpha,
        A, CUDA_R_16F, static_cast<int>(M),
        B, CUDA_R_16F, static_cast<int>(K),
        &beta,
        C, CUDA_R_16F, static_cast<int>(M),
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx failed in tensor_matmul_float16");
    }
}

void tensor_matmul_float32(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    cublasStatus_t stat = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        &alpha,
        A, CUDA_R_32F, static_cast<int>(M),
        B, CUDA_R_32F, static_cast<int>(K),
        &beta,
        C, CUDA_R_32F, static_cast<int>(M),
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx failed in tensor_matmul_float32");
    }
}

void tensor_matmul_float64(double* C, const double* A, const double* B, size_t M, size_t N, size_t K) {
    const double alpha = 1.0;
    const double beta  = 0.0;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    cublasStatus_t stat = cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        &alpha,
        A, static_cast<int>(M),
        B, static_cast<int>(K),
        &beta,
        C, static_cast<int>(M));
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasDgemm failed in tensor_matmul_float64");
    }
}

// Outer product operations
void tensor_outer_product_float16(half* result, const half* a, const half* b, size_t M, size_t N) {
    launch_outer_product<half>(result, a, b, M, N);
}

void tensor_outer_product_float32(float* result, const float* a, const float* b, size_t M, size_t N) {
    launch_outer_product<float>(result, a, b, M, N);
}

void tensor_outer_product_float64(double* result, const double* a, const double* b, size_t M, size_t N) {
    launch_outer_product<double>(result, a, b, M, N);
}

// Matrix-vector multiplication operations
void tensor_matvec_float32(float* result, const float* A, const float* v, size_t M, size_t N) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    cublasStatus_t stat = cublasSgemv(
        handle,
        CUBLAS_OP_N,
        static_cast<int>(M), static_cast<int>(N),
        &alpha,
        A, static_cast<int>(M),
        v, 1,
        &beta,
        result, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemv failed in tensor_matvec_float32");
    }
}

void tensor_matvec_float64(double* result, const double* A, const double* v, size_t M, size_t N) {
    const double alpha = 1.0;
    const double beta  = 0.0;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    cublasStatus_t stat = cublasDgemv(
        handle,
        CUBLAS_OP_N,
        static_cast<int>(M), static_cast<int>(N),
        &alpha,
        A, static_cast<int>(M),
        v, 1,
        &beta,
        result, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasDgemv failed in tensor_matvec_float64");
    }
}

// Vector-matrix multiplication operations
void tensor_vecmat_float32(float* result, const float* v, const float* A, size_t M, size_t N) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    // v^T (1×M) * A (M×N) = result (1×N)
    cublasStatus_t stat = cublasSgemv(
        handle,
        CUBLAS_OP_T,
        static_cast<int>(M), static_cast<int>(N),
        &alpha,
        A, static_cast<int>(M),
        v, 1,
        &beta,
        result, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemv (vecmat) failed in tensor_vecmat_float32");
    }
}

void tensor_vecmat_float64(double* result, const double* v, const double* A, size_t M, size_t N) {
    const double alpha = 1.0;
    const double beta  = 0.0;
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    cublasStatus_t stat = cublasDgemv(
        handle,
        CUBLAS_OP_T,
        static_cast<int>(M), static_cast<int>(N),
        &alpha,
        A, static_cast<int>(M),
        v, 1,
        &beta,
        result, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasDgemv (vecmat) failed in tensor_vecmat_float64");
    }
}

// Strided copy operations
void tensor_strided_copy_float32(float* dest, const float* src, const int* strides, const int* shape, int ndims, size_t total_elements) {
    std::vector<int> stride_vec(strides, strides + ndims);
    std::vector<int> shape_vec(shape, shape + ndims);
    launch_strided_copy<float>(dest, src, stride_vec, shape_vec, total_elements);
}

void tensor_strided_copy_float64(double* dest, const double* src, const int* strides, const int* shape, int ndims, size_t total_elements) {
    std::vector<int> stride_vec(strides, strides + ndims);
    std::vector<int> shape_vec(shape, shape + ndims);
    launch_strided_copy<double>(dest, src, stride_vec, shape_vec, total_elements);
}

// Strided operations - now fully implemented with strided kernels
void tensor_add_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<float, AddOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_add_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<double, AddOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_add_strided_float16(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<half, AddOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_add_strided_int8(const cuda_utils::TensorDescriptor& out_desc,
                            const cuda_utils::TensorDescriptor& a_desc,
                            const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<int8_t, AddOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_exp_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<float, ExpOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_exp_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<double, ExpOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

// Additional strided operations for more comprehensive support
void tensor_mul_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<float, MulOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_mul_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<double, MulOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_mul_strided_float16(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<half, MulOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_sub_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<float, SubOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_sub_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<double, SubOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_div_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<float, DivOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_div_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<double, DivOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_sub_strided_float16(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<half, SubOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_div_strided_float16(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<half, DivOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_log_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<float, LogOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_log_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<double, LogOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_sqrt_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                 const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<float, SqrtOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_sqrt_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                 const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<double, SqrtOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

// Strided scalar operations - the key missing piece for view efficiency!
void tensor_scalar_add_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       float scalar) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<float, float, AddOp><<<grid_size, block_size>>>(
        out_desc, in_desc, scalar, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_scalar_add_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       double scalar) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<double, double, AddOp><<<grid_size, block_size>>>(
        out_desc, in_desc, scalar, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_scalar_mul_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       float scalar) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<float, float, MulOp><<<grid_size, block_size>>>(
        out_desc, in_desc, scalar, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_scalar_mul_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       double scalar) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<double, double, MulOp><<<grid_size, block_size>>>(
        out_desc, in_desc, scalar, total_elements
    );
    cudaDeviceSynchronize();
}

// New strided math functions for Phase 3.1
void tensor_floor_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                  const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<float, FloorOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_floor_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                  const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<double, FloorOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_ceil_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                 const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<float, CeilOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_ceil_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                 const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<double, CeilOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_round_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                  const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<float, RoundOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_round_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                  const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<double, RoundOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_erf_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<float, ErfOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_erf_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                const cuda_utils::TensorDescriptor& in_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_unary_kernel<double, ErfOp><<<grid_size, block_size>>>(
        out_desc, in_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_pow_scalar_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       float exponent) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<float, float, PowScalarOp><<<grid_size, block_size>>>(
        out_desc, in_desc, exponent, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_pow_scalar_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       double exponent) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<double, double, PowScalarOp><<<grid_size, block_size>>>(
        out_desc, in_desc, exponent, total_elements
    );
    cudaDeviceSynchronize();
}

// New binary element-wise operations for Phase 3.2
void tensor_max_elemwise_float32(float* result, const float* a, const float* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, MaxOp());
    cudaDeviceSynchronize();
}

void tensor_max_elemwise_float64(double* result, const double* a, const double* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, MaxOp());
    cudaDeviceSynchronize();
}

void tensor_min_elemwise_float32(float* result, const float* a, const float* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, MinOp());
    cudaDeviceSynchronize();
}

void tensor_min_elemwise_float64(double* result, const double* a, const double* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, MinOp());
    cudaDeviceSynchronize();
}

void tensor_pow_elemwise_float32(float* result, const float* a, const float* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, PowOp());
    cudaDeviceSynchronize();
}

void tensor_pow_elemwise_float64(double* result, const double* a, const double* b, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_binary_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, a, b, n, PowOp());
    cudaDeviceSynchronize();
}

// New strided binary element-wise operations for Phase 3.2
void tensor_max_elemwise_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                         const cuda_utils::TensorDescriptor& a_desc,
                                         const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<float, MaxOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_max_elemwise_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                         const cuda_utils::TensorDescriptor& a_desc,
                                         const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<double, MaxOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_min_elemwise_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                         const cuda_utils::TensorDescriptor& a_desc,
                                         const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<float, MinOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_min_elemwise_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                         const cuda_utils::TensorDescriptor& a_desc,
                                         const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<double, MinOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_pow_elemwise_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                         const cuda_utils::TensorDescriptor& a_desc,
                                         const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<float, PowOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_pow_elemwise_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                         const cuda_utils::TensorDescriptor& a_desc,
                                         const cuda_utils::TensorDescriptor& b_desc) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_binary_kernel<double, PowOp><<<grid_size, block_size>>>(
        out_desc, a_desc, b_desc, total_elements
    );
    cudaDeviceSynchronize();
}

// Product reductions

// ... existing code ...

// New scalar operations for Phase 3.2
void tensor_scalar_sub_float32(float* result, const float* input, float scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, -scalar, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_scalar_sub_float64(double* result, const double* input, double scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, -scalar, n, AddOp());
    cudaDeviceSynchronize();
}

void tensor_scalar_div_float32(float* result, const float* input, float scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, 1.0f / scalar, n, MulOp());
    cudaDeviceSynchronize();
}

void tensor_scalar_div_float64(double* result, const double* input, double scalar, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(result, input, 1.0 / scalar, n, MulOp());
    cudaDeviceSynchronize();
}

// New strided scalar operations for Phase 3.2
void tensor_scalar_sub_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       float scalar) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<float, float, AddOp><<<grid_size, block_size>>>(
        out_desc, in_desc, -scalar, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_scalar_sub_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       double scalar) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<double, double, AddOp><<<grid_size, block_size>>>(
        out_desc, in_desc, -scalar, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_scalar_div_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       float scalar) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<float, float, MulOp><<<grid_size, block_size>>>(
        out_desc, in_desc, 1.0f / scalar, total_elements
    );
    cudaDeviceSynchronize();
}

void tensor_scalar_div_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                       const cuda_utils::TensorDescriptor& in_desc,
                                       double scalar) {
    size_t total_elements = out_desc.total_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    strided_scalar_kernel<double, double, MulOp><<<grid_size, block_size>>>(
        out_desc, in_desc, 1.0 / scalar, total_elements
    );
    cudaDeviceSynchronize();
}

// New scalar operations for Phase 3.2

// ... existing code ...

} // extern "C"