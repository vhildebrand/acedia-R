// tensor_kernels.cu
// Core CUDA kernel implementations for tensor operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <algorithm>
#include "kernel_utils.cuh"
#include "../gpuTensor.h"
#include "../cuda_utils.h"

using namespace cuda_utils;

// Generic templated kernels
template<typename T>
__global__ void fill_kernel(T* data, T value, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = value;
    }
}

template<typename T, typename Op>
__global__ void elementwise_binary_kernel(T* result, const T* a, const T* b, size_t n, Op op) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = op(a[tid], b[tid]);
    }
}

template<typename T, typename U, typename Op>
__global__ void elementwise_scalar_kernel(T* result, const T* input, U scalar, size_t n, Op op) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = op(input[tid], convert_type<T>(scalar));
    }
}

// Unary elementwise kernel for math functions like exp, log, sqrt
template<typename T, typename Op>
__global__ void elementwise_unary_kernel(T* result, const T* input, size_t n, Op op) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = op(input[tid]);
    }
}

// Type conversion kernel
template<typename To, typename From>
__global__ void type_conversion_kernel(To* output, const From* input, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = convert_type<To, From>(input[tid]);
    }
}

// Reduction kernel with templated accumulator
template<typename T, typename AccumType, typename Op>
__global__ void reduction_kernel(AccumType* result, const T* input, size_t n, Op op, AccumType init_val) {
    extern __shared__ char shared_mem[];
    AccumType* shared_data = reinterpret_cast<AccumType*>(shared_mem);
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_data[tid] = (i < n) ? convert_type<AccumType>(input[i]) : init_val;
    __syncthreads();
    
    // Tree-based reduction
    for (size_t s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            shared_data[tid] = op(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

// Matrix multiplication kernel with templated types
template<typename T>
__global__ void matmul_kernel(T* C, const T* A, const T* B, size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = T(0);
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized tiled matrix multiplication with shared memory
template<typename T>
__global__ void matmul_tiled_kernel(T* C, const T* A, const T* B, size_t M, size_t N, size_t K) {
    const int TILE_SIZE = 16;
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    T sum = T(0);
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = T(0);
            
        if (col < N && t * TILE_SIZE + ty < K)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = T(0);
            
        __syncthreads();
        
        // Compute
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Generic broadcast kernel for element-wise binary operations
template<typename T, typename Op>
__global__ void broadcast_binary_kernel(
    T* result, 
    const T* a, const T* b,
    const int* a_strides, const int* b_strides, const int* result_strides,
    const int* shape, int ndims, size_t total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear index to multidimensional coordinates (COLUMN-MAJOR for R)
    int coords[8]; // max 8 dimensions
    int temp_idx = idx;
    for (int i = 0; i < ndims; i++) {
        coords[i] = temp_idx % shape[i];
        temp_idx /= shape[i];
    }
    
    // Calculate source indices using strides
    int a_idx = 0, b_idx = 0;
    for (int i = 0; i < ndims; i++) {
        a_idx += coords[i] * a_strides[i];
        b_idx += coords[i] * b_strides[i];
    }
    
    result[idx] = Op{}(a[a_idx], b[b_idx]);
}

// Strided copy kernel for making tensors contiguous
template<typename T>
__global__ void strided_copy_kernel(
    T* dest, const T* src,
    const int* src_strides, const int* shape, int ndims, size_t total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear output index to multidimensional coordinates (COLUMN-MAJOR for R)
    int coords[8]; // max 8 dimensions
    int temp_idx = idx;
    for (int i = 0; i < ndims; i++) {
        coords[i] = temp_idx % shape[i];
        temp_idx /= shape[i];
    }
    
    // Calculate source index using strides
    int src_idx = 0;
    for (int i = 0; i < ndims; i++) {
        src_idx += coords[i] * src_strides[i];
    }
    
    dest[idx] = src[src_idx];
} 

// Concatenation kernel along specified axis
template<typename T>
__global__ void concat_kernel(
    T* result, T** inputs, const int* input_sizes, int num_tensors,
    const int* result_strides, const int* input_strides_list, 
    const int* shape, int ndims, int concat_axis, size_t total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear index to result coordinates (COLUMN-MAJOR for R)
    int coords[8];
    int temp_idx = idx;
    for (int i = 0; i < ndims; i++) {
        coords[i] = temp_idx % shape[i];
        temp_idx /= shape[i];
    }
    
    // Find which input tensor this element comes from
    int concat_coord = coords[concat_axis];
    int input_tensor = 0;
    int offset_in_concat = 0;
    
    for (int t = 0; t < num_tensors; t++) {
        if (concat_coord < offset_in_concat + input_sizes[t]) {
            input_tensor = t;
            break;
        }
        offset_in_concat += input_sizes[t];
    }
    
    // Adjust coordinate for input tensor
    coords[concat_axis] = concat_coord - offset_in_concat;
    
    // Calculate source index
    const int* input_strides = &input_strides_list[input_tensor * ndims];
    int src_idx = 0;
    for (int i = 0; i < ndims; i++) {
        src_idx += coords[i] * input_strides[i];
    }
    
    result[idx] = inputs[input_tensor][src_idx];
}

// Stack kernel - creates new dimension
template<typename T>
__global__ void stack_kernel(
    T* result, T** inputs, int num_tensors,
    const int* input_strides, const int* result_shape, 
    int ndims, int stack_axis, size_t total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear index to result coordinates (COLUMN-MAJOR for R)
    int coords[8];
    int temp_idx = idx;
    for (int i = 0; i < ndims; i++) {
        coords[i] = temp_idx % result_shape[i];
        temp_idx /= result_shape[i];
    }
    
    // Extract which tensor (stack dimension)
    int tensor_idx = coords[stack_axis];
    
    // Calculate input coordinates (remove stack dimension)
    int input_coords[7];
    int j = 0;
    for (int i = 0; i < ndims; i++) {
        if (i != stack_axis) {
            input_coords[j++] = coords[i];
        }
    }
    
    // Calculate source index using input strides (column-major order)
    int src_idx = 0;
    for (int i = 0; i < ndims - 1; ++i) {
        src_idx += input_coords[i] * input_strides[i];
    }
    
    result[idx] = inputs[tensor_idx][src_idx];
}

// Repeat kernel - repeats tensor along specified dimensions
template<typename T>
__global__ void repeat_kernel(
    T* result, const T* input,
    const int* input_strides, const int* repeat_counts,
    const int* input_shape, const int* result_shape,
    int ndims, size_t total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert linear index to result coordinates (COLUMN-MAJOR for R)
    int coords[8];
    int temp_idx = idx;
    for (int i = 0; i < ndims; i++) {
        coords[i] = temp_idx % result_shape[i];
        temp_idx /= result_shape[i];
    }
    
    // Map to input coordinates based on repeat counts
    int input_coords[8];
    for (int i = 0; i < ndims; i++) {
        int rep = repeat_counts[i];
        input_coords[i] = coords[i] / rep;  // integer division to collapse repeats
    }
    
    // Calculate source index
    int src_idx = 0;
    for (int i = 0; i < ndims; i++) {
        src_idx += input_coords[i] * input_strides[i];
    }
    
    result[idx] = input[src_idx];
}

// Pad kernel with different padding modes
template<typename T>
__global__ void pad_kernel(
    T* result, const T* input,
    const int* input_strides, const int* input_shape,
    const int* pad_before, const int* pad_after,
    const int* result_shape, int ndims, 
    T pad_value, int pad_mode, size_t total_elements  // mode: 0=constant, 1=reflect, 2=replicate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Convert to result coordinates (COLUMN-MAJOR for R)
    int coords[8];
    int temp_idx = idx;
    for (int i = 0; i < ndims; i++) {
        coords[i] = temp_idx % result_shape[i];
        temp_idx /= result_shape[i];
    }
    
    // Check if we're in padding region and calculate input coordinates
    bool in_padding = false;
    int input_coords[8];
    
    for (int i = 0; i < ndims; i++) {
        int coord = coords[i];
        if (coord < pad_before[i] || coord >= pad_before[i] + input_shape[i]) {
            in_padding = true;
            if (pad_mode == 0) {  // Constant padding
                result[idx] = pad_value;
                return;
            } else if (pad_mode == 1) {  // Reflect padding
                if (coord < pad_before[i]) {
                    input_coords[i] = pad_before[i] - coord - 1;
                } else {
                    input_coords[i] = 2 * input_shape[i] - (coord - pad_before[i]) - 1;
                }
            } else if (pad_mode == 2) {  // Replicate padding
                input_coords[i] = min(max(coord - pad_before[i], 0), input_shape[i] - 1);
            }
        } else {
            input_coords[i] = coord - pad_before[i];
        }
    }
    
    // Calculate source index
    int src_idx = 0;
    for (int i = 0; i < ndims; i++) {
        src_idx += input_coords[i] * input_strides[i];
    }
    
    result[idx] = input[src_idx];
} 

 