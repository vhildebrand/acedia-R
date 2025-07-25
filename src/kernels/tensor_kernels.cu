// tensor_kernels.cu
// Core CUDA kernel implementations for tensor operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <algorithm>
#include "kernel_utils.cuh"
#include "tensor_kernels.cuh"
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

// Outer product kernel: result[i,j] = a[i] * b[j]
// Note: R uses column-major layout, so we use col * M + row indexing
template<typename T>
__global__ void outer_product_kernel(T* result, const T* a, const T* b, size_t M, size_t N) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        // For R's column-major layout: result[i,j] is at col * M + row
        result[col * M + row] = a[row] * b[col];
    }
}

// Matrix-vector multiplication kernel: result[i] = sum_j(A[i,j] * v[j])
// Note: R uses column-major layout, so A[i,j] is at j * M + i
template<typename T>
__global__ void matvec_kernel(T* result, const T* A, const T* v, size_t M, size_t N) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        T sum = T(0);
        for (size_t j = 0; j < N; ++j) {
            // For R's column-major layout: A[i,j] is at j * M + i
            sum += A[j * M + row] * v[j];
        }
        result[row] = sum;
    }
}

// Optimized matrix-vector multiplication with shared memory reduction
template<typename T>
__global__ void matvec_optimized_kernel(T* result, const T* A, const T* v, size_t M, size_t N) {
    extern __shared__ char shared_mem[];
    T* shared_v = reinterpret_cast<T*>(shared_mem);
    
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t blockSize = blockDim.x;
    
    if (row >= M) return;
    
    // Load vector into shared memory (if it fits)
    if (N <= blockSize) {
        if (tid < N) {
            shared_v[tid] = v[tid];
        }
        __syncthreads();
        
        // Each thread computes partial sum
        T sum = T(0);
        for (size_t j = tid; j < N; j += blockSize) {
            // For R's column-major layout: A[i,j] is at j * M + i
            sum += A[j * M + row] * shared_v[j];
        }
        
        // Reduce within block
        shared_v[tid] = sum;
        __syncthreads();
        
        // Tree reduction
        for (size_t s = blockSize / 2; s > 0; s /= 2) {
            if (tid < s) {
                shared_v[tid] += shared_v[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            result[row] = shared_v[0];
        }
    } else {
        // Fallback to simple version for large vectors
        if (tid == 0) {
            T sum = T(0);
            for (size_t j = 0; j < N; ++j) {
                // For R's column-major layout: A[i,j] is at j * M + i  
                sum += A[j * M + row] * v[j];
            }
            result[row] = sum;
        }
    }
}

// Vector-matrix multiplication kernel: result[j] = sum_i(v[i] * A[i,j])
// Note: R uses column-major layout, so A[i,j] is at j * M + i
template<typename T>
__global__ void vecmat_kernel(T* result, const T* v, const T* A, size_t M, size_t N) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < N) {
        T sum = T(0);
        for (size_t i = 0; i < M; ++i) {
            // For R's column-major layout: A[i,j] is at j * M + i
            sum += v[i] * A[col * M + i];
        }
        result[col] = sum;
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
    
    // Calculate source index using input  strides (column-major order)
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

// NEW: Axis-aware reduction kernels

/**
 * @brief Axis-aware reduction kernel that reduces along specified axes
 * 
 * This kernel is designed so that each thread block computes one output element.
 * It handles non-contiguous tensors using strides and can reduce along multiple axes.
 * 
 * @param result Output tensor
 * @param input Input tensor 
 * @param input_strides Input tensor strides
 * @param input_shape Input tensor shape
 * @param result_strides Output tensor strides
 * @param reduction_axes Array of axes to reduce over
 * @param num_reduction_axes Number of axes being reduced
 * @param ndims Number of dimensions in input tensor
 * @param output_size Total number of elements in output tensor
 * @param op Reduction operation (0=sum, 1=mean, 2=max, 3=min, 4=prod)
 */
template<typename T, typename AccumType>
__global__ void axis_reduction_kernel(
    T* result,
    const T* input,
    const int* input_strides,
    const int* input_shape,
    const int* result_strides,
    const int* reduction_axes,
    int num_reduction_axes,
    int ndims,
    size_t output_size,
    int op  // 0=sum, 1=mean, 2=max, 3=min, 4=prod
) {
    // Each thread block handles one output element
    int output_idx = blockIdx.x;
    if (output_idx >= output_size) return;
    
    // Convert linear output index to output coordinates (COLUMN-MAJOR for R)
    int output_coords[8];
    int temp_idx = output_idx;
    
    // Calculate result shape from input shape (fixed version)
    int result_shape[8];
    int result_dim = 0;
    for (int i = 0; i < ndims; i++) {
        bool is_reduction_axis = false;
        for (int j = 0; j < num_reduction_axes; j++) {
            if (reduction_axes[j] == i) {
                is_reduction_axis = true;
                break;
            }
        }
        if (!is_reduction_axis) {
            result_shape[result_dim] = input_shape[i];
            result_dim++;
        }
    }
    
    // Convert output index to coordinates with bounds checking
    for (int i = 0; i < result_dim && i < 8; i++) {
        if (result_shape[i] > 0) {
            output_coords[i] = temp_idx % result_shape[i];
            temp_idx /= result_shape[i];
        } else {
            output_coords[i] = 0;
        }
    }
    
    // Map output coordinates back to input coordinates (inserting reduced dimensions)
    int input_coords[8];
    int out_dim = 0;
    for (int i = 0; i < ndims; i++) {
        bool is_reduction_axis = false;
        for (int j = 0; j < num_reduction_axes; j++) {
            if (reduction_axes[j] == i) {
                is_reduction_axis = true;
                break;
            }
        }
        if (is_reduction_axis) {
            input_coords[i] = 0;  // Start from 0 for reduced dimensions
        } else {
            if (out_dim < result_dim) {
                input_coords[i] = output_coords[out_dim];
                out_dim++;
            } else {
                input_coords[i] = 0;  // Safety fallback
            }
        }
    }
    
    // Initialize accumulator
    AccumType accumulator;
    if (op == 0 || op == 1) {  // sum or mean
        accumulator = AccumType(0);
    } else if (op == 4) {  // prod
        accumulator = AccumType(1);
    } else {  // max or min - will be initialized with first value
        accumulator = AccumType(0);  // placeholder
    }
    
    bool first_element = true;
    size_t num_elements = 1;
    
    // Calculate total number of elements to reduce over
    for (int i = 0; i < num_reduction_axes; i++) {
        num_elements *= input_shape[reduction_axes[i]];
    }
    
    // Each thread processes a subset of the reduction
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    // Use generic shared memory buffer and cast to AccumType*
    extern __shared__ char shared_mem[];
    AccumType* sdata = reinterpret_cast<AccumType*>(shared_mem);
    // shared_mem size is blockDim.x * sizeof(AccumType) allocated at launch
 
    AccumType thread_accumulator;
    
    if (op == 0 || op == 1) {  // sum or mean
        thread_accumulator = AccumType(0);
    } else if (op == 4) {  // prod
        thread_accumulator = AccumType(1);
    } else {
        thread_accumulator = AccumType(0);  // will be set with first element
    }
    
    bool thread_first = true;
    
    // Iterate over all combinations of reduction axes
    // This is a simplified version - for complex multi-axis reductions,
    // a more sophisticated iteration strategy would be needed
    for (size_t elem_idx = tid; elem_idx < num_elements; elem_idx += blockSize) {
        // Convert element index to coordinates in the reduction space
        size_t temp_elem = elem_idx;
        int reduction_coords[8];
        
        for (int i = num_reduction_axes - 1; i >= 0; i--) {
            int axis = reduction_axes[i];
            reduction_coords[i] = temp_elem % input_shape[axis];
            temp_elem /= input_shape[axis];
        }
        
        // Set input coordinates for this element
        for (int i = 0; i < num_reduction_axes; i++) {
            input_coords[reduction_axes[i]] = reduction_coords[i];
        }
        
        // Calculate linear input index
        size_t input_idx = 0;
        for (int i = 0; i < ndims; i++) {
            input_idx += static_cast<size_t>(input_coords[i]) * static_cast<size_t>(input_strides[i]);
        }
        
        // Accumulate value
        AccumType val = convert_type<AccumType>(input[input_idx]);
        
        if (op == 0 || op == 1) {  // sum or mean
            thread_accumulator += val;
        } else if (op == 2) {  // max
            if (thread_first) {
                thread_accumulator = val;
                thread_first = false;
            } else {
                thread_accumulator = max(thread_accumulator, val);
            }
        } else if (op == 3) {  // min
            if (thread_first) {
                thread_accumulator = val;
                thread_first = false;
            } else {
                thread_accumulator = min(thread_accumulator, val);
            }
        } else if (op == 4) {  // prod
            thread_accumulator *= val;
        }
    }
    
    // Store thread result in shared memory, but handle threads that processed no elements
    if (op == 3 && thread_first) {  // min operation and thread processed no elements
        // Set to positive infinity so it doesn't affect the min reduction
        if constexpr (std::is_same_v<AccumType, float>) {
            sdata[tid] = INFINITY;
        } else if constexpr (std::is_same_v<AccumType, double>) {
            sdata[tid] = INFINITY;
        } else {
            sdata[tid] = thread_accumulator;  // Fallback for other types
        }
    } else if (op == 2 && thread_first) {  // max operation and thread processed no elements
        // Set to negative infinity so it doesn't affect the max reduction
        if constexpr (std::is_same_v<AccumType, float>) {
            sdata[tid] = -INFINITY;
        } else if constexpr (std::is_same_v<AccumType, double>) {
            sdata[tid] = -INFINITY;
        } else {
            sdata[tid] = thread_accumulator;  // Fallback for other types
        }
    } else {
        sdata[tid] = thread_accumulator;
    }
    __syncthreads();
    
    // Reduce within block
    for (int s = blockSize / 2; s > 0; s /= 2) {
        if (tid < s) {
            if (op == 0 || op == 1) {  // sum or mean
                sdata[tid] += sdata[tid + s];
            } else if (op == 2) {  // max
                sdata[tid] = max(sdata[tid], sdata[tid + s]);
            } else if (op == 3) {  // min
                sdata[tid] = min(sdata[tid], sdata[tid + s]);
            } else if (op == 4) {  // prod
                sdata[tid] *= sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes final result
    if (tid == 0) {
        AccumType final_result = sdata[0];
        
        // For mean, divide by number of elements
        if (op == 1) {
            final_result /= AccumType(num_elements);
        }
        
        result[output_idx] = convert_type<T>(final_result);
    }
}

// Variance kernel (needs special handling for two-pass algorithm)

/**
 * @brief Axis-aware argmax/argmin kernel that finds extreme values and their indices
 * 
 * This kernel computes argmax or argmin along specified reduction axes.
 * Each thread block handles one output element.
 * 
 * @param indices_out Output tensor for indices (int64_t)
 * @param values_out Output tensor for values (can be nullptr if only indices needed)
 * @param input Input tensor
 * @param in_strides Input tensor strides
 * @param in_shape Input tensor shape
 * @param out_strides Output tensor strides (unused but kept for consistency)
 * @param red_axes Array of axes to reduce over
 * @param n_red_axes Number of axes being reduced
 * @param ndims Number of dimensions in input tensor
 * @param out_elems Total number of elements in output tensor
 * @param op Operation type (0=argmax, 1=argmin)
 */
template<typename T, typename AccumType>
__global__ void axis_arg_extreme_kernel(
    int64_t* indices_out, T* values_out,
    const T* input,
    const int* in_strides, const int* in_shape,
    const int* out_strides,
    const int* red_axes, int n_red_axes,
    int ndims, size_t out_elems,
    int op  // 0=argmax, 1=argmin
) {
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_elems) return;
    
    // Convert linear output index to coordinates for non-reduced dimensions
    int coords[8] = {0};
    size_t tmp = out_idx;
    
    // Build coordinate array for input tensor, leaving reduced dims as 0
    int out_dim_idx = 0;
    for (int d = 0; d < ndims; ++d) {
        bool is_reduced = false;
        for (int j = 0; j < n_red_axes; ++j) {
            if (red_axes[j] == d) {
                is_reduced = true;
                break;
            }
        }
        
        if (!is_reduced) {
            // This dimension appears in output - extract coordinate
            coords[d] = tmp % in_shape[d];
            tmp /= in_shape[d];
        }
        // else: coords[d] remains 0 (will be iterated over in reduction loop)
    }
    
    // Walk through reduction space to find extreme value
    int64_t best_idx = 0;
    AccumType best_val;
    bool first = true;
    
    // Calculate base linear index for non-reduced coordinates
    size_t base_linear = 0;
    for (int d = 0; d < ndims; ++d) {
        base_linear += coords[d] * in_strides[d];
    }
    
    // Calculate total elements in reduction space
    size_t red_total = 1;
    for (int j = 0; j < n_red_axes; ++j) {
        red_total *= in_shape[red_axes[j]];
    }
    
    // Iterate through all combinations in reduction space
    for (size_t r = 0; r < red_total; ++r) {
        // Convert r to reduction coordinates
        size_t tmp_r = r;
        size_t linear_idx = base_linear;
        
        for (int j = n_red_axes - 1; j >= 0; --j) {
            int axis = red_axes[j];
            int dim_size = in_shape[axis];
            int coord_val = tmp_r % dim_size;
            tmp_r /= dim_size;
            linear_idx += coord_val * in_strides[axis];
        }
        
        AccumType val = convert_type<AccumType>(input[linear_idx]);
        
        if (first) {
            first = false;
            best_val = val;
            best_idx = r;
        } else {
            bool better = (op == 0) ? (val > best_val) : (val < best_val);
            if (better) {
                best_val = val;
                best_idx = r;
            }
        }
    }
    
    // Store results
    indices_out[out_idx] = best_idx;
    if (values_out) {
        values_out[out_idx] = convert_type<T>(best_val);
    }
}

// Variance kernel (needs special handling for two-pass algorithm)
template<typename T>
__global__ void axis_variance_kernel(
    T* result,
    const T* input,
    const int* input_strides,
    const int* input_shape,
    const int* result_strides,
    const int* reduction_axes,
    int num_reduction_axes,
    int ndims,
    size_t output_size
) {
    // Similar structure to axis_reduction_kernel but with two-pass variance computation
    // First pass: compute mean
    // Second pass: compute squared deviations from mean
    // This is a placeholder - full implementation would be more complex
    
    int output_idx = blockIdx.x;
    if (output_idx >= output_size) return;
    
    // For now, just set to zero - this needs proper two-pass implementation
    if (threadIdx.x == 0) {
        result[output_idx] = T(0);
    }
} 

// Argmax/Argmin kernels
template<typename T>
__global__ void argmax_kernel(int64_t* result, const T* input, size_t n) {
    extern __shared__ char shared_mem[];
    T* shared_values = reinterpret_cast<T*>(shared_mem);
    int64_t* shared_indices = reinterpret_cast<int64_t*>(shared_mem + blockDim.x * sizeof(T));
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (i < n) {
        shared_values[tid] = input[i];
        shared_indices[tid] = static_cast<int64_t>(i);
    } else {
        if constexpr (std::is_same_v<T, float>) {
            shared_values[tid] = -INFINITY;
        } else if constexpr (std::is_same_v<T, double>) {
            shared_values[tid] = -INFINITY;
        } else {
            shared_values[tid] = T(-INFINITY);
        }
        shared_indices[tid] = -1;
    }
    __syncthreads();
    
    // Tree-based reduction for argmax
    for (size_t s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            if (shared_values[tid] < shared_values[tid + s]) {
                shared_values[tid] = shared_values[tid + s];
                shared_indices[tid] = shared_indices[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        result[blockIdx.x] = shared_indices[0];
    }
}

template<typename T>
__global__ void argmin_kernel(int64_t* result, const T* input, size_t n) {
    extern __shared__ char shared_mem[];
    T* shared_values = reinterpret_cast<T*>(shared_mem);
    int64_t* shared_indices = reinterpret_cast<int64_t*>(shared_mem + blockDim.x * sizeof(T));
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (i < n) {
        shared_values[tid] = input[i];
        shared_indices[tid] = static_cast<int64_t>(i);
    } else {
        if constexpr (std::is_same_v<T, float>) {
            shared_values[tid] = INFINITY;
        } else if constexpr (std::is_same_v<T, double>) {
            shared_values[tid] = INFINITY;
        } else {
            shared_values[tid] = T(INFINITY);
        }
        shared_indices[tid] = -1;
    }
    __syncthreads();
    
    // Tree-based reduction for argmin
    for (size_t s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            if (shared_values[tid] > shared_values[tid + s]) {
                shared_values[tid] = shared_values[tid + s];
                shared_indices[tid] = shared_indices[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        result[blockIdx.x] = shared_indices[0];
    }
}

// Helper kernel for copying values at indices
template<typename T>
__global__ void copy_values_at_indices_kernel(T* dest, const T* src, const int64_t* indices, int n, T default_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && indices[i] >= 0) {
        dest[i] = src[indices[i]];
    } else if (i < n) {
        dest[i] = default_val;
    }
}

// C wrapper functions for axis-aware reductions

extern "C" {

// Axis-aware sum reduction
void tensor_axis_sum_float32(
    float* result, const float* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    // Add comprehensive error checking
    cudaError_t err;
    
    // Check input parameters
    if (!result || !input || !input_strides || !input_shape || !result_strides || !reduction_axes) {
        throw std::runtime_error("Null pointer passed to tensor_axis_sum_float32");
    }
    
    if (output_size == 0) {
        throw std::runtime_error("Zero output size in tensor_axis_sum_float32");
    }
    
    // Clear any previous CUDA errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "Pre-existing CUDA error before kernel launch: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
    
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(float);
    
    // Validate launch parameters
    if (gridSize <= 0) {
        throw std::runtime_error("Invalid grid size in tensor_axis_sum_float32: " + std::to_string(gridSize));
    }
    
    axis_reduction_kernel<float, float><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 0  // 0 = sum
    );
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel launch failed in tensor_axis_sum_float32: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
    
    // Synchronize and check for kernel execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel execution failed in tensor_axis_sum_float32: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
}

void tensor_axis_sum_float64(
    double* result, const double* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    cudaError_t err;

    // Pre-launch sanity checks
    if (!result || !input || !input_strides || !input_shape || !result_strides || !reduction_axes) {
        throw std::runtime_error("Null pointer passed to tensor_axis_sum_float64");
    }
    if (output_size == 0) {
        throw std::runtime_error("Zero output size in tensor_axis_sum_float64");
    }

    // Clear any stale error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string msg = "Pre-existing CUDA error before kernel launch: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }

    int blockSize = 256;
    int gridSize  = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(double);
    if (gridSize <= 0) {
        throw std::runtime_error("Invalid grid size in tensor_axis_sum_float64: " + std::to_string(gridSize));
    }

    axis_reduction_kernel<double, double><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 0  // 0 = sum
    );

    // Post-launch checks
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string msg = "CUDA kernel launch failed in tensor_axis_sum_float64: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::string msg = "CUDA kernel execution failed in tensor_axis_sum_float64: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}

// Axis-aware mean reduction
void tensor_axis_mean_float32(
    float* result, const float* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(float);
    
    axis_reduction_kernel<float, float><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 1  // 1 = mean
    );
    cudaDeviceSynchronize();
}

void tensor_axis_mean_float64(
    double* result, const double* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(double);
    
    axis_reduction_kernel<double, double><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 1  // 1 = mean
    );
    cudaDeviceSynchronize();
}

// Axis-aware max reduction
void tensor_axis_max_float32(
    float* result, const float* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(float);
    
    axis_reduction_kernel<float, float><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 2  // 2 = max
    );
    cudaDeviceSynchronize();
}

void tensor_axis_max_float64(
    double* result, const double* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(double);
    
    axis_reduction_kernel<double, double><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 2  // 2 = max
    );
    cudaDeviceSynchronize();
}

// Axis-aware min reduction
void tensor_axis_min_float32(
    float* result, const float* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(float);
    
    axis_reduction_kernel<float, float><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 3  // 3 = min
    );
    cudaDeviceSynchronize();
}

void tensor_axis_min_float64(
    double* result, const double* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(double);
    
    axis_reduction_kernel<double, double><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 3  // 3 = min
    );
    cudaDeviceSynchronize();
}

// Axis-aware prod reduction
void tensor_axis_prod_float32(
    float* result, const float* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(float);
    
    axis_reduction_kernel<float, float><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 4  // 4 = prod
    );
    cudaDeviceSynchronize();
}

void tensor_axis_prod_float64(
    double* result, const double* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    size_t shared_mem_size = blockSize * sizeof(double);
    
    axis_reduction_kernel<double, double><<<gridSize, blockSize, shared_mem_size>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size, 4  // 4 = prod
    );
    cudaDeviceSynchronize();
}

// Axis-aware variance reduction
void tensor_axis_var_float32(
    float* result, const float* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    
    axis_variance_kernel<float><<<gridSize, blockSize>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size
    );
    cudaDeviceSynchronize();
}

void tensor_axis_var_float64(
    double* result, const double* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
) {
    int blockSize = 256;
    int gridSize = static_cast<int>(output_size);
    
    axis_variance_kernel<double><<<gridSize, blockSize>>>(
        result, input, input_strides, input_shape, result_strides,
        reduction_axes, num_reduction_axes, ndims, output_size
    );
    cudaDeviceSynchronize();
}

// ------------------------------------------------------------
// NEW: Global argmax / argmin wrappers (float32 & float64)
// ------------------------------------------------------------

static int64_t launch_and_fetch_index(void (*kernel)(int64_t*, const void*, size_t), const void* d_input, size_t n) {
    // Never called – template trick placeholder.  Real overloads below.
    return -1;
}

// tensor_argmax_* are provided in tensor_ops.cu; avoid duplicate definitions here.
// float32 ARGMIN
int64_t tensor_argmin_float32(const float* input, size_t n) {
    int64_t* d_idx;
    cudaMalloc(&d_idx, sizeof(int64_t));
    int blockSize = 256;
    argmin_kernel<float><<<1, blockSize>>>(d_idx, input, n);
    cudaDeviceSynchronize();
    int64_t h_idx;
    cudaMemcpy(&h_idx, d_idx, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_idx);
    return h_idx;
}

// float64 ARGMIN
int64_t tensor_argmin_float64(const double* input, size_t n) {
    int64_t* d_idx;
    cudaMalloc(&d_idx, sizeof(int64_t));
    int blockSize = 256;
    argmin_kernel<double><<<1, blockSize>>>(d_idx, input, n);
    cudaDeviceSynchronize();
    int64_t h_idx;
    cudaMemcpy(&h_idx, d_idx, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_idx);
    return h_idx;
}

// ------------------------------------------------------------
// NEW: Axis-aware argmax / argmin wrappers
// ------------------------------------------------------------

// Axis-aware ARGMAX wrappers
void tensor_axis_argmax_float32(
    int64_t* idx_out, const float* input,
    const int* in_strides, const int* in_shape, 
    const int* out_strides, const int* axes, 
    int n_axes, int ndims, size_t out_elems
) {
    cudaError_t err;
    
    // Pre-launch checks
    if (!idx_out || !input || !in_strides || !in_shape || !axes) {
        throw std::runtime_error("Null pointer passed to tensor_axis_argmax_float32");
    }
    if (out_elems == 0) {
        throw std::runtime_error("Zero output size in tensor_axis_argmax_float32");
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Pre-existing CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    
    if (out_elems > INT_MAX) {
        throw std::runtime_error("Output size too large for grid dimensions");
    }
    
    int blockSize = 256;
    axis_arg_extreme_kernel<float, float><<<out_elems, blockSize>>>(
        idx_out, nullptr, input, in_strides, in_shape,
        out_strides, axes, n_axes, ndims, out_elems, 0  // 0 = argmax
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed in tensor_axis_argmax_float32: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution failed in tensor_axis_argmax_float32: " + std::string(cudaGetErrorString(err)));
    }
}

void tensor_axis_argmax_float64(
    int64_t* idx_out, const double* input,
    const int* in_strides, const int* in_shape, 
    const int* out_strides, const int* axes, 
    int n_axes, int ndims, size_t out_elems
) {
    cudaError_t err;
    
    if (!idx_out || !input || !in_strides || !in_shape || !axes) {
        throw std::runtime_error("Null pointer passed to tensor_axis_argmax_float64");
    }
    if (out_elems == 0) {
        throw std::runtime_error("Zero output size in tensor_axis_argmax_float64");
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Pre-existing CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    
    if (out_elems > INT_MAX) {
        throw std::runtime_error("Output size too large for grid dimensions");
    }
    
    int blockSize = 256;
    axis_arg_extreme_kernel<double, double><<<out_elems, blockSize>>>(
        idx_out, nullptr, input, in_strides, in_shape,
        out_strides, axes, n_axes, ndims, out_elems, 0  // 0 = argmax
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed in tensor_axis_argmax_float64: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution failed in tensor_axis_argmax_float64: " + std::string(cudaGetErrorString(err)));
    }
}

// Axis-aware ARGMIN wrappers
void tensor_axis_argmin_float32(
    int64_t* idx_out, const float* input,
    const int* in_strides, const int* in_shape, 
    const int* out_strides, const int* axes, 
    int n_axes, int ndims, size_t out_elems
) {
    cudaError_t err;
    
    if (!idx_out || !input || !in_strides || !in_shape || !axes) {
        throw std::runtime_error("Null pointer passed to tensor_axis_argmin_float32");
    }
    if (out_elems == 0) {
        throw std::runtime_error("Zero output size in tensor_axis_argmin_float32");
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Pre-existing CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    
    if (out_elems > INT_MAX) {
        throw std::runtime_error("Output size too large for grid dimensions");
    }
    
    int blockSize = 256;
    axis_arg_extreme_kernel<float, float><<<out_elems, blockSize>>>(
        idx_out, nullptr, input, in_strides, in_shape,
        out_strides, axes, n_axes, ndims, out_elems, 1  // 1 = argmin
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed in tensor_axis_argmin_float32: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution failed in tensor_axis_argmin_float32: " + std::string(cudaGetErrorString(err)));
    }
}

void tensor_axis_argmin_float64(
    int64_t* idx_out, const double* input,
    const int* in_strides, const int* in_shape, 
    const int* out_strides, const int* axes, 
    int n_axes, int ndims, size_t out_elems
) {
    cudaError_t err;
    
    if (!idx_out || !input || !in_strides || !in_shape || !axes) {
        throw std::runtime_error("Null pointer passed to tensor_axis_argmin_float64");
    }
    if (out_elems == 0) {
        throw std::runtime_error("Zero output size in tensor_axis_argmin_float64");
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Pre-existing CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    
    if (out_elems > INT_MAX) {
        throw std::runtime_error("Output size too large for grid dimensions");
    }
    
    int blockSize = 256;
    axis_arg_extreme_kernel<double, double><<<out_elems, blockSize>>>(
        idx_out, nullptr, input, in_strides, in_shape,
        out_strides, axes, n_axes, ndims, out_elems, 1  // 1 = argmin
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed in tensor_axis_argmin_float64: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution failed in tensor_axis_argmin_float64: " + std::string(cudaGetErrorString(err)));
    }
}

} // extern "C"
// Explicit template instantiations for kernels used by tensor_ops.cu
// This ensures the linker can find all the template specializations

// Elementwise binary kernels
template __global__ void elementwise_binary_kernel<float, AddOp>(float*, const float*, const float*, size_t, AddOp);
template __global__ void elementwise_binary_kernel<double, AddOp>(double*, const double*, const double*, size_t, AddOp);
template __global__ void elementwise_binary_kernel<half, AddOp>(half*, const half*, const half*, size_t, AddOp);
template __global__ void elementwise_binary_kernel<int8_t, AddOp>(int8_t*, const int8_t*, const int8_t*, size_t, AddOp);
template __global__ void elementwise_binary_kernel<int32_t, AddOp>(int32_t*, const int32_t*, const int32_t*, size_t, AddOp);
template __global__ void elementwise_binary_kernel<int64_t, AddOp>(int64_t*, const int64_t*, const int64_t*, size_t, AddOp);

template __global__ void elementwise_binary_kernel<float, MulOp>(float*, const float*, const float*, size_t, MulOp);
template __global__ void elementwise_binary_kernel<double, MulOp>(double*, const double*, const double*, size_t, MulOp);
template __global__ void elementwise_binary_kernel<half, MulOp>(half*, const half*, const half*, size_t, MulOp);

template __global__ void elementwise_binary_kernel<float, SubOp>(float*, const float*, const float*, size_t, SubOp);
template __global__ void elementwise_binary_kernel<double, SubOp>(double*, const double*, const double*, size_t, SubOp);
template __global__ void elementwise_binary_kernel<half, SubOp>(half*, const half*, const half*, size_t, SubOp);

template __global__ void elementwise_binary_kernel<float, DivOp>(float*, const float*, const float*, size_t, DivOp);
template __global__ void elementwise_binary_kernel<double, DivOp>(double*, const double*, const double*, size_t, DivOp);
template __global__ void elementwise_binary_kernel<half, DivOp>(half*, const half*, const half*, size_t, DivOp);

template __global__ void elementwise_binary_kernel<float, GreaterOp>(float*, const float*, const float*, size_t, GreaterOp);
template __global__ void elementwise_binary_kernel<double, GreaterOp>(double*, const double*, const double*, size_t, GreaterOp);

template __global__ void elementwise_binary_kernel<float, LessOp>(float*, const float*, const float*, size_t, LessOp);
template __global__ void elementwise_binary_kernel<double, LessOp>(double*, const double*, const double*, size_t, LessOp);

template __global__ void elementwise_binary_kernel<float, EqualOp>(float*, const float*, const float*, size_t, EqualOp);
template __global__ void elementwise_binary_kernel<double, EqualOp>(double*, const double*, const double*, size_t, EqualOp);

// New binary operations for Phase 3.2
template __global__ void elementwise_binary_kernel<float, MaxOp>(float*, const float*, const float*, size_t, MaxOp);
template __global__ void elementwise_binary_kernel<double, MaxOp>(double*, const double*, const double*, size_t, MaxOp);

template __global__ void elementwise_binary_kernel<float, MinOp>(float*, const float*, const float*, size_t, MinOp);
template __global__ void elementwise_binary_kernel<double, MinOp>(double*, const double*, const double*, size_t, MinOp);

template __global__ void elementwise_binary_kernel<float, PowOp>(float*, const float*, const float*, size_t, PowOp);
template __global__ void elementwise_binary_kernel<double, PowOp>(double*, const double*, const double*, size_t, PowOp);

// Elementwise unary kernels
template __global__ void elementwise_unary_kernel<float, ExpOp>(float*, const float*, size_t, ExpOp);
template __global__ void elementwise_unary_kernel<double, ExpOp>(double*, const double*, size_t, ExpOp);

template __global__ void elementwise_unary_kernel<float, LogOp>(float*, const float*, size_t, LogOp);
template __global__ void elementwise_unary_kernel<double, LogOp>(double*, const double*, size_t, LogOp);

template __global__ void elementwise_unary_kernel<float, SqrtOp>(float*, const float*, size_t, SqrtOp);
template __global__ void elementwise_unary_kernel<double, SqrtOp>(double*, const double*, size_t, SqrtOp);

template __global__ void elementwise_unary_kernel<float, TanhOp>(float*, const float*, size_t, TanhOp);
template __global__ void elementwise_unary_kernel<double, TanhOp>(double*, const double*, size_t, TanhOp);

template __global__ void elementwise_unary_kernel<float, SigmoidOp>(float*, const float*, size_t, SigmoidOp);
template __global__ void elementwise_unary_kernel<double, SigmoidOp>(double*, const double*, size_t, SigmoidOp);

template __global__ void elementwise_unary_kernel<float, ReluOp>(float*, const float*, size_t, ReluOp);
template __global__ void elementwise_unary_kernel<double, ReluOp>(double*, const double*, size_t, ReluOp);

template __global__ void elementwise_unary_kernel<float, SinOp>(float*, const float*, size_t, SinOp);
template __global__ void elementwise_unary_kernel<double, SinOp>(double*, const double*, size_t, SinOp);

template __global__ void elementwise_unary_kernel<float, CosOp>(float*, const float*, size_t, CosOp);
template __global__ void elementwise_unary_kernel<double, CosOp>(double*, const double*, size_t, CosOp);

template __global__ void elementwise_unary_kernel<float, AbsOp>(float*, const float*, size_t, AbsOp);
template __global__ void elementwise_unary_kernel<double, AbsOp>(double*, const double*, size_t, AbsOp);

template __global__ void elementwise_unary_kernel<float, SquareOp>(float*, const float*, size_t, SquareOp);
template __global__ void elementwise_unary_kernel<double, SquareOp>(double*, const double*, size_t, SquareOp);

// New unary operations for Phase 3.1
template __global__ void elementwise_unary_kernel<float, FloorOp>(float*, const float*, size_t, FloorOp);
template __global__ void elementwise_unary_kernel<double, FloorOp>(double*, const double*, size_t, FloorOp);

template __global__ void elementwise_unary_kernel<float, CeilOp>(float*, const float*, size_t, CeilOp);
template __global__ void elementwise_unary_kernel<double, CeilOp>(double*, const double*, size_t, CeilOp);

template __global__ void elementwise_unary_kernel<float, RoundOp>(float*, const float*, size_t, RoundOp);
template __global__ void elementwise_unary_kernel<double, RoundOp>(double*, const double*, size_t, RoundOp);

template __global__ void elementwise_unary_kernel<float, ErfOp>(float*, const float*, size_t, ErfOp);
template __global__ void elementwise_unary_kernel<double, ErfOp>(double*, const double*, size_t, ErfOp);

// Elementwise scalar kernels
template __global__ void elementwise_scalar_kernel<float, float, AddOp>(float*, const float*, float, size_t, AddOp);
template __global__ void elementwise_scalar_kernel<double, double, AddOp>(double*, const double*, double, size_t, AddOp);
template __global__ void elementwise_scalar_kernel<half, float, AddOp>(half*, const half*, float, size_t, AddOp);

template __global__ void elementwise_scalar_kernel<float, float, MulOp>(float*, const float*, float, size_t, MulOp);
template __global__ void elementwise_scalar_kernel<double, double, MulOp>(double*, const double*, double, size_t, MulOp);
template __global__ void elementwise_scalar_kernel<half, float, MulOp>(half*, const half*, float, size_t, MulOp);

// New scalar operations for Phase 3.1
template __global__ void elementwise_scalar_kernel<float, float, PowScalarOp>(float*, const float*, float, size_t, PowScalarOp);
template __global__ void elementwise_scalar_kernel<double, double, PowScalarOp>(double*, const double*, double, size_t, PowScalarOp);

// Other template instantiations for completeness
template __global__ void fill_kernel<float>(float*, float, size_t);
template __global__ void fill_kernel<double>(double*, double, size_t);
template __global__ void fill_kernel<half>(half*, half, size_t);

// Reduction kernels used in variance and other computations
template __global__ void reduction_kernel<float, float, AddOp>(float*, const float*, size_t, AddOp, float);
template __global__ void reduction_kernel<double, double, AddOp>(double*, const double*, size_t, AddOp, double);
template __global__ void reduction_kernel<float, float, MulOp>(float*, const float*, size_t, MulOp, float);
template __global__ void reduction_kernel<double, double, MulOp>(double*, const double*, size_t, MulOp, double);
template __global__ void reduction_kernel<float, float, MaxOp>(float*, const float*, size_t, MaxOp, float);
template __global__ void reduction_kernel<double, double, MaxOp>(double*, const double*, size_t, MaxOp, double);
template __global__ void reduction_kernel<float, float, MinOp>(float*, const float*, size_t, MinOp, float);
template __global__ void reduction_kernel<double, double, MinOp>(double*, const double*, size_t, MinOp, double);

// Matrix multiplication kernels
template __global__ void matmul_kernel<float>(float*, const float*, const float*, size_t, size_t, size_t);
template __global__ void matmul_kernel<double>(double*, const double*, const double*, size_t, size_t, size_t);
template __global__ void matmul_kernel<half>(half*, const half*, const half*, size_t, size_t, size_t);

template __global__ void matmul_tiled_kernel<float>(float*, const float*, const float*, size_t, size_t, size_t);
template __global__ void matmul_tiled_kernel<double>(double*, const double*, const double*, size_t, size_t, size_t);
template __global__ void matmul_tiled_kernel<half>(half*, const half*, const half*, size_t, size_t, size_t);

// Linear algebra kernels
template __global__ void outer_product_kernel<float>(float*, const float*, const float*, size_t, size_t);
template __global__ void outer_product_kernel<double>(double*, const double*, const double*, size_t, size_t);
template __global__ void outer_product_kernel<half>(half*, const half*, const half*, size_t, size_t);

template __global__ void matvec_kernel<float>(float*, const float*, const float*, size_t, size_t);
template __global__ void matvec_kernel<double>(double*, const double*, const double*, size_t, size_t);
template __global__ void matvec_kernel<half>(half*, const half*, const half*, size_t, size_t);

template __global__ void vecmat_kernel<float>(float*, const float*, const float*, size_t, size_t);
template __global__ void vecmat_kernel<double>(double*, const double*, const double*, size_t, size_t);
template __global__ void vecmat_kernel<half>(half*, const half*, const half*, size_t, size_t);

// Broadcast and advanced kernels
template __global__ void broadcast_binary_kernel<float, AddOp>(float*, const float*, const float*, const int*, const int*, const int*, const int*, int, size_t);
template __global__ void broadcast_binary_kernel<double, AddOp>(double*, const double*, const double*, const int*, const int*, const int*, const int*, int, size_t);
template __global__ void broadcast_binary_kernel<float, MulOp>(float*, const float*, const float*, const int*, const int*, const int*, const int*, int, size_t);
template __global__ void broadcast_binary_kernel<double, MulOp>(double*, const double*, const double*, const int*, const int*, const int*, const int*, int, size_t);

template __global__ void strided_copy_kernel<float>(float*, const float*, const int*, const int*, int, size_t);
template __global__ void strided_copy_kernel<double>(double*, const double*, const int*, const int*, int, size_t);
template __global__ void strided_copy_kernel<half>(half*, const half*, const int*, const int*, int, size_t);

// Type conversion kernels
template __global__ void type_conversion_kernel<float, half>(float*, const half*, size_t);
template __global__ void type_conversion_kernel<half, float>(half*, const float*, size_t);
template __global__ void type_conversion_kernel<double, float>(double*, const float*, size_t);
template __global__ void type_conversion_kernel<float, double>(float*, const double*, size_t); 

// Advanced tensor operation kernels (missing instantiations)
template __global__ void stack_kernel<float>(float*, float**, int, const int*, const int*, int, int, size_t);
template __global__ void stack_kernel<double>(double*, double**, int, const int*, const int*, int, int, size_t);
template __global__ void stack_kernel<half>(half*, half**, int, const int*, const int*, int, int, size_t);

template __global__ void concat_kernel<float>(float*, float**, const int*, int, const int*, const int*, const int*, int, int, size_t);
template __global__ void concat_kernel<double>(double*, double**, const int*, int, const int*, const int*, const int*, int, int, size_t);
template __global__ void concat_kernel<half>(half*, half**, const int*, int, const int*, const int*, const int*, int, int, size_t);

template __global__ void repeat_kernel<float>(float*, const float*, const int*, const int*, const int*, const int*, int, size_t);
template __global__ void repeat_kernel<double>(double*, const double*, const int*, const int*, const int*, const int*, int, size_t);
template __global__ void repeat_kernel<half>(half*, const half*, const int*, const int*, const int*, const int*, int, size_t);

template __global__ void pad_kernel<float>(float*, const float*, const int*, const int*, const int*, const int*, const int*, int, float, int, size_t);
template __global__ void pad_kernel<double>(double*, const double*, const int*, const int*, const int*, const int*, const int*, int, double, int, size_t);
template __global__ void pad_kernel<half>(half*, const half*, const int*, const int*, const int*, const int*, const int*, int, half, int, size_t);

// NEW: Axis-aware argmax/argmin kernel instantiations
template __global__ void axis_arg_extreme_kernel<float, float>(
    int64_t*, float*, const float*, const int*, const int*, const int*, 
    const int*, int, int, size_t, int);

template __global__ void axis_arg_extreme_kernel<double, double>(
    int64_t*, double*, const double*, const int*, const int*, const int*, 
    const int*, int, int, size_t, int);

// Axis-aware reduction kernels (explicit instantiations)
template __global__ void axis_reduction_kernel<float, float>(float*, const float*, const int*, const int*, const int*, const int*, int, int, size_t, int);
template __global__ void axis_reduction_kernel<double, double>(double*, const double*, const int*, const int*, const int*, const int*, int, int, size_t, int);

template __global__ void axis_variance_kernel<float>(float*, const float*, const int*, const int*, const int*, const int*, int, int, size_t);
template __global__ void axis_variance_kernel<double>(double*, const double*, const int*, const int*, const int*, const int*, int, int, size_t); 

// Argmax/Argmin kernels
template __global__ void argmax_kernel<float>(int64_t*, const float*, size_t);
template __global__ void argmax_kernel<double>(int64_t*, const double*, size_t);

template __global__ void argmin_kernel<float>(int64_t*, const float*, size_t);
template __global__ void argmin_kernel<double>(int64_t*, const double*, size_t);

// Helper kernel instantiations
template __global__ void copy_values_at_indices_kernel<float>(float*, const float*, const int64_t*, int, float);
template __global__ void copy_values_at_indices_kernel<double>(double*, const double*, const int64_t*, int, double); 

 