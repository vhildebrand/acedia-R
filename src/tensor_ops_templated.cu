// tensor_ops_templated.cu
// Fully templated CUDA kernels for all tensor operations supporting mixed precision

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <algorithm>
#include <type_traits>
#include "gpuTensor.h"

// Utility templates for type conversions
template<typename To, typename From>
__device__ __forceinline__ To convert_type(From val) {
    return static_cast<To>(val);
}

// Specializations for half precision types
template<>
__device__ __forceinline__ half convert_type<half, float>(float val) {
    return __float2half(val);
}

template<>
__device__ __forceinline__ float convert_type<float, half>(half val) {
    return __half2float(val);
}

template<>
__device__ __forceinline__ half convert_type<half, double>(double val) {
    return __float2half(static_cast<float>(val));
}

template<>
__device__ __forceinline__ double convert_type<double, half>(half val) {
    return static_cast<double>(__half2float(val));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// BFloat16 conversions (available on Ampere and later)
template<>
__device__ __forceinline__ __nv_bfloat16 convert_type<__nv_bfloat16, float>(float val) {
    return __float2bfloat16(val);
}

template<>
__device__ __forceinline__ float convert_type<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}
#endif

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

// C-style wrapper functions for each type (for R interface)
extern "C" {

// Fill operations
void tensor_fill_float16(half* data, float value, size_t n) {
    launch_fill(data, __float2half(value), n);
    cudaDeviceSynchronize();
}

void tensor_fill_float32(float* data, float value, size_t n) {
    launch_fill(data, value, n);
    cudaDeviceSynchronize();
}

void tensor_fill_float64(double* data, double value, size_t n) {
    launch_fill(data, value, n);
    cudaDeviceSynchronize();
}

void tensor_fill_int8(int8_t* data, int value, size_t n) {
    launch_fill(data, static_cast<int8_t>(value), n);
    cudaDeviceSynchronize();
}

void tensor_fill_int32(int32_t* data, int value, size_t n) {
    launch_fill(data, value, n);
    cudaDeviceSynchronize();
}

void tensor_fill_int64(int64_t* data, long long value, size_t n) {
    launch_fill(data, static_cast<int64_t>(value), n);
    cudaDeviceSynchronize();
}

// Addition operations
void tensor_add_float16(half* result, const half* a, const half* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, AddOp{});
    cudaDeviceSynchronize();
}

void tensor_add_float32(float* result, const float* a, const float* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, AddOp{});
    cudaDeviceSynchronize();
}

void tensor_add_float64(double* result, const double* a, const double* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, AddOp{});
    cudaDeviceSynchronize();
}

void tensor_add_int8(int8_t* result, const int8_t* a, const int8_t* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, AddOp{});
    cudaDeviceSynchronize();
}

void tensor_add_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, AddOp{});
    cudaDeviceSynchronize();
}

void tensor_add_int64(int64_t* result, const int64_t* a, const int64_t* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, AddOp{});
    cudaDeviceSynchronize();
}

// Multiplication operations
void tensor_mul_float16(half* result, const half* a, const half* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, MulOp{});
    cudaDeviceSynchronize();
}

void tensor_mul_float32(float* result, const float* a, const float* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, MulOp{});
    cudaDeviceSynchronize();
}

void tensor_mul_float64(double* result, const double* a, const double* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, MulOp{});
    cudaDeviceSynchronize();
}

// Scalar multiplication
void tensor_scalar_mul_float16(half* result, const half* input, float scalar, size_t n) {
    launch_elementwise_scalar(result, input, scalar, n, MulOp{});
    cudaDeviceSynchronize();
}

void tensor_scalar_mul_float32(float* result, const float* input, float scalar, size_t n) {
    launch_elementwise_scalar(result, input, scalar, n, MulOp{});
    cudaDeviceSynchronize();
}

void tensor_scalar_mul_float64(double* result, const double* input, double scalar, size_t n) {
    launch_elementwise_scalar(result, input, scalar, n, MulOp{});
    cudaDeviceSynchronize();
}

// Matrix multiplication
void tensor_matmul_float16(half* C, const half* A, const half* B, size_t M, size_t N, size_t K) {
    launch_matmul(C, A, B, M, N, K);
    cudaDeviceSynchronize();
}

void tensor_matmul_float32(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
    launch_matmul(C, A, B, M, N, K);
    cudaDeviceSynchronize();
}

void tensor_matmul_float64(double* C, const double* A, const double* B, size_t M, size_t N, size_t K) {
    launch_matmul(C, A, B, M, N, K);
    cudaDeviceSynchronize();
}

// Sum reductions
float tensor_sum_float16(const half* input, size_t n) {
    return launch_reduction(input, n, AddOp{}, 0.0f);
}

float tensor_sum_float32(const float* input, size_t n) {
    return launch_reduction(input, n, AddOp{}, 0.0f);
}

double tensor_sum_float64(const double* input, size_t n) {
    return launch_reduction(input, n, AddOp{}, 0.0);
}

long long tensor_sum_int64(const int64_t* input, size_t n) {
    return launch_reduction(input, n, AddOp{}, static_cast<int64_t>(0));
}

// Type conversion functions
void convert_float32_to_float16(half* output, const float* input, size_t n) {
    launch_type_conversion(output, input, n);
    cudaDeviceSynchronize();
}

void convert_float16_to_float32(float* output, const half* input, size_t n) {
    launch_type_conversion(output, input, n);
    cudaDeviceSynchronize();
}

void convert_float64_to_float32(float* output, const double* input, size_t n) {
    launch_type_conversion(output, input, n);
    cudaDeviceSynchronize();
}

void convert_float32_to_float64(double* output, const float* input, size_t n) {
    launch_type_conversion(output, input, n);
    cudaDeviceSynchronize();
}

void convert_int32_to_float32(float* output, const int32_t* input, size_t n) {
    launch_type_conversion(output, input, n);
    cudaDeviceSynchronize();
}

void convert_float32_to_int32(int32_t* output, const float* input, size_t n) {
    launch_type_conversion(output, input, n);
    cudaDeviceSynchronize();
}

// Mixed precision training utilities
void convert_gradients_fp32_to_fp16(half* fp16_grads, const float* fp32_grads, size_t n) {
    convert_float32_to_float16(fp16_grads, fp32_grads, n);
}

void accumulate_gradients_fp16_to_fp32(float* fp32_grads, const half* fp16_grads, size_t n) {
    // Convert to fp32 and accumulate
    float* temp_fp32;
    cudaMalloc(&temp_fp32, n * sizeof(float));
    convert_float16_to_float32(temp_fp32, fp16_grads, n);
    launch_elementwise_binary(fp32_grads, fp32_grads, temp_fp32, n, AddOp{});
    cudaFree(temp_fp32);
    cudaDeviceSynchronize();
}

} // extern "C" 