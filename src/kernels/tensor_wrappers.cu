// tensor_wrappers.cu
// C-style wrapper functions for R interface

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include "kernel_utils.cuh"
#include "tensor_ops.cu"
#include "../gpuTensor.h"

// Simple error checking function
inline void cudaSafeCall(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "%s: %s", msg, cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
}

// C-style wrapper functions for each type (for R interface)
extern "C" {

// Broadcast addition wrappers
void tensor_add_broadcast_float32(
    float* result, const float* a, const float* b,
    const int* a_strides, const int* b_strides, const int* result_strides,
    const int* shape, int ndims, size_t total_elements
) {
    launch_broadcast_binary(result, a, b, a_strides, b_strides, result_strides, shape, ndims, total_elements, AddOp{});
    cudaDeviceSynchronize();
}

void tensor_add_broadcast_float64(
    double* result, const double* a, const double* b,
    const int* a_strides, const int* b_strides, const int* result_strides,
    const int* shape, int ndims, size_t total_elements
) {
    launch_broadcast_binary(result, a, b, a_strides, b_strides, result_strides, shape, ndims, total_elements, AddOp{});
    cudaDeviceSynchronize();
}

// Broadcast multiplication wrappers
void tensor_mul_broadcast_float32(
    float* result, const float* a, const float* b,
    const int* a_strides, const int* b_strides, const int* result_strides,
    const int* shape, int ndims, size_t total_elements
) {
    launch_broadcast_binary(result, a, b, a_strides, b_strides, result_strides, shape, ndims, total_elements, MulOp{});
    cudaDeviceSynchronize();
}

void tensor_mul_broadcast_float64(
    double* result, const double* a, const double* b,
    const int* a_strides, const int* b_strides, const int* result_strides,
    const int* shape, int ndims, size_t total_elements
) {
    launch_broadcast_binary(result, a, b, a_strides, b_strides, result_strides, shape, ndims, total_elements, MulOp{});
    cudaDeviceSynchronize();
}

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

// Subtraction operations
void tensor_sub_float16(half* result, const half* a, const half* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, SubOp{});
    cudaDeviceSynchronize();
}

void tensor_sub_float32(float* result, const float* a, const float* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, SubOp{});
    cudaDeviceSynchronize();
}

void tensor_sub_float64(double* result, const double* a, const double* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, SubOp{});
    cudaDeviceSynchronize();
}

// Division operations
void tensor_div_float16(half* result, const half* a, const half* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, DivOp{});
    cudaDeviceSynchronize();
}

void tensor_div_float32(float* result, const float* a, const float* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, DivOp{});
    cudaDeviceSynchronize();
}

void tensor_div_float64(double* result, const double* a, const double* b, size_t n) {
    launch_elementwise_binary(result, a, b, n, DivOp{});
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

// Scalar addition
void tensor_scalar_add_float16(half* result, const half* input, float scalar, size_t n) {
    launch_elementwise_scalar(result, input, scalar, n, AddOp{});
    cudaDeviceSynchronize();
}

void tensor_scalar_add_float32(float* result, const float* input, float scalar, size_t n) {
    launch_elementwise_scalar(result, input, scalar, n, AddOp{});
    cudaDeviceSynchronize();
}

void tensor_scalar_add_float64(double* result, const double* input, double scalar, size_t n) {
    launch_elementwise_scalar(result, input, scalar, n, AddOp{});
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

// Advanced tensor operations

// Concat operation
void tensor_concat_float32(float* result, const float** inputs, const int* input_sizes, int num_tensors,
                          const int* result_strides, const int* input_strides_list, const int* shape, 
                          int ndims, int concat_axis, size_t total_elements) {
    // Convert to std::vector for launch helper
    std::vector<int> result_strides_vec(result_strides, result_strides + ndims);
    std::vector<int> shape_vec(shape, shape + ndims);
    
    std::vector<std::vector<int>> input_strides_vec;
    for (int i = 0; i < num_tensors; i++) {
        std::vector<int> strides(input_strides_list + i * ndims, input_strides_list + (i+1) * ndims);
        input_strides_vec.push_back(strides);
    }
    
    launch_concat(result, inputs, input_sizes, num_tensors, result_strides_vec, input_strides_vec, shape_vec, concat_axis, total_elements);
    cudaDeviceSynchronize();
}

void tensor_concat_float64(double* result, const double** inputs, const int* input_sizes, int num_tensors,
                          const int* result_strides, const int* input_strides_list, const int* shape, 
                          int ndims, int concat_axis, size_t total_elements) {
    std::vector<int> result_strides_vec(result_strides, result_strides + ndims);
    std::vector<int> shape_vec(shape, shape + ndims);
    
    std::vector<std::vector<int>> input_strides_vec;
    for (int i = 0; i < num_tensors; i++) {
        std::vector<int> strides(input_strides_list + i * ndims, input_strides_list + (i+1) * ndims);
        input_strides_vec.push_back(strides);
    }
    
    launch_concat(result, inputs, input_sizes, num_tensors, result_strides_vec, input_strides_vec, shape_vec, concat_axis, total_elements);
    cudaDeviceSynchronize();
}

// Stack operation
void tensor_stack_float32(float* result, const float** inputs, int num_tensors,
                         const int* input_strides, const int* result_shape, int ndims, int stack_axis, size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims - 1);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    
    launch_stack(result, inputs, num_tensors, input_strides_vec, result_shape_vec, stack_axis, total_elements);
    cudaDeviceSynchronize();
}

void tensor_stack_float64(double* result, const double** inputs, int num_tensors,
                         const int* input_strides, const int* result_shape, int ndims, int stack_axis, size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims - 1);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    
    launch_stack(result, inputs, num_tensors, input_strides_vec, result_shape_vec, stack_axis, total_elements);
    cudaDeviceSynchronize();
}

// Repeat operation
void tensor_repeat_float32(float* result, const float* input, const int* input_strides, const int* repeat_counts,
                          const int* input_shape, const int* result_shape, int ndims, size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims);
    std::vector<int> repeat_counts_vec(repeat_counts, repeat_counts + ndims);
    std::vector<int> input_shape_vec(input_shape, input_shape + ndims);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    
    launch_repeat(result, input, input_strides_vec, repeat_counts_vec, input_shape_vec, result_shape_vec, total_elements);
    cudaDeviceSynchronize();
}

void tensor_repeat_float64(double* result, const double* input, const int* input_strides, const int* repeat_counts,
                          const int* input_shape, const int* result_shape, int ndims, size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims);
    std::vector<int> repeat_counts_vec(repeat_counts, repeat_counts + ndims);
    std::vector<int> input_shape_vec(input_shape, input_shape + ndims);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    
    launch_repeat(result, input, input_strides_vec, repeat_counts_vec, input_shape_vec, result_shape_vec, total_elements);
    cudaDeviceSynchronize();
}

// Pad operation
void tensor_pad_float32(float* result, const float* input, const int* input_strides, const int* input_shape,
                       const int* pad_before, const int* pad_after, const int* result_shape, 
                       int ndims, float pad_value, int pad_mode, size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims);
    std::vector<int> input_shape_vec(input_shape, input_shape + ndims);
    std::vector<int> pad_before_vec(pad_before, pad_before + ndims);
    std::vector<int> pad_after_vec(pad_after, pad_after + ndims);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    
    launch_pad(result, input, input_strides_vec, input_shape_vec, pad_before_vec, pad_after_vec, result_shape_vec, pad_value, pad_mode, total_elements);
    cudaDeviceSynchronize();
}

void tensor_pad_float64(double* result, const double* input, const int* input_strides, const int* input_shape,
                       const int* pad_before, const int* pad_after, const int* result_shape, 
                       int ndims, double pad_value, int pad_mode, size_t total_elements) {
    std::vector<int> input_strides_vec(input_strides, input_strides + ndims);
    std::vector<int> input_shape_vec(input_shape, input_shape + ndims);
    std::vector<int> pad_before_vec(pad_before, pad_before + ndims);
    std::vector<int> pad_after_vec(pad_after, pad_after + ndims);
    std::vector<int> result_shape_vec(result_shape, result_shape + ndims);
    
    launch_pad(result, input, input_strides_vec, input_shape_vec, pad_before_vec, pad_after_vec, result_shape_vec, pad_value, pad_mode, total_elements);
    cudaDeviceSynchronize();
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

// Strided copy operations
void tensor_strided_copy_float32(float* dest, const float* src, const int* strides, const int* shape, int ndims, size_t total_elements) {
    std::vector<int> stride_vec(strides, strides + ndims);
    std::vector<int> shape_vec(shape, shape + ndims);
    launch_strided_copy(dest, src, stride_vec, shape_vec, total_elements);
    cudaDeviceSynchronize();
}

void tensor_strided_copy_float64(double* dest, const double* src, const int* strides, const int* shape, int ndims, size_t total_elements) {
    std::vector<int> stride_vec(strides, strides + ndims);
    std::vector<int> shape_vec(shape, shape + ndims);
    launch_strided_copy(dest, src, stride_vec, shape_vec, total_elements);
    cudaDeviceSynchronize();
}

// Unary Math Operations
void tensor_exp_float32(float* result, const float* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_exp_float32");
    
    launch_elementwise_unary(result, input, n, ExpOp());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_exp_float32");
}

void tensor_exp_float64(double* result, const double* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_exp_float64");
    
    launch_elementwise_unary(result, input, n, ExpOp());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_exp_float64");
}

void tensor_log_float32(float* result, const float* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_log_float32");
    
    launch_elementwise_unary(result, input, n, LogOp());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_log_float32");
}

void tensor_log_float64(double* result, const double* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_log_float64");
    
    launch_elementwise_unary(result, input, n, LogOp());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_log_float64");
}

void tensor_sqrt_float32(float* result, const float* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_sqrt_float32");
    
    launch_elementwise_unary(result, input, n, SqrtOp());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_sqrt_float32");
}

void tensor_sqrt_float64(double* result, const double* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_sqrt_float64");
    
    launch_elementwise_unary(result, input, n, SqrtOp());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_sqrt_float64");
}

// Reduction operations
float tensor_max_float32(const float* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_max_float32");
    
    float result = launch_reduction(input, n, MaxOp(), 
                                   std::numeric_limits<float>::lowest());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_max_float32");
    return result;
}

double tensor_max_float64(const double* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_max_float64");
    
    double result = launch_reduction(input, n, MaxOp(), 
                                    std::numeric_limits<double>::lowest());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_max_float64");
    return result;
}

float tensor_min_float32(const float* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_min_float32");
    
    float result = launch_reduction(input, n, MinOp(), 
                                   std::numeric_limits<float>::max());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_min_float32");
    return result;
}

double tensor_min_float64(const double* input, size_t n) {
    cudaError_t err = cudaGetLastError();
    cudaSafeCall(err, "CUDA error before tensor_min_float64");
    
    double result = launch_reduction(input, n, MinOp(), 
                                    std::numeric_limits<double>::max());
    
    err = cudaDeviceSynchronize();
    cudaSafeCall(err, "cudaDeviceSynchronize failed in tensor_min_float64");
    return result;
}

} // extern "C" 