// tensor_wrappers.cu
// C-style wrapper functions for R interface

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "tensor_ops.cu"

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

// Unary math operations
void tensor_exp_float32(float* result, const float* input, size_t n) {
    launch_elementwise_unary(result, input, n, ExpOp{});
    cudaDeviceSynchronize();
}

void tensor_exp_float64(double* result, const double* input, size_t n) {
    launch_elementwise_unary(result, input, n, ExpOp{});
    cudaDeviceSynchronize();
}

void tensor_log_float32(float* result, const float* input, size_t n) {
    launch_elementwise_unary(result, input, n, LogOp{});
    cudaDeviceSynchronize();
}

void tensor_log_float64(double* result, const double* input, size_t n) {
    launch_elementwise_unary(result, input, n, LogOp{});
    cudaDeviceSynchronize();
}

void tensor_sqrt_float32(float* result, const float* input, size_t n) {
    launch_elementwise_unary(result, input, n, SqrtOp{});
    cudaDeviceSynchronize();
}

void tensor_sqrt_float64(double* result, const double* input, size_t n) {
    launch_elementwise_unary(result, input, n, SqrtOp{});
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

} // extern "C" 