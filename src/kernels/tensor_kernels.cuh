#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Operation functors - shared between tensor_kernels.cu and tensor_ops.cu

// Binary operation functors
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

// Comparison functors
struct GreaterOp {
    template<typename T>
    __device__ T operator()(const T& a, const T& b) const { return (a > b) ? T(1) : T(0); }
};

struct LessOp {
    template<typename T>
    __device__ T operator()(const T& a, const T& b) const { return (a < b) ? T(1) : T(0); }
};

struct EqualOp {
    template<typename T>
    __device__ T operator()(const T& a, const T& b) const { return (a == b) ? T(1) : T(0); }
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

struct TanhOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return tanh(a); }
};

struct SigmoidOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return T(1) / (T(1) + exp(-a)); }
};

struct ReluOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return fmax(a, T(0)); }
};

struct LeakyReluOp {
    template<typename T>
    __device__ T operator()(const T& a, const T& alpha = T(0.01)) const { 
        return a > T(0) ? a : alpha * a; 
    }
};

struct SinOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return sin(a); }
};

struct CosOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return cos(a); }
};

struct AbsOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return fabs(a); }
};

struct SquareOp {
    template<typename T>
    __device__ T operator()(const T& a) const { return a * a; }
};

struct FloorOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a) const { 
#ifdef __CUDA_ARCH__
        return floor(a); // Device version
#else
        return std::floor(a); // Host version
#endif
    }
};

struct CeilOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a) const { 
#ifdef __CUDA_ARCH__
        return ceil(a); // Device version
#else
        return std::ceil(a); // Host version
#endif
    }
};

struct RoundOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a) const { 
#ifdef __CUDA_ARCH__
        // Use rint() which implements banker's rounding (round half to even)
        return rint(a);
#else
        return std::round(a); 
#endif
    }
};

struct PowScalarOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& exponent) const { 
#ifdef __CUDA_ARCH__
        return pow(a, exponent); // Device version
#else
        return std::pow(a, exponent); // Host version
#endif
    }
};

struct ErfOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a) const { 
#ifdef __CUDA_ARCH__
        return erf(a); // Device version
#else
        return std::erf(a); // Host version
#endif
    }
};

// Binary PowOp for element-wise power operations
struct PowOp {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b) const { 
#ifdef __CUDA_ARCH__
        return pow(a, b); // Device version
#else
        return std::pow(a, b); // Host version
#endif
    }
};

// Template kernel declarations from tensor_kernels.cu

// Basic kernels
template<typename T>
__global__ void fill_kernel(T* data, T value, size_t n);

template<typename T, typename Op>
__global__ void elementwise_binary_kernel(T* result, const T* a, const T* b, size_t n, Op op);

template<typename T, typename U, typename Op>
__global__ void elementwise_scalar_kernel(T* result, const T* input, U scalar, size_t n, Op op);

template<typename T, typename Op>
__global__ void elementwise_unary_kernel(T* result, const T* input, size_t n, Op op);

template<typename To, typename From>
__global__ void type_conversion_kernel(To* output, const From* input, size_t n);

template<typename T, typename AccumType, typename Op>
__global__ void reduction_kernel(AccumType* result, const T* input, size_t n, Op op, AccumType init_val);

// Linear algebra kernels
template<typename T>
__global__ void matmul_kernel(T* C, const T* A, const T* B, size_t M, size_t N, size_t K);

template<typename T>
__global__ void matmul_tiled_kernel(T* C, const T* A, const T* B, size_t M, size_t N, size_t K);

template<typename T>
__global__ void outer_product_kernel(T* result, const T* a, const T* b, size_t M, size_t N);

template<typename T>
__global__ void matvec_kernel(T* result, const T* A, const T* v, size_t M, size_t N);

template<typename T>
__global__ void matvec_optimized_kernel(T* result, const T* A, const T* v, size_t M, size_t N);

template<typename T>
__global__ void vecmat_kernel(T* result, const T* v, const T* A, size_t M, size_t N);

// Advanced kernels
template<typename T, typename Op>
__global__ void broadcast_binary_kernel(
    T* result, 
    const T* a, const T* b,
    const int* a_strides, const int* b_strides, const int* result_strides,
    const int* shape, int ndims, size_t total_elements
);

template<typename T>
__global__ void strided_copy_kernel(
    T* dest, const T* src,
    const int* src_strides, const int* shape, int ndims, size_t total_elements
);

template<typename T>
__global__ void concat_kernel(
    T* result, T** inputs, const int* input_sizes, int num_tensors,
    const int* result_strides, const int* input_strides_list, 
    const int* shape, int ndims, int concat_axis, size_t total_elements
);

template<typename T>
__global__ void stack_kernel(
    T* result, T** inputs, int num_tensors,
    const int* input_strides, const int* result_shape, 
    int ndims, int stack_axis, size_t total_elements
);

template<typename T>
__global__ void repeat_kernel(
    T* result, const T* input,
    const int* input_strides, const int* repeat_counts,
    const int* input_shape, const int* result_shape,
    int ndims, size_t total_elements
);

template<typename T>
__global__ void pad_kernel(
    T* result, const T* input,
    const int* input_strides, const int* input_shape,
    const int* pad_before, const int* pad_after,
    const int* result_shape, int ndims, 
    T pad_value, int pad_mode, size_t total_elements
);

// Axis-aware reduction kernels
template<typename T, typename AccumType>
__global__ void axis_reduction_kernel(
    AccumType* result, const T* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size, int operation
);

template<typename T>
__global__ void axis_variance_kernel(
    T* result, const T* input,
    const int* input_strides, const int* input_shape,
    const int* result_strides, const int* reduction_axes,
    int num_reduction_axes, int ndims, size_t output_size
); 