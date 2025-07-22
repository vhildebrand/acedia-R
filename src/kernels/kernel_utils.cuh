#ifndef KERNEL_UTILS_CUH
#define KERNEL_UTILS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

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

#endif // KERNEL_UTILS_CUH 