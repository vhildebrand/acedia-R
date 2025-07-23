// tensor_slice_update.cu
// CUDA kernels for in-place slice updates on tensors

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernel_utils.cuh"

// ----------------------------------------------------------------------------
// Kernel: add scalar to a rectangular slice (n <= 8 dims)
// ----------------------------------------------------------------------------

template <typename T>
__global__ void add_scalar_slice_kernel(
    T* data,
    double scalar,                 // scalar (converted in-kernel)
    const int* start,              // start indices for each dim (0-based)
    const int* slice_shape,        // extents for each dim
    const int* strides,            // strides of parent tensor
    int ndims,                     // number of dimensions
    size_t total_elements)         // total elements in slice
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Convert linear index -> coordinates in slice (column-major as elsewhere)
    int coords[8] = {0};
    size_t tmp = idx;
    for (int d = 0; d < ndims; ++d) {
        coords[d] = tmp % slice_shape[d];
        tmp /= slice_shape[d];
    }

    // Map to parent offset
    int offset = 0;
    for (int d = 0; d < ndims; ++d) {
        offset += (start[d] + coords[d]) * strides[d];
    }

    data[offset] += static_cast<T>(scalar);
}

// ----------------------------------------------------------------------------
// C-style launch helpers (extern "C") so we can call from C++/R
// ----------------------------------------------------------------------------
extern "C" {

void tensor_slice_add_scalar_float32(
    float* data,
    double scalar,
    const int* start,
    const int* slice_shape,
    const int* strides,
    int ndims,
    size_t total_elements)
{
    int threads = 256;
    int blocks  = static_cast<int>((total_elements + threads - 1) / threads);
    add_scalar_slice_kernel<float><<<blocks, threads>>>(
        data, scalar, start, slice_shape, strides, ndims, total_elements);
}

void tensor_slice_add_scalar_float64(
    double* data,
    double scalar,
    const int* start,
    const int* slice_shape,
    const int* strides,
    int ndims,
    size_t total_elements)
{
    int threads = 256;
    int blocks  = static_cast<int>((total_elements + threads - 1) / threads);
    add_scalar_slice_kernel<double><<<blocks, threads>>>(
        data, scalar, start, slice_shape, strides, ndims, total_elements);
}

} // extern "C" 