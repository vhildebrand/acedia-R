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
// Kernel: set scalar to a rectangular slice (general assignment)
// ----------------------------------------------------------------------------

template <typename T>
__global__ void set_scalar_slice_kernel(
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

    data[offset] = static_cast<T>(scalar);
}

// ----------------------------------------------------------------------------
// Kernel: set tensor values to a rectangular slice (tensor-to-slice assignment)
// ----------------------------------------------------------------------------

template <typename T>
__global__ void set_tensor_slice_kernel(
    T* dest_data,
    const T* src_data,
    const int* start,              // start indices for each dim (0-based)
    const int* slice_shape,        // extents for each dim
    const int* dest_strides,       // strides of destination tensor
    const int* src_strides,        // strides of source tensor
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

    // Map to destination offset (with start offset)
    int dest_offset = 0;
    for (int d = 0; d < ndims; ++d) {
        dest_offset += (start[d] + coords[d]) * dest_strides[d];
    }

    // Map to source offset (no start offset - source coordinates are direct)
    int src_offset = 0;
    for (int d = 0; d < ndims; ++d) {
        src_offset += coords[d] * src_strides[d];
    }

    dest_data[dest_offset] = src_data[src_offset];
}

// ----------------------------------------------------------------------------
// Kernel: boolean mask assignment
// ----------------------------------------------------------------------------

template <typename T>
__global__ void set_mask_kernel(
    T* data,
    const bool* mask,
    T value,
    size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    if (mask[idx]) {
        data[idx] = value;
    }
}

// ----------------------------------------------------------------------------
// C-style launch helpers (extern "C") so we can call from C++/R
// ----------------------------------------------------------------------------
extern "C" {

// Existing add scalar functions
static void launch_slice_add_scalar_float(float* data, double scalar,
        const int* h_start, const int* h_shape, const int* h_strides,
        int ndims, size_t total_elements) {
    int *d_start, *d_shape, *d_strides;
    size_t bytes = ndims * sizeof(int);
    cudaMalloc(&d_start, bytes);
    cudaMalloc(&d_shape, bytes);
    cudaMalloc(&d_strides, bytes);
    cudaMemcpy(d_start, h_start, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, h_shape, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, h_strides, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = static_cast<int>((total_elements + threads - 1) / threads);
    add_scalar_slice_kernel<float><<<blocks, threads>>>(
        data, scalar, d_start, d_shape, d_strides, ndims, total_elements);
    cudaDeviceSynchronize();

    cudaFree(d_start);
    cudaFree(d_shape);
    cudaFree(d_strides);
}

void tensor_slice_add_scalar_float32(
    float* data,
    double scalar,
    const int* start,
    const int* slice_shape,
    const int* strides,
    int ndims,
    size_t total_elements)
{
    launch_slice_add_scalar_float(data, scalar, start, slice_shape, strides,
                                   ndims, total_elements);
}

static void launch_slice_add_scalar_double(double* data, double scalar,
        const int* h_start, const int* h_shape, const int* h_strides,
        int ndims, size_t total_elements) {
    int *d_start, *d_shape, *d_strides;
    size_t bytes = ndims * sizeof(int);
    cudaMalloc(&d_start, bytes);
    cudaMalloc(&d_shape, bytes);
    cudaMalloc(&d_strides, bytes);
    cudaMemcpy(d_start, h_start, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, h_shape, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, h_strides, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = static_cast<int>((total_elements + threads - 1) / threads);
    add_scalar_slice_kernel<double><<<blocks, threads>>>(
        data, scalar, d_start, d_shape, d_strides, ndims, total_elements);
    cudaDeviceSynchronize();

    cudaFree(d_start);
    cudaFree(d_shape);
    cudaFree(d_strides);
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
    launch_slice_add_scalar_double(data, scalar, start, slice_shape, strides,
                                   ndims, total_elements);
}

// New set scalar functions
static void launch_slice_set_scalar_float(float* data, double scalar,
        const int* h_start, const int* h_shape, const int* h_strides,
        int ndims, size_t total_elements) {
    int *d_start, *d_shape, *d_strides;
    size_t bytes = ndims * sizeof(int);
    cudaMalloc(&d_start, bytes);
    cudaMalloc(&d_shape, bytes);
    cudaMalloc(&d_strides, bytes);
    cudaMemcpy(d_start, h_start, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, h_shape, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, h_strides, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = static_cast<int>((total_elements + threads - 1) / threads);
    set_scalar_slice_kernel<float><<<blocks, threads>>>(
        data, scalar, d_start, d_shape, d_strides, ndims, total_elements);
    cudaDeviceSynchronize();

    cudaFree(d_start);
    cudaFree(d_shape);
    cudaFree(d_strides);
}

void tensor_slice_set_scalar_float32(
    float* data,
    double scalar,
    const int* start,
    const int* slice_shape,
    const int* strides,
    int ndims,
    size_t total_elements)
{
    launch_slice_set_scalar_float(data, scalar, start, slice_shape, strides,
                                   ndims, total_elements);
}

static void launch_slice_set_scalar_double(double* data, double scalar,
        const int* h_start, const int* h_shape, const int* h_strides,
        int ndims, size_t total_elements) {
    int *d_start, *d_shape, *d_strides;
    size_t bytes = ndims * sizeof(int);
    cudaMalloc(&d_start, bytes);
    cudaMalloc(&d_shape, bytes);
    cudaMalloc(&d_strides, bytes);
    cudaMemcpy(d_start, h_start, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, h_shape, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, h_strides, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = static_cast<int>((total_elements + threads - 1) / threads);
    set_scalar_slice_kernel<double><<<blocks, threads>>>(
        data, scalar, d_start, d_shape, d_strides, ndims, total_elements);
    cudaDeviceSynchronize();

    cudaFree(d_start);
    cudaFree(d_shape);
    cudaFree(d_strides);
}

void tensor_slice_set_scalar_float64(
    double* data,
    double scalar,
    const int* start,
    const int* slice_shape,
    const int* strides,
    int ndims,
    size_t total_elements)
{
    launch_slice_set_scalar_double(data, scalar, start, slice_shape, strides,
                                   ndims, total_elements);
}

// New tensor-to-slice assignment functions
static void launch_slice_set_tensor_float(float* dest_data, const float* src_data,
        const int* h_start, const int* h_shape, 
        const int* h_dest_strides, const int* h_src_strides,
        int ndims, size_t total_elements) {
    int *d_start, *d_shape, *d_dest_strides, *d_src_strides;
    size_t bytes = ndims * sizeof(int);
    cudaMalloc(&d_start, bytes);
    cudaMalloc(&d_shape, bytes);
    cudaMalloc(&d_dest_strides, bytes);
    cudaMalloc(&d_src_strides, bytes);
    cudaMemcpy(d_start, h_start, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, h_shape, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest_strides, h_dest_strides, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_strides, h_src_strides, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = static_cast<int>((total_elements + threads - 1) / threads);
    set_tensor_slice_kernel<float><<<blocks, threads>>>(
        dest_data, src_data, d_start, d_shape, d_dest_strides, d_src_strides, ndims, total_elements);
    cudaDeviceSynchronize();

    cudaFree(d_start);
    cudaFree(d_shape);
    cudaFree(d_dest_strides);
    cudaFree(d_src_strides);
}

void tensor_slice_set_tensor_float32(
    float* dest_data,
    const float* src_data,
    const int* start,
    const int* slice_shape,
    const int* dest_strides,
    const int* src_strides,
    int ndims,
    size_t total_elements)
{
    launch_slice_set_tensor_float(dest_data, src_data, start, slice_shape, 
                                  dest_strides, src_strides, ndims, total_elements);
}

static void launch_slice_set_tensor_double(double* dest_data, const double* src_data,
        const int* h_start, const int* h_shape, 
        const int* h_dest_strides, const int* h_src_strides,
        int ndims, size_t total_elements) {
    int *d_start, *d_shape, *d_dest_strides, *d_src_strides;
    size_t bytes = ndims * sizeof(int);
    cudaMalloc(&d_start, bytes);
    cudaMalloc(&d_shape, bytes);
    cudaMalloc(&d_dest_strides, bytes);
    cudaMalloc(&d_src_strides, bytes);
    cudaMemcpy(d_start, h_start, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, h_shape, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest_strides, h_dest_strides, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_strides, h_src_strides, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = static_cast<int>((total_elements + threads - 1) / threads);
    set_tensor_slice_kernel<double><<<blocks, threads>>>(
        dest_data, src_data, d_start, d_shape, d_dest_strides, d_src_strides, ndims, total_elements);
    cudaDeviceSynchronize();

    cudaFree(d_start);
    cudaFree(d_shape);
    cudaFree(d_dest_strides);
    cudaFree(d_src_strides);
}

void tensor_slice_set_tensor_float64(
    double* dest_data,
    const double* src_data,
    const int* start,
    const int* slice_shape,
    const int* dest_strides,
    const int* src_strides,
    int ndims,
    size_t total_elements)
{
    launch_slice_set_tensor_double(dest_data, src_data, start, slice_shape, 
                                   dest_strides, src_strides, ndims, total_elements);
}

// Boolean mask assignment functions
void tensor_mask_set_scalar_float32(
    float* data,
    const bool* mask,
    float value,
    size_t total_elements)
{
    int threads = 256;
    int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    set_mask_kernel<float><<<blocks, threads>>>(data, mask, value, total_elements);
    cudaDeviceSynchronize();
}

void tensor_mask_set_scalar_float64(
    double* data,
    const bool* mask,
    double value,
    size_t total_elements)
{
    int threads = 256;
    int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    set_mask_kernel<double><<<blocks, threads>>>(data, mask, value, total_elements);
    cudaDeviceSynchronize();
}

} // extern "C" 