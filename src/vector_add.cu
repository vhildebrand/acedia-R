// vector_add.cu

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Kernel: executed in multiple threads in parallel
__global__ void addKernel(double *c, const double *a, const double *b, int n) {
    // global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread handles one elem of vector
    // check bounds of n to have smaller number of threads than elements
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}



// Host function with C-style linkage to other C/C++ code
extern "C" void addVectorsOnGpu(double *h_c, const double *h_a, const double *h_b, int n) {
    double *d_a, *d_b, *d_c;
    size_t size = n * sizeof(double);

    // allocate memory on GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 256;
    // calculate blocks needed in grid
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, n);

    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}