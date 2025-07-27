#ifndef CUSOLVER_UTILS_H
#define CUSOLVER_UTILS_H

#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cusolver_utils {

// Error checking function for cuSOLVER calls
inline void cusolver_check(cusolverStatus_t status, const char* call) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER error: " + std::string(call) + " failed with status " + std::to_string(status));
    }
}

// Lazy global cuSOLVER handle (thread-safe since C++11 static init)
inline cusolverDnHandle_t get_cusolver_handle() {
    static cusolverDnHandle_t handle = nullptr;
    static bool initialized = false;
    if (!initialized) {
        cusolverStatus_t status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSOLVER handle");
        }
        initialized = true;
    }
    return handle;
}

// Convenience: set stream for the global handle and return it
inline cusolverDnHandle_t get_cusolver_handle(cudaStream_t stream) {
    auto handle = get_cusolver_handle();
    cusolverStatus_t status = cusolverDnSetStream(handle, stream);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to set cuSOLVER stream");
    }
    return handle;
}

// Helper to get workspace size for different operations
template<typename T>
struct CusolverTraits;

template<>
struct CusolverTraits<float> {
    static cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork) {
        return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, lwork);
    }
    
    static cusolverStatus_t getrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* workspace, int* ipiv, int* info) {
        return cusolverDnSgetrf(handle, m, n, A, lda, workspace, ipiv, info);
    }
    
    static cusolverStatus_t getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A, int lda, const int* ipiv, float* B, int ldb, int* info) {
        return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    }
    
    static cusolverStatus_t geqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork) {
        return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork);
    }
    
    static cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* tau, float* workspace, int lwork, int* info) {
        return cusolverDnSgeqrf(handle, m, n, A, lda, tau, workspace, lwork, info);
    }
    
    static cusolverStatus_t orgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork) {
        return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    }
    
    static cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m, int n, int k, float* A, int lda, const float* tau, float* workspace, int lwork, int* info) {
        return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, workspace, lwork, info);
    }
    
    static cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork) {
        return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    }
    
    static cusolverStatus_t potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* workspace, int lwork, int* info) {
        return cusolverDnSpotrf(handle, uplo, n, A, lda, workspace, lwork, info);
    }
    
    static cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork) {
        return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
    }
    
    static cusolverStatus_t syevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* workspace, int lwork, int* info) {
        return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, workspace, lwork, info);
    }
};

template<>
struct CusolverTraits<double> {
    static cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork) {
        return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, lwork);
    }
    
    static cusolverStatus_t getrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* workspace, int* ipiv, int* info) {
        return cusolverDnDgetrf(handle, m, n, A, lda, workspace, ipiv, info);
    }
    
    static cusolverStatus_t getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A, int lda, const int* ipiv, double* B, int ldb, int* info) {
        return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    }
    
    static cusolverStatus_t geqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork) {
        return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork);
    }
    
    static cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* tau, double* workspace, int lwork, int* info) {
        return cusolverDnDgeqrf(handle, m, n, A, lda, tau, workspace, lwork, info);
    }
    
    static cusolverStatus_t orgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork) {
        return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    }
    
    static cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m, int n, int k, double* A, int lda, const double* tau, double* workspace, int lwork, int* info) {
        return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, workspace, lwork, info);
    }
    
    static cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork) {
        return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    }
    
    static cusolverStatus_t potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* workspace, int lwork, int* info) {
        return cusolverDnDpotrf(handle, uplo, n, A, lda, workspace, lwork, info);
    }
    
    static cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork) {
        return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
    }
    
    static cusolverStatus_t syevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* workspace, int lwork, int* info) {
        return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, workspace, lwork, info);
    }
};

} // namespace cusolver_utils

#endif // CUSOLVER_UTILS_H 