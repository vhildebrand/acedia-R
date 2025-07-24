#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <cublas_v2.h>
#include <memory>
#include <optional> // Added for std::optional

using namespace Rcpp;

// Forward declarations for matrix operations
extern "C" {
    // Matrix multiplication
    void tensor_matmul_float16(half* C, const half* A, const half* B, size_t M, size_t N, size_t K);
    void tensor_matmul_float32(float* C, const float* A, const float* B, size_t M, size_t N, size_t K);
    void tensor_matmul_float64(double* C, const double* A, const double* B, size_t M, size_t N, size_t K);
    
    // Outer product
    void tensor_outer_product_float16(half* result, const half* a, const half* b, size_t M, size_t N);
    void tensor_outer_product_float32(float* result, const float* a, const float* b, size_t M, size_t N);
    void tensor_outer_product_float64(double* result, const double* a, const double* b, size_t M, size_t N);
    
    // Matrix-vector multiplication
    void tensor_matvec_float16(half* result, const half* A, const half* v, size_t M, size_t N);
    void tensor_matvec_float32(float* result, const float* A, const float* v, size_t M, size_t N);
    void tensor_matvec_float64(double* result, const double* A, const double* v, size_t M, size_t N);
    
    // Vector-matrix multiplication
    void tensor_vecmat_float16(half* result, const half* v, const half* A, size_t M, size_t N);
    void tensor_vecmat_float32(float* result, const float* v, const float* A, size_t M, size_t N);
    void tensor_vecmat_float64(double* result, const double* v, const double* A, size_t M, size_t N);
}

// Detect a 2-D view that is just a simple transpose of a contiguous tensor.
// Criteria: (1) tensor has 2 dims, (2) not contiguous, (3) stride pattern is swapped
//           i.e., strides = {shape[1], 1} when column-major base layout is {1, shape[0]}.
template<typename T>
static inline bool is_simple_transpose_view(const gpuTensor<T>& t) {
    if (t.ndims() != 2) return false;
    if (t.is_contiguous()) return false;
    const auto& strides = t.strides();
    const auto& shape   = t.shape();
    // column-major base contiguous strides would be {1, shape[0]}
    // a pure transpose view swaps them to {shape[1], 1}
    return (strides[1] == 1) && (strides[0] == shape[1]);
}

// [[Rcpp::export]]
SEXP tensor_matmul_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        
        if (!a_tensor || !b_tensor) {
            stop("Invalid tensor pointer(s)");
        }
        
        // Check that both tensors are 2D
        if (a_tensor->ndims() != 2 || b_tensor->ndims() != 2) {
            stop("Matrix multiplication requires 2D tensors");
        }
        
        auto a_shape = a_tensor->shape();
        auto b_shape = b_tensor->shape();
        
        // Check dimensions compatibility for matrix multiplication (M x K) * (K x N) = (M x N)
        if (a_shape.dims[1] != b_shape.dims[0]) {
            stop("Incompatible dimensions for matrix multiplication");
        }
        
        size_t M = a_shape.dims[0];
        size_t K = a_shape.dims[1];
        size_t N = b_shape.dims[1];
        
        DType dtype_a = a_tensor->dtype();
        DType dtype_b = b_tensor->dtype();
        
        if (dtype_a != dtype_b) {
            stop("Cannot multiply tensors with different dtypes");
        }
        
        Shape result_shape({M, N});
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype_a) {
            case DType::FLOAT32: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }

                // Decide contiguity / transpose handling for A
                const auto& a_ref = a_wrapper->tensor();
                const auto& b_ref = b_wrapper->tensor();

                std::optional<gpuTensor<float>> a_temp;  // will hold contiguous copy if needed
                std::optional<gpuTensor<float>> b_temp;

                const gpuTensor<float>* a_mat;
                const gpuTensor<float>* b_mat;
                cublasOperation_t opA = CUBLAS_OP_N;
                cublasOperation_t opB = CUBLAS_OP_N;

                if (a_ref.is_contiguous()) {
                    a_mat = &a_ref;
                    opA   = CUBLAS_OP_N;
                } else if (is_simple_transpose_view(a_ref)) {
                    a_mat = &a_ref;
                    opA   = CUBLAS_OP_T;
                } else {
                    a_temp = a_ref.contiguous();
                    a_mat  = &(*a_temp);
                    opA    = CUBLAS_OP_N;
                }

                if (b_ref.is_contiguous()) {
                    b_mat = &b_ref;
                    opB   = CUBLAS_OP_N;
                } else if (is_simple_transpose_view(b_ref)) {
                    b_mat = &b_ref;
                    opB   = CUBLAS_OP_T;
                } else {
                    b_temp = b_ref.contiguous();
                    b_mat  = &(*b_temp);
                    opB    = CUBLAS_OP_N;
                }

                auto result = std::make_shared<gpuTensor<float>>(result_shape);

                const float alpha = 1.0f;
                const float beta  = 0.0f;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasSgemm(
                    handle,
                    opA,
                    opB,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    static_cast<int>(K),
                    &alpha,
                    a_mat->data(),
                    (opA == CUBLAS_OP_N) ? static_cast<int>(M) : static_cast<int>(K),
                    b_mat->data(),
                    (opB == CUBLAS_OP_N) ? static_cast<int>(K) : static_cast<int>(N),
                    &beta,
                    result->data(),
                    static_cast<int>(M));

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasSgemm failed");
                }
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }

                const auto& a_ref = a_wrapper->tensor();
                const auto& b_ref = b_wrapper->tensor();

                std::optional<gpuTensor<double>> a_temp;
                std::optional<gpuTensor<double>> b_temp;

                const gpuTensor<double>* a_mat;
                const gpuTensor<double>* b_mat;
                cublasOperation_t opA = CUBLAS_OP_N;
                cublasOperation_t opB = CUBLAS_OP_N;

                if (a_ref.is_contiguous()) {
                    a_mat = &a_ref;
                } else if (is_simple_transpose_view(a_ref)) {
                    a_mat = &a_ref;
                    opA   = CUBLAS_OP_T;
                } else {
                    a_temp = a_ref.contiguous();
                    a_mat  = &(*a_temp);
                }

                if (b_ref.is_contiguous()) {
                    b_mat = &b_ref;
                } else if (is_simple_transpose_view(b_ref)) {
                    b_mat = &b_ref;
                    opB   = CUBLAS_OP_T;
                } else {
                    b_temp = b_ref.contiguous();
                    b_mat  = &(*b_temp);
                }

                auto result = std::make_shared<gpuTensor<double>>(result_shape);

                const double alpha = 1.0;
                const double beta  = 0.0;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasDgemm(
                    handle,
                    opA,
                    opB,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    static_cast<int>(K),
                    &alpha,
                    a_mat->data(),
                    (opA == CUBLAS_OP_N) ? static_cast<int>(M) : static_cast<int>(K),
                    b_mat->data(),
                    (opB == CUBLAS_OP_N) ? static_cast<int>(K) : static_cast<int>(N),
                    &beta,
                    result->data(),
                    static_cast<int>(M));

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasDgemm failed");
                }
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            case DType::FLOAT16: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<half>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<half>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT16");
                }

                const auto& a_ref = a_wrapper->tensor();
                const auto& b_ref = b_wrapper->tensor();

                std::optional<gpuTensor<half>> a_temp;
                std::optional<gpuTensor<half>> b_temp;

                const gpuTensor<half>* a_mat;
                const gpuTensor<half>* b_mat;
                cublasOperation_t opA = CUBLAS_OP_N;
                cublasOperation_t opB = CUBLAS_OP_N;

                if (a_ref.is_contiguous()) {
                    a_mat = &a_ref;
                } else if (is_simple_transpose_view(a_ref)) {
                    a_mat = &a_ref;
                    opA   = CUBLAS_OP_T;
                } else {
                    a_temp = a_ref.contiguous();
                    a_mat  = &(*a_temp);
                }

                if (b_ref.is_contiguous()) {
                    b_mat = &b_ref;
                } else if (is_simple_transpose_view(b_ref)) {
                    b_mat = &b_ref;
                    opB   = CUBLAS_OP_T;
                } else {
                    b_temp = b_ref.contiguous();
                    b_mat  = &(*b_temp);
                }

                auto result = std::make_shared<gpuTensor<half>>(result_shape);

                const float alpha_f = 1.0f;
                const float beta_f  = 0.0f;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasGemmEx(
                    handle,
                    opA,
                    opB,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    static_cast<int>(K),
                    &alpha_f,
                    a_mat->data(), CUDA_R_16F,
                    (opA == CUBLAS_OP_N) ? static_cast<int>(M) : static_cast<int>(K),
                    b_mat->data(), CUDA_R_16F,
                    (opB == CUBLAS_OP_N) ? static_cast<int>(K) : static_cast<int>(N),
                    &beta_f,
                    result->data(), CUDA_R_16F,
                    static_cast<int>(M),
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasGemmEx (float16) failed");
                }

                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            default:
                stop("Matrix multiplication not implemented for dtype: " + dtype_to_string(dtype_a));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype_a);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor matrix multiplication: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_outer_product_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        
        if (!a_tensor || !b_tensor) {
            stop("Invalid tensor pointer(s)");
        }
        
        // Check that both tensors are 1D
        if (a_tensor->ndims() != 1 || b_tensor->ndims() != 1) {
            stop("Outer product requires 1D tensors (vectors)");
        }
        
        auto a_shape = a_tensor->shape();
        auto b_shape = b_tensor->shape();
        
        size_t M = a_shape.dims[0];  // length of vector a
        size_t N = b_shape.dims[0];  // length of vector b
        
        DType dtype_a = a_tensor->dtype();
        DType dtype_b = b_tensor->dtype();
        
        if (dtype_a != dtype_b) {
            stop("Cannot compute outer product of tensors with different dtypes");
        }
        
        Shape result_shape({M, N});  // M×N matrix
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype_a) {
            case DType::FLOAT16: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<half>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<half>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT16");
                }
                
                auto a_contig = a_wrapper->tensor().is_contiguous() ? a_wrapper->tensor() : a_wrapper->tensor().contiguous();
                auto b_contig = b_wrapper->tensor().is_contiguous() ? b_wrapper->tensor() : b_wrapper->tensor().contiguous();

                auto result = std::make_shared<gpuTensor<half>>(result_shape);

                const float alpha_f = 1.0f;
                const float beta_f  = 0.0f;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                // Use GEMMEx with K=1 (rank-1 outer product)
                cublasStatus_t stat = cublasGemmEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,             // a (M x 1) * b^T (1 x N)
                    static_cast<int>(M),
                    static_cast<int>(N),
                    1,
                    &alpha_f,
                    a_contig.data(), CUDA_R_16F, static_cast<int>(M),
                    b_contig.data(), CUDA_R_16F, static_cast<int>(N),
                    &beta_f,
                    result->data(), CUDA_R_16F, static_cast<int>(M),
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasGemmEx (outer product float16) failed");
                }

                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                
                // Ensure contiguity
                auto a_contig = a_wrapper->tensor().is_contiguous() ? a_wrapper->tensor() : a_wrapper->tensor().contiguous();
                auto b_contig = b_wrapper->tensor().is_contiguous() ? b_wrapper->tensor() : b_wrapper->tensor().contiguous();

                auto result = std::make_shared<gpuTensor<float>>(result_shape);

                // Initialize result to zero
                cudaMemset(result->data(), 0, sizeof(float)*M*N);

                const float alpha = 1.0f;
                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasSger(
                    handle,
                    static_cast<int>(M),   // rows
                    static_cast<int>(N),   // cols
                    &alpha,
                    a_contig.data(), 1,    // x
                    b_contig.data(), 1,    // y
                    result->data(), static_cast<int>(M)); // A

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasSger failed");
                }

                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                
                auto a_contig = a_wrapper->tensor().is_contiguous() ? a_wrapper->tensor() : a_wrapper->tensor().contiguous();
                auto b_contig = b_wrapper->tensor().is_contiguous() ? b_wrapper->tensor() : b_wrapper->tensor().contiguous();

                auto result = std::make_shared<gpuTensor<double>>(result_shape);
                cudaMemset(result->data(), 0, sizeof(double)*M*N);

                const double alpha = 1.0;
                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasDger(
                    handle,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    &alpha,
                    a_contig.data(), 1,
                    b_contig.data(), 1,
                    result->data(), static_cast<int>(M));

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasDger failed");
                }

                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Outer product not implemented for dtype: " + dtype_to_string(dtype_a));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype_a);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor outer product: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_matvec_unified(SEXP A_ptr, SEXP v_ptr) {
    try {
        XPtr<TensorBase> A_tensor(A_ptr);
        XPtr<TensorBase> v_tensor(v_ptr);
        
        if (!A_tensor || !v_tensor) {
            stop("Invalid tensor pointer(s)");
        }
        
        // Check dimensions: A must be 2D, v must be 1D
        if (A_tensor->ndims() != 2) {
            stop("Matrix-vector multiplication requires 2D matrix");
        }
        if (v_tensor->ndims() != 1) {
            stop("Matrix-vector multiplication requires 1D vector");
        }
        
        auto A_shape = A_tensor->shape();
        auto v_shape = v_tensor->shape();
        
        size_t M = A_shape.dims[0];  // rows of matrix
        size_t N = A_shape.dims[1];  // cols of matrix
        size_t V = v_shape.dims[0];  // length of vector
        
        // Check compatibility: matrix cols must equal vector length
        if (N != V) {
            stop("Incompatible dimensions for matrix-vector multiplication: matrix cols (" + 
                 std::to_string(N) + ") != vector length (" + std::to_string(V) + ")");
        }
        
        DType dtype_A = A_tensor->dtype();
        DType dtype_v = v_tensor->dtype();
        
        if (dtype_A != dtype_v) {
            stop("Cannot multiply matrix and vector with different dtypes");
        }
        
        Shape result_shape({M, 1});  // Result is M×1 matrix (like R)
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype_A) {
            case DType::FLOAT16: {
                auto A_wrapper = dynamic_cast<const TensorWrapper<half>*>(A_tensor.get());
                auto v_wrapper = dynamic_cast<const TensorWrapper<half>*>(v_tensor.get());
                if (!A_wrapper || !v_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT16");
                }
                
                auto A_contig = A_wrapper->tensor().is_contiguous() ? A_wrapper->tensor() : A_wrapper->tensor().contiguous();
                auto v_contig = v_wrapper->tensor().is_contiguous() ? v_wrapper->tensor() : v_wrapper->tensor().contiguous();

                auto result = std::make_shared<gpuTensor<half>>(result_shape);

                const float alpha_f = 1.0f;
                const float beta_f  = 0.0f;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                // Use GEMMEx with vector as 1xN matrix (transpose trick)
                cublasStatus_t stat = cublasGemmEx(
                    handle,
                    CUBLAS_OP_T, CUBLAS_OP_T,
                    1, static_cast<int>(M), static_cast<int>(N),
                    &alpha_f,
                    v_contig.data(), CUDA_R_16F, 1,
                    A_contig.data(), CUDA_R_16F, static_cast<int>(M),
                    &beta_f,
                    result->data(), CUDA_R_16F, 1,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasGemmEx (matvec float16) failed");
                }

                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto A_wrapper = dynamic_cast<const TensorWrapper<float>*>(A_tensor.get());
                auto v_wrapper = dynamic_cast<const TensorWrapper<float>*>(v_tensor.get());
                if (!A_wrapper || !v_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }

                const auto& A_ref = A_wrapper->tensor();
                const auto& v_ref = v_wrapper->tensor();

                std::optional<gpuTensor<float>> A_temp;

                const gpuTensor<float>* A_mat;
                cublasOperation_t opA = CUBLAS_OP_N;

                if (A_ref.is_contiguous()) {
                    A_mat = &A_ref;
                } else if (is_simple_transpose_view(A_ref)) {
                    A_mat = &A_ref;
                    opA   = CUBLAS_OP_T;
                } else {
                    A_temp = A_ref.contiguous();
                    A_mat  = &(*A_temp);
                }

                auto result = std::make_shared<gpuTensor<float>>(result_shape);

                const float alpha = 1.0f;
                const float beta  = 0.0f;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasSgemv(
                    handle,
                    opA,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    &alpha,
                    A_mat->data(),
                    static_cast<int>(M),  // lda is always M (rows of original matrix)
                    v_ref.data(), 1,
                    &beta,
                    result->data(), 1);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasSgemv (matvec) failed");
                }

                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto A_wrapper = dynamic_cast<const TensorWrapper<double>*>(A_tensor.get());
                auto v_wrapper = dynamic_cast<const TensorWrapper<double>*>(v_tensor.get());
                if (!A_wrapper || !v_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }

                const auto& A_ref = A_wrapper->tensor();
                const auto& v_ref = v_wrapper->tensor();

                std::optional<gpuTensor<double>> A_temp;
                const gpuTensor<double>* A_mat;
                cublasOperation_t opA = CUBLAS_OP_N;

                if (A_ref.is_contiguous()) {
                    A_mat = &A_ref;
                } else if (is_simple_transpose_view(A_ref)) {
                    A_mat = &A_ref;
                    opA   = CUBLAS_OP_T;
                } else {
                    A_temp = A_ref.contiguous();
                    A_mat  = &(*A_temp);
                }

                auto result = std::make_shared<gpuTensor<double>>(result_shape);

                const double alpha = 1.0;
                const double beta  = 0.0;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasDgemv(
                    handle,
                    opA,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    &alpha,
                    A_mat->data(),
                    static_cast<int>(M),  // lda is always M (rows of original matrix)
                    v_ref.data(), 1,
                    &beta,
                    result->data(), 1);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasDgemv (matvec) failed");
                }

                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Matrix-vector multiplication not implemented for dtype: " + dtype_to_string(dtype_A));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype_A);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor matrix-vector multiplication: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_vecmat_unified(SEXP v_ptr, SEXP A_ptr) {
    try {
        XPtr<TensorBase> v_tensor(v_ptr);
        XPtr<TensorBase> A_tensor(A_ptr);
        
        if (!v_tensor || !A_tensor) {
            stop("Invalid tensor pointer(s)");
        }
        
        // Check dimensions: v must be 1D, A must be 2D
        if (v_tensor->ndims() != 1) {
            stop("Vector-matrix multiplication requires 1D vector");
        }
        if (A_tensor->ndims() != 2) {
            stop("Vector-matrix multiplication requires 2D matrix");
        }
        
        auto v_shape = v_tensor->shape();
        auto A_shape = A_tensor->shape();
        
        size_t V = v_shape.dims[0];  // length of vector
        size_t M = A_shape.dims[0];  // rows of matrix
        size_t N = A_shape.dims[1];  // cols of matrix
        
        // Check compatibility: vector length must equal matrix rows
        if (V != M) {
            stop("Incompatible dimensions for vector-matrix multiplication: vector length (" + 
                 std::to_string(V) + ") != matrix rows (" + std::to_string(M) + ")");
        }
        
        DType dtype_v = v_tensor->dtype();
        DType dtype_A = A_tensor->dtype();
        
        if (dtype_v != dtype_A) {
            stop("Cannot multiply vector and matrix with different dtypes");
        }
        
        Shape result_shape({1, N});  // Result is 1×N matrix (like R)
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype_v) {
            case DType::FLOAT16: {
                auto v_wrapper = dynamic_cast<const TensorWrapper<half>*>(v_tensor.get());
                auto A_wrapper = dynamic_cast<const TensorWrapper<half>*>(A_tensor.get());
                if (!v_wrapper || !A_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT16");
                }
                
                auto v_contig = v_wrapper->tensor().is_contiguous() ? v_wrapper->tensor() : v_wrapper->tensor().contiguous();
                auto A_contig = A_wrapper->tensor().is_contiguous() ? A_wrapper->tensor() : A_wrapper->tensor().contiguous();

                auto result = std::make_shared<gpuTensor<half>>(result_shape);

                const float alpha_f = 1.0f;
                const float beta_f  = 0.0f;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                // Treat v as 1xM, A as MxN: use GEMMEx
                cublasStatus_t stat = cublasGemmEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), 1, static_cast<int>(M),
                    &alpha_f,
                    A_contig.data(), CUDA_R_16F, static_cast<int>(N), // row-major mem but OP_N, dims N x M? but we follow pattern
                    v_contig.data(), CUDA_R_16F, static_cast<int>(M),
                    &beta_f,
                    result->data(), CUDA_R_16F, static_cast<int>(N),
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasGemmEx (vecmat float16) failed");
                }

                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto v_wrapper = dynamic_cast<const TensorWrapper<float>*>(v_tensor.get());
                auto A_wrapper = dynamic_cast<const TensorWrapper<float>*>(A_tensor.get());
                if (!v_wrapper || !A_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }

                const auto& v_ref = v_wrapper->tensor();
                const auto& A_ref = A_wrapper->tensor();

                std::optional<gpuTensor<float>> A_temp;
                const gpuTensor<float>* A_mat;
                cublasOperation_t opA = CUBLAS_OP_T; // existing code uses transpose; we'll decide dynamically

                if (A_ref.is_contiguous()) {
                    A_mat = &A_ref;
                    opA   = CUBLAS_OP_T; // we want vector^T * A ; A will be transposed so we keep same
                } else if (is_simple_transpose_view(A_ref)) {
                    // If A is already a transpose view we can use OP_N
                    A_mat = &A_ref;
                    opA   = CUBLAS_OP_N;
                } else {
                    A_temp = A_ref.contiguous();
                    A_mat  = &(*A_temp);
                    opA    = CUBLAS_OP_T;
                }

                auto result = std::make_shared<gpuTensor<float>>(result_shape);

                const float alpha = 1.0f;
                const float beta  = 0.0f;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasSgemv(
                    handle,
                    opA,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    &alpha,
                    A_mat->data(),
                    static_cast<int>(M),  // lda is always M (rows of original matrix)
                    v_ref.data(), 1,
                    &beta,
                    result->data(), 1);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasSgemv (vecmat) failed");
                }

                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto v_wrapper = dynamic_cast<const TensorWrapper<double>*>(v_tensor.get());
                auto A_wrapper = dynamic_cast<const TensorWrapper<double>*>(A_tensor.get());
                if (!v_wrapper || !A_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }

                const auto& v_ref = v_wrapper->tensor();
                const auto& A_ref = A_wrapper->tensor();

                std::optional<gpuTensor<double>> A_temp;
                const gpuTensor<double>* A_mat;
                cublasOperation_t opA = CUBLAS_OP_T;

                if (A_ref.is_contiguous()) {
                    A_mat = &A_ref;
                } else if (is_simple_transpose_view(A_ref)) {
                    A_mat = &A_ref;
                    opA   = CUBLAS_OP_N;
                } else {
                    A_temp = A_ref.contiguous();
                    A_mat  = &(*A_temp);
                }

                auto result = std::make_shared<gpuTensor<double>>(result_shape);

                const double alpha = 1.0;
                const double beta  = 0.0;

                cublasHandle_t handle = cuda_utils::get_cublas_handle();

                cublasStatus_t stat = cublasDgemv(
                    handle,
                    opA,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    &alpha,
                    A_mat->data(),
                    static_cast<int>(M),  // lda is always M (rows of original matrix)
                    v_ref.data(), 1,
                    &beta,
                    result->data(), 1);

                if (stat != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasDgemv (vecmat) failed");
                }

                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Vector-matrix multiplication not implemented for dtype: " + dtype_to_string(dtype_v));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype_v);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor vector-matrix multiplication: " + std::string(e.what()));
    }
} 