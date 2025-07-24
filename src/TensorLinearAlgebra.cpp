#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

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
            case DType::FLOAT16: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<half>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<half>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT16");
                }
                auto result = std::make_shared<gpuTensor<half>>(result_shape);
                // Note: CUDA kernel expects row-major, but R uses column-major
                tensor_matmul_float16(result->data(), b_wrapper->tensor().data(), 
                                    a_wrapper->tensor().data(), N, M, K);
                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(result_shape);
                // Note: CUDA kernel expects row-major, but R uses column-major
                // So we compute B^T * A^T = (A * B)^T, then the result is already transposed to R's column-major  
                tensor_matmul_float32(result->data(), b_wrapper->tensor().data(), 
                                    a_wrapper->tensor().data(), N, M, K);
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(result_shape);
                // Note: CUDA kernel expects row-major, but R uses column-major
                tensor_matmul_float64(result->data(), b_wrapper->tensor().data(), 
                                    a_wrapper->tensor().data(), N, M, K);
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
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
                
                // Ensure both tensors are contiguous before kernel launch
                auto a_contiguous = a_wrapper->tensor().is_contiguous() ? a_wrapper->tensor() : a_wrapper->tensor().contiguous();
                auto b_contiguous = b_wrapper->tensor().is_contiguous() ? b_wrapper->tensor() : b_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<half>>(result_shape);
                tensor_outer_product_float16(result->data(), a_contiguous.data(), 
                                           b_contiguous.data(), M, N);
                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                
                // Ensure both tensors are contiguous before kernel launch
                auto a_contiguous = a_wrapper->tensor().is_contiguous() ? a_wrapper->tensor() : a_wrapper->tensor().contiguous();
                auto b_contiguous = b_wrapper->tensor().is_contiguous() ? b_wrapper->tensor() : b_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<float>>(result_shape);
                tensor_outer_product_float32(result->data(), a_contiguous.data(), 
                                            b_contiguous.data(), M, N);
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                
                // Ensure both tensors are contiguous before kernel launch
                auto a_contiguous = a_wrapper->tensor().is_contiguous() ? a_wrapper->tensor() : a_wrapper->tensor().contiguous();
                auto b_contiguous = b_wrapper->tensor().is_contiguous() ? b_wrapper->tensor() : b_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<double>>(result_shape);
                tensor_outer_product_float64(result->data(), a_contiguous.data(), 
                                           b_contiguous.data(), M, N);
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
        
        Shape result_shape({M});  // Result is M-length vector
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype_A) {
            case DType::FLOAT16: {
                auto A_wrapper = dynamic_cast<const TensorWrapper<half>*>(A_tensor.get());
                auto v_wrapper = dynamic_cast<const TensorWrapper<half>*>(v_tensor.get());
                if (!A_wrapper || !v_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT16");
                }
                
                // Ensure both tensors are contiguous before kernel launch
                auto A_contiguous = A_wrapper->tensor().is_contiguous() ? A_wrapper->tensor() : A_wrapper->tensor().contiguous();
                auto v_contiguous = v_wrapper->tensor().is_contiguous() ? v_wrapper->tensor() : v_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<half>>(result_shape);
                tensor_matvec_float16(result->data(), A_contiguous.data(), 
                                    v_contiguous.data(), M, N);
                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto A_wrapper = dynamic_cast<const TensorWrapper<float>*>(A_tensor.get());
                auto v_wrapper = dynamic_cast<const TensorWrapper<float>*>(v_tensor.get());
                if (!A_wrapper || !v_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                
                // Ensure both tensors are contiguous before kernel launch
                auto A_contiguous = A_wrapper->tensor().is_contiguous() ? A_wrapper->tensor() : A_wrapper->tensor().contiguous();
                auto v_contiguous = v_wrapper->tensor().is_contiguous() ? v_wrapper->tensor() : v_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<float>>(result_shape);
                tensor_matvec_float32(result->data(), A_contiguous.data(), 
                                    v_contiguous.data(), M, N);
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto A_wrapper = dynamic_cast<const TensorWrapper<double>*>(A_tensor.get());
                auto v_wrapper = dynamic_cast<const TensorWrapper<double>*>(v_tensor.get());
                if (!A_wrapper || !v_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                
                // Ensure both tensors are contiguous before kernel launch
                auto A_contiguous = A_wrapper->tensor().is_contiguous() ? A_wrapper->tensor() : A_wrapper->tensor().contiguous();
                auto v_contiguous = v_wrapper->tensor().is_contiguous() ? v_wrapper->tensor() : v_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<double>>(result_shape);
                tensor_matvec_float64(result->data(), A_contiguous.data(), 
                                    v_contiguous.data(), M, N);
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
        
        Shape result_shape({N});  // Result is N-length vector
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype_v) {
            case DType::FLOAT16: {
                auto v_wrapper = dynamic_cast<const TensorWrapper<half>*>(v_tensor.get());
                auto A_wrapper = dynamic_cast<const TensorWrapper<half>*>(A_tensor.get());
                if (!v_wrapper || !A_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT16");
                }
                
                // Ensure both tensors are contiguous before kernel launch
                auto v_contiguous = v_wrapper->tensor().is_contiguous() ? v_wrapper->tensor() : v_wrapper->tensor().contiguous();
                auto A_contiguous = A_wrapper->tensor().is_contiguous() ? A_wrapper->tensor() : A_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<half>>(result_shape);
                tensor_vecmat_float16(result->data(), v_contiguous.data(), 
                                    A_contiguous.data(), M, N);
                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto v_wrapper = dynamic_cast<const TensorWrapper<float>*>(v_tensor.get());
                auto A_wrapper = dynamic_cast<const TensorWrapper<float>*>(A_tensor.get());
                if (!v_wrapper || !A_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                
                // Ensure both tensors are contiguous before kernel launch
                auto v_contiguous = v_wrapper->tensor().is_contiguous() ? v_wrapper->tensor() : v_wrapper->tensor().contiguous();
                auto A_contiguous = A_wrapper->tensor().is_contiguous() ? A_wrapper->tensor() : A_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<float>>(result_shape);
                tensor_vecmat_float32(result->data(), v_contiguous.data(), 
                                    A_contiguous.data(), M, N);
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto v_wrapper = dynamic_cast<const TensorWrapper<double>*>(v_tensor.get());
                auto A_wrapper = dynamic_cast<const TensorWrapper<double>*>(A_tensor.get());
                if (!v_wrapper || !A_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                
                // Ensure both tensors are contiguous before kernel launch
                auto v_contiguous = v_wrapper->tensor().is_contiguous() ? v_wrapper->tensor() : v_wrapper->tensor().contiguous();
                auto A_contiguous = A_wrapper->tensor().is_contiguous() ? A_wrapper->tensor() : A_wrapper->tensor().contiguous();
                
                auto result = std::make_shared<gpuTensor<double>>(result_shape);
                tensor_vecmat_float64(result->data(), v_contiguous.data(), 
                                    A_contiguous.data(), M, N);
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