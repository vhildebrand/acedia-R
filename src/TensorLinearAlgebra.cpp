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