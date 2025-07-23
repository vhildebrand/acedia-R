#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// Declare the tensor_dtype_unified function (defined in TensorDataAccess.cpp)
std::string tensor_dtype_unified(SEXP tensor_ptr);

// [[Rcpp::export]]
SEXP tensor_view_unified(SEXP tensor_ptr, IntegerVector new_shape) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        // Convert shape vector
        std::vector<size_t> shape_dims;
        for (int i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] <= 0) {
                stop("Shape dimensions must be positive");
            }
            shape_dims.push_back(static_cast<size_t>(new_shape[i]));
        }
        Shape shape(shape_dims);
        
        // Check size compatibility
        if (shape.size() != tensor->size()) {
            stop("View shape size must match original tensor size");
        }
        
        // Use the TensorBase view method
        auto result_tensor = tensor->view(shape);
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor view: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_reshape_unified(SEXP tensor_ptr, IntegerVector new_shape) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        // Convert shape vector
        std::vector<size_t> shape_dims;
        for (int i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] <= 0) {
                stop("Shape dimensions must be positive");
            }
            shape_dims.push_back(static_cast<size_t>(new_shape[i]));
        }
        Shape shape(shape_dims);
        
        // Check size compatibility
        if (shape.size() != tensor->size()) {
            stop("Reshape shape size must match original tensor size");
        }
        
        // Use the TensorBase reshape method (handles contiguous/non-contiguous automatically)
        auto result_tensor = tensor->reshape(shape);
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor reshape: " + std::string(e.what()));
    }
} 

// [[Rcpp::export]]
SEXP tensor_transpose_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        if (tensor->ndims() != 2) {
            stop("Transpose currently supports 2D tensors only");
        }
        
        // Get the underlying tensor and create transpose
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto transposed = tw->tensor().transpose();
                result_tensor = std::make_unique<TensorWrapper<float>>(
                    std::make_shared<gpuTensor<float>>(std::move(transposed))
                );
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto transposed = tw->tensor().transpose();
                result_tensor = std::make_unique<TensorWrapper<double>>(
                    std::make_shared<gpuTensor<double>>(std::move(transposed))
                );
                break;
            }
            default:
                stop("Transpose not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor transpose: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_permute_unified(SEXP tensor_ptr, IntegerVector dims) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        if (dims.size() != (int)tensor->ndims()) {
            stop("Number of dimensions in 'dims' must match tensor dimensions");
        }
        
        // Convert to 0-indexed for C++
        std::vector<int> dims_vec;
        for (int i = 0; i < dims.size(); ++i) {
            dims_vec.push_back(dims[i] - 1);  // Convert from 1-indexed to 0-indexed
        }
        
        // Get the underlying tensor and create permutation
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto permuted = tw->tensor().permute(dims_vec);
                result_tensor = std::make_unique<TensorWrapper<float>>(
                    std::make_shared<gpuTensor<float>>(std::move(permuted))
                );
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto permuted = tw->tensor().permute(dims_vec);
                result_tensor = std::make_unique<TensorWrapper<double>>(
                    std::make_shared<gpuTensor<double>>(std::move(permuted))
                );
                break;
            }
            default:
                stop("Permute not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor permute: " + std::string(e.what()));
    }
} 