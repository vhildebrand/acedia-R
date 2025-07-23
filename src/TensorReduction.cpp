#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// Forward declarations for reduction operations
extern "C" {
    // Sum reductions
    float tensor_sum_float16(const half* input, size_t n);
    float tensor_sum_float32(const float* input, size_t n);
    double tensor_sum_float64(const double* input, size_t n);
    int64_t tensor_sum_int64(const int64_t* input, size_t n);
    
    // Min/max reductions
    float tensor_max_float32(const float* input, size_t n);
    double tensor_max_float64(const double* input, size_t n);
    float tensor_min_float32(const float* input, size_t n);
    double tensor_min_float64(const double* input, size_t n);
}

// Declare the tensor_sum_unified function (needed for mean calculation)
double tensor_sum_unified(SEXP tensor_ptr);

// [[Rcpp::export]]
double tensor_sum_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        double result = 0.0;
        
        switch (dtype) {
            case DType::FLOAT16: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<half>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT16");
                }
                result = static_cast<double>(tensor_sum_float16(tensor_wrapper->tensor().data(), 
                                                              tensor->size()));
                break;
            }
            case DType::FLOAT32: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                }
                result = static_cast<double>(tensor_sum_float32(tensor_wrapper->tensor().data(), 
                                                              tensor->size()));
                break;
            }
            case DType::FLOAT64: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                }
                result = tensor_sum_float64(tensor_wrapper->tensor().data(), tensor->size());
                break;
            }
            case DType::INT64: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<int64_t>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for INT64");
                }
                result = static_cast<double>(tensor_sum_int64(tensor_wrapper->tensor().data(), 
                                                            tensor->size()));
                break;
            }
            default:
                stop("Sum operation not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        return result;
    } catch (const std::exception& e) {
        stop("Error in unified tensor sum: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
double tensor_mean_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        if (tensor->size() == 0) {
            return 0.0;
        }
        
        // Get sum and divide by size
        double total_sum = tensor_sum_unified(tensor_ptr);
        return total_sum / static_cast<double>(tensor->size());
    } catch (const std::exception& e) {
        stop("Error in unified tensor mean: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
double tensor_max_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        double result = 0.0;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                }
                result = static_cast<double>(tensor_max_float32(tensor_wrapper->tensor().data(), 
                                                              tensor->size()));
                break;
            }
            case DType::FLOAT64: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                }
                result = tensor_max_float64(tensor_wrapper->tensor().data(), tensor->size());
                break;
            }
            default:
                stop("Max operation not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        return result;
    } catch (const std::exception& e) {
        stop("Error in unified tensor max: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
double tensor_min_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        double result = 0.0;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                }
                result = static_cast<double>(tensor_min_float32(tensor_wrapper->tensor().data(), 
                                                              tensor->size()));
                break;
            }
            case DType::FLOAT64: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                }
                result = tensor_min_float64(tensor_wrapper->tensor().data(), tensor->size());
                break;
            }
            default:
                stop("Min operation not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        return result;
    } catch (const std::exception& e) {
        stop("Error in unified tensor min: " + std::string(e.what()));
    }
} 