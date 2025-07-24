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

    // Product reductions
    float tensor_prod_float32(const float* input, size_t n);
    double tensor_prod_float64(const double* input, size_t n);

    // Variance computations (population)
    double tensor_var_float32(const float* input, size_t n);
    double tensor_var_float64(const double* input, size_t n);
}

// [[Rcpp::export]]
SEXP tensor_sum_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        // Use the new TensorBase::sum() method that returns a tensor
        auto result_tensor = tensor->sum();
        
        // Wrap the result and return as SEXP
        XPtr<TensorBase> result_ptr(result_tensor.release(), true);
        result_ptr.attr("class") = "gpuTensor";
        result_ptr.attr("dtype") = dtype_to_string(result_ptr->dtype());
        return result_ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor sum: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_mean_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        if (tensor->size() == 0) {
            stop("Cannot compute mean of empty tensor");
        }
        
        // Use the new TensorBase::mean() method that returns a tensor
        auto result_tensor = tensor->mean();
        
        // Wrap the result and return as SEXP
        XPtr<TensorBase> result_ptr(result_tensor.release(), true);
        result_ptr.attr("class") = "gpuTensor";
        result_ptr.attr("dtype") = dtype_to_string(result_ptr->dtype());
        return result_ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor mean: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_max_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        // Use the new TensorBase::max() method that returns a tensor
        auto result_tensor = tensor->max();
        
        // Wrap the result and return as SEXP
        XPtr<TensorBase> result_ptr(result_tensor.release(), true);
        result_ptr.attr("class") = "gpuTensor";
        result_ptr.attr("dtype") = dtype_to_string(result_ptr->dtype());
        return result_ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor max: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_min_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        // Use the new TensorBase::min() method that returns a tensor
        auto result_tensor = tensor->min();
        
        // Wrap the result and return as SEXP
        XPtr<TensorBase> result_ptr(result_tensor.release(), true);
        result_ptr.attr("class") = "gpuTensor";
        result_ptr.attr("dtype") = dtype_to_string(result_ptr->dtype());
        return result_ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor min: " + std::string(e.what()));
    }
} 

// [[Rcpp::export]]
SEXP tensor_prod_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        if (!tensor) stop("Invalid tensor pointer");
        
        // Use the new TensorBase::prod() method that returns a tensor
        auto result_tensor = tensor->prod();
        
        // Wrap the result and return as SEXP
        XPtr<TensorBase> result_ptr(result_tensor.release(), true);
        result_ptr.attr("class") = "gpuTensor";
        result_ptr.attr("dtype") = dtype_to_string(result_ptr->dtype());
        return result_ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor prod: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_var_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        if (!tensor) stop("Invalid tensor pointer");
        if (tensor->size() == 0) stop("Cannot compute variance of empty tensor");
        
        // Use the new TensorBase::var() method that returns a tensor
        auto result_tensor = tensor->var();
        
        // Wrap the result and return as SEXP
        XPtr<TensorBase> result_ptr(result_tensor.release(), true);
        result_ptr.attr("class") = "gpuTensor";
        result_ptr.attr("dtype") = dtype_to_string(result_ptr->dtype());
        return result_ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor variance: " + std::string(e.what()));
    }
} 