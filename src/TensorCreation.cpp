#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP create_tensor_unified(NumericVector data, IntegerVector shape_vec, std::string dtype = "float32") {
    try {
        if (!cuda_utils::isGpuAvailable()) {
            stop("GPU not available - cannot create gpuTensor objects");
        }
        
        // Convert shape vector
        std::vector<size_t> shape_dims;
        for (int i = 0; i < shape_vec.size(); ++i) {
            if (shape_vec[i] <= 0) {
                stop("Shape dimensions must be positive");
            }
            shape_dims.push_back(static_cast<size_t>(shape_vec[i]));
        }
        Shape shape(shape_dims);
        
        // Verify data size matches shape
        if (data.size() != static_cast<int>(shape.size())) {
            stop("Data size doesn't match shape size");
        }
        
        // Convert to std::vector<double>
        std::vector<double> data_vec(data.begin(), data.end());
        
        // Create tensor using factory
        auto tensor = TensorFactory::create_tensor_by_dtype(data_vec, shape, dtype);
        
        // Wrap in external pointer
        auto tensor_ptr = std::unique_ptr<TensorBase>(tensor.release());
        XPtr<TensorBase> ptr(tensor_ptr.release(), true);
        
        // Set attributes
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype;
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error creating unified tensor: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP create_empty_tensor_unified(IntegerVector shape_vec, std::string dtype = "float32") {
    try {
        if (!cuda_utils::isGpuAvailable()) {
            stop("GPU not available - cannot create gpuTensor objects");
        }
        
        // Convert shape vector
        std::vector<size_t> shape_dims;
        for (int i = 0; i < shape_vec.size(); ++i) {
            if (shape_vec[i] <= 0) {
                stop("Shape dimensions must be positive");
            }
            shape_dims.push_back(static_cast<size_t>(shape_vec[i]));
        }
        Shape shape(shape_dims);
        
        // Create empty tensor
        auto tensor = TensorFactory::create_empty_tensor_by_dtype(shape, dtype);
        
        // Wrap in external pointer
        auto tensor_ptr = std::unique_ptr<TensorBase>(tensor.release());
        XPtr<TensorBase> ptr(tensor_ptr.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype;
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error creating empty unified tensor: " + std::string(e.what()));
    }
} 