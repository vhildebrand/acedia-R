#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// Forward declarations of CUDA wrappers (defined in tensor_ops.cu)
extern "C" {
    void tensor_gt_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_gt_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_lt_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_lt_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_eq_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_eq_float64(double* result, const double* a, const double* b, size_t n);
    
    // Broadcast comparison functions
    void tensor_gt_broadcast_float32(float* result, const float* a, const float* b, const int* a_strides,
                                  const int* b_strides, const int* result_strides,
                                  const int* shape, int ndims, size_t total_elements);
    void tensor_gt_broadcast_float64(double* result, const double* a, const double* b, const int* a_strides,
                                  const int* b_strides, const int* result_strides,
                                  const int* shape, int ndims, size_t total_elements);
    void tensor_lt_broadcast_float32(float* result, const float* a, const float* b, const int* a_strides,
                                  const int* b_strides, const int* result_strides,
                                  const int* shape, int ndims, size_t total_elements);
    void tensor_lt_broadcast_float64(double* result, const double* a, const double* b, const int* a_strides,
                                  const int* b_strides, const int* result_strides,
                                  const int* shape, int ndims, size_t total_elements);
    void tensor_eq_broadcast_float32(float* result, const float* a, const float* b, const int* a_strides,
                                  const int* b_strides, const int* result_strides,
                                  const int* shape, int ndims, size_t total_elements);
    void tensor_eq_broadcast_float64(double* result, const double* a, const double* b, const int* a_strides,
                                  const int* b_strides, const int* result_strides,
                                  const int* shape, int ndims, size_t total_elements);
}

// Forward declaration of broadcast utility function (defined in TensorArithmetic.cpp)
std::vector<int> compute_broadcast_strides(const Shape& tensor_shape, const Shape& broadcast_shape);

// [[Rcpp::export]]
SEXP tensor_gt_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        
        if (!a_tensor || !b_tensor) {
            stop("Invalid tensor pointer(s)");
        }
        
        if (a_tensor->dtype() != b_tensor->dtype()) {
            stop("Comparison requires tensors with identical dtypes");
        }
        
        // Check shapes compatibility
        bool shapes_equal = (a_tensor->shape() == b_tensor->shape());
        bool scalar_broadcast = (!shapes_equal) && (a_tensor->size() == 1 || b_tensor->size() == 1);
        bool needs_full_broadcast = !shapes_equal && !scalar_broadcast;

        if (needs_full_broadcast) {
            // Check if shapes are broadcastable
            if (!a_tensor->shape().broadcastable_with(b_tensor->shape())) {
                stop("Tensor shapes are not broadcastable for comparison");
            }
            
            // Compute broadcast shape
            Shape broadcast_shape = a_tensor->shape().broadcast_with(b_tensor->shape());
            DType dtype = a_tensor->dtype();
            
            std::unique_ptr<TensorBase> result_tensor;
            
            // Compute strides for broadcasting
            auto a_strides = compute_broadcast_strides(a_tensor->shape(), broadcast_shape);
            auto b_strides = compute_broadcast_strides(b_tensor->shape(), broadcast_shape);
            std::vector<int> result_strides(broadcast_shape.ndims());
            int stride = 1;
            for (int i = broadcast_shape.ndims() - 1; i >= 0; i--) {
                result_strides[i] = stride;
                stride *= broadcast_shape[i];
            }
            
            // Convert shape dimensions from size_t to int for CUDA kernel
            std::vector<int> shape_int(broadcast_shape.dims.begin(), broadcast_shape.dims.end());
            
            if (dtype == DType::FLOAT32) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(broadcast_shape);
                tensor_gt_broadcast_float32(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
            } else if (dtype == DType::FLOAT64) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(broadcast_shape);
                tensor_gt_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast comparison only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype);
            
            return ptr;
        }
        
        // Handle scalar broadcasting (one tensor has size 1)
        if (scalar_broadcast) {
            // For scalar broadcasting, just promote the scalar to the other tensor's shape
            // This is handled by the R-level code, so this shouldn't happen often
            stop("Scalar broadcasting should be handled at R level");
        }
        
        // Fast path for equal shapes
        DType dtype = a_tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        switch (dtype) {
            case DType::FLOAT32: {
                auto a_wrap = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<float>>(a_contig.shape());
                tensor_gt_float32(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result_gpu);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrap = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<double>>(a_contig.shape());
                tensor_gt_float64(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result_gpu);
                break;
            }
            default:
                stop("Comparison not yet implemented for dtype");
        }
        auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(uniq.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop(std::string("Error in comparison function: ") + e.what());
    }
}

// [[Rcpp::export]]
SEXP tensor_lt_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        
        if (!a_tensor || !b_tensor) {
            stop("Invalid tensor pointer(s)");
        }
        
        if (a_tensor->dtype() != b_tensor->dtype()) {
            stop("Comparison requires tensors with identical dtypes");
        }
        
        // Check shapes compatibility
        bool shapes_equal = (a_tensor->shape() == b_tensor->shape());
        bool scalar_broadcast = (!shapes_equal) && (a_tensor->size() == 1 || b_tensor->size() == 1);
        bool needs_full_broadcast = !shapes_equal && !scalar_broadcast;

        if (needs_full_broadcast) {
            // Check if shapes are broadcastable
            if (!a_tensor->shape().broadcastable_with(b_tensor->shape())) {
                stop("Tensor shapes are not broadcastable for comparison");
            }
            
            // Compute broadcast shape
            Shape broadcast_shape = a_tensor->shape().broadcast_with(b_tensor->shape());
            DType dtype = a_tensor->dtype();
            
            std::unique_ptr<TensorBase> result_tensor;
            
            // Compute strides for broadcasting
            auto a_strides = compute_broadcast_strides(a_tensor->shape(), broadcast_shape);
            auto b_strides = compute_broadcast_strides(b_tensor->shape(), broadcast_shape);
            std::vector<int> result_strides(broadcast_shape.ndims());
            int stride = 1;
            for (int i = broadcast_shape.ndims() - 1; i >= 0; i--) {
                result_strides[i] = stride;
                stride *= broadcast_shape[i];
            }
            
            // Convert shape dimensions from size_t to int for CUDA kernel
            std::vector<int> shape_int(broadcast_shape.dims.begin(), broadcast_shape.dims.end());
            
            if (dtype == DType::FLOAT32) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(broadcast_shape);
                tensor_lt_broadcast_float32(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
            } else if (dtype == DType::FLOAT64) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(broadcast_shape);
                tensor_lt_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast comparison only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype);
            
            return ptr;
        }
        
        // Handle scalar broadcasting (one tensor has size 1)
        if (scalar_broadcast) {
            // For scalar broadcasting, just promote the scalar to the other tensor's shape
            // This is handled by the R-level code, so this shouldn't happen often
            stop("Scalar broadcasting should be handled at R level");
        }
        
        // Fast path for equal shapes
        DType dtype = a_tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        switch (dtype) {
            case DType::FLOAT32: {
                auto a_wrap = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<float>>(a_contig.shape());
                tensor_lt_float32(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result_gpu);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrap = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<double>>(a_contig.shape());
                tensor_lt_float64(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result_gpu);
                break;
            }
            default:
                stop("Comparison not yet implemented for dtype");
        }
        auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(uniq.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop(std::string("Error in comparison function: ") + e.what());
    }
}

// [[Rcpp::export]]
SEXP tensor_eq_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        
        if (!a_tensor || !b_tensor) {
            stop("Invalid tensor pointer(s)");
        }
        
        if (a_tensor->dtype() != b_tensor->dtype()) {
            stop("Comparison requires tensors with identical dtypes");
        }
        
        // Check shapes compatibility
        bool shapes_equal = (a_tensor->shape() == b_tensor->shape());
        bool scalar_broadcast = (!shapes_equal) && (a_tensor->size() == 1 || b_tensor->size() == 1);
        bool needs_full_broadcast = !shapes_equal && !scalar_broadcast;

        if (needs_full_broadcast) {
            // Check if shapes are broadcastable
            if (!a_tensor->shape().broadcastable_with(b_tensor->shape())) {
                stop("Tensor shapes are not broadcastable for comparison");
            }
            
            // Compute broadcast shape
            Shape broadcast_shape = a_tensor->shape().broadcast_with(b_tensor->shape());
            DType dtype = a_tensor->dtype();
            
            std::unique_ptr<TensorBase> result_tensor;
            
            // Compute strides for broadcasting
            auto a_strides = compute_broadcast_strides(a_tensor->shape(), broadcast_shape);
            auto b_strides = compute_broadcast_strides(b_tensor->shape(), broadcast_shape);
            std::vector<int> result_strides(broadcast_shape.ndims());
            int stride = 1;
            for (int i = broadcast_shape.ndims() - 1; i >= 0; i--) {
                result_strides[i] = stride;
                stride *= broadcast_shape[i];
            }
            
            // Convert shape dimensions from size_t to int for CUDA kernel
            std::vector<int> shape_int(broadcast_shape.dims.begin(), broadcast_shape.dims.end());
            
            if (dtype == DType::FLOAT32) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(broadcast_shape);
                tensor_eq_broadcast_float32(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
            } else if (dtype == DType::FLOAT64) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(broadcast_shape);
                tensor_eq_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),  
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast comparison only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype);
            
            return ptr;
        }
        
        // Handle scalar broadcasting (one tensor has size 1)
        if (scalar_broadcast) {
            // For scalar broadcasting, just promote the scalar to the other tensor's shape
            // This is handled by the R-level code, so this shouldn't happen often
            stop("Scalar broadcasting should be handled at R level");
        }
        
        // Fast path for equal shapes
        DType dtype = a_tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        switch (dtype) {
            case DType::FLOAT32: {
                auto a_wrap = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<float>>(a_contig.shape());
                tensor_eq_float32(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result_gpu);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrap = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<double>>(a_contig.shape());
                tensor_eq_float64(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result_gpu);
                break;
            }
            default:
                stop("Comparison not yet implemented for dtype");
        }
        auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(uniq.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop(std::string("Error in comparison function: ") + e.what());
    }
} 