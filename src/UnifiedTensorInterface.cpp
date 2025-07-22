#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// Forward declarations of templated CUDA functions
extern "C" {
    // Fill operations for different types
    void tensor_fill_float16(half* data, float value, size_t n);
    void tensor_fill_float32(float* data, float value, size_t n);
    void tensor_fill_float64(double* data, double value, size_t n);
    void tensor_fill_int8(int8_t* data, int value, size_t n);
    void tensor_fill_int32(int32_t* data, int value, size_t n);
    void tensor_fill_int64(int64_t* data, long long value, size_t n);
    
    // Addition operations
    void tensor_add_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_add_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_add_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_add_int8(int8_t* result, const int8_t* a, const int8_t* b, size_t n);
    void tensor_add_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t n);
    void tensor_add_int64(int64_t* result, const int64_t* a, const int64_t* b, size_t n);
    
    // Multiplication operations
    void tensor_mul_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_mul_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_mul_float64(double* result, const double* a, const double* b, size_t n);
    
    // Scalar multiplication
    void tensor_scalar_mul_float16(half* result, const half* input, float scalar, size_t n);
    void tensor_scalar_mul_float32(float* result, const float* input, float scalar, size_t n);
    void tensor_scalar_mul_float64(double* result, const double* input, double scalar, size_t n);
    
    // Matrix multiplication
    void tensor_matmul_float16(half* C, const half* A, const half* B, size_t M, size_t N, size_t K);
    void tensor_matmul_float32(float* C, const float* A, const float* B, size_t M, size_t N, size_t K);
    void tensor_matmul_float64(double* C, const double* A, const double* B, size_t M, size_t N, size_t K);
    
    // Sum reductions
    float tensor_sum_float16(const half* input, size_t n);
    float tensor_sum_float32(const float* input, size_t n);
    double tensor_sum_float64(const double* input, size_t n);
    long long tensor_sum_int64(const int64_t* input, size_t n);
    
    // Type conversions
    void convert_float32_to_float16(half* output, const float* input, size_t n);
    void convert_float16_to_float32(float* output, const half* input, size_t n);
    void convert_float64_to_float32(float* output, const double* input, size_t n);
    void convert_float32_to_float64(double* output, const float* input, size_t n);
    void convert_int32_to_float32(float* output, const int32_t* input, size_t n);
    void convert_float32_to_int32(int32_t* output, const float* input, size_t n);
}

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

// [[Rcpp::export]]
NumericVector tensor_to_r_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> ptr(tensor_ptr);
        if (!ptr) {
            stop("Invalid tensor pointer");
        }
        
        if (ptr->size() == 0) {
            return NumericVector();
        }
        
        DType dtype = ptr->dtype();
        std::vector<double> host_data(ptr->size());
        
        // Type-specific data copying from GPU to host
        switch (dtype) {
            case DType::FLOAT16: {
                std::vector<half> temp_data(ptr->size());
                ptr->copy_to_host_generic(temp_data.data());
                for (size_t i = 0; i < ptr->size(); ++i) {
                    host_data[i] = static_cast<double>(__half2float(temp_data[i]));
                }
                break;
            }
            case DType::FLOAT32: {
                std::vector<float> temp_data(ptr->size());
                ptr->copy_to_host_generic(temp_data.data());
                for (size_t i = 0; i < ptr->size(); ++i) {
                    host_data[i] = static_cast<double>(temp_data[i]);
                }
                break;
            }
            case DType::FLOAT64: {
                // Direct copy for double
                ptr->copy_to_host_generic(host_data.data());
                break;
            }
            case DType::INT8: {
                std::vector<int8_t> temp_data(ptr->size());
                ptr->copy_to_host_generic(temp_data.data());
                for (size_t i = 0; i < ptr->size(); ++i) {
                    host_data[i] = static_cast<double>(temp_data[i]);
                }
                break;
            }
            case DType::INT32: {
                std::vector<int32_t> temp_data(ptr->size());
                ptr->copy_to_host_generic(temp_data.data());
                for (size_t i = 0; i < ptr->size(); ++i) {
                    host_data[i] = static_cast<double>(temp_data[i]);
                }
                break;
            }
            case DType::INT64: {
                std::vector<int64_t> temp_data(ptr->size());
                ptr->copy_to_host_generic(temp_data.data());
                for (size_t i = 0; i < ptr->size(); ++i) {
                    host_data[i] = static_cast<double>(temp_data[i]);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype for tensor_to_r: " + dtype_to_string(dtype));
        }
        
        NumericVector result(host_data.begin(), host_data.end());
        
        // Set shape as attribute
        IntegerVector shape_attr(ptr->ndims());
        auto shape = ptr->shape();
        for (size_t i = 0; i < ptr->ndims(); ++i) {
            shape_attr[i] = static_cast<int>(shape[i]);
        }
        result.attr("dim") = shape_attr;
        
        return result;
    } catch (const std::exception& e) {
        stop("Error converting tensor to R: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
IntegerVector tensor_shape_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> ptr(tensor_ptr);
        if (!ptr) {
            stop("Invalid tensor pointer");
        }
        
        auto shape = ptr->shape();
        IntegerVector result(shape.ndims());
        for (size_t i = 0; i < shape.ndims(); ++i) {
            result[i] = static_cast<int>(shape[i]);
        }
        return result;
    } catch (const std::exception& e) {
        stop("Error getting tensor shape: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
size_t tensor_size_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> ptr(tensor_ptr);
        if (!ptr) {
            stop("Invalid tensor pointer");
        }
        return ptr->size();
    } catch (const std::exception& e) {
        stop("Error getting tensor size: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
std::string tensor_dtype_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> ptr(tensor_ptr);
        if (!ptr) {
            stop("Invalid tensor pointer");
        }
        return dtype_to_string(ptr->dtype());
    } catch (const std::exception& e) {
        stop("Error getting tensor dtype: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
std::string tensor_info_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> ptr(tensor_ptr);
        if (!ptr) {
            return "Invalid tensor pointer";
        }
        return ptr->info();
    } catch (const std::exception& e) {
        return "Error getting tensor info: " + std::string(e.what());
    }
}

// [[Rcpp::export]]
SEXP tensor_to_dtype_unified(SEXP tensor_ptr, std::string target_dtype) {
    try {
        XPtr<TensorBase> ptr(tensor_ptr);
        if (!ptr) {
            stop("Invalid tensor pointer");
        }
        
        std::unique_ptr<TensorBase> converted_tensor;
        
        if (target_dtype == "float32" || target_dtype == "float") {
            converted_tensor = ptr->to_float();
        } else if (target_dtype == "float64" || target_dtype == "double") {
            converted_tensor = ptr->to_double();
        } else if (target_dtype == "float16" || target_dtype == "half") {
            converted_tensor = ptr->to_half();
        } else if (target_dtype == "bfloat16" || target_dtype == "bf16") {
            converted_tensor = ptr->to_bfloat16();
        } else if (target_dtype == "int8") {
            converted_tensor = ptr->to_int8();
        } else if (target_dtype == "int32" || target_dtype == "int") {
            converted_tensor = ptr->to_int32();
        } else if (target_dtype == "int64" || target_dtype == "long") {
            converted_tensor = ptr->to_int64();
        } else {
            stop("Unsupported target dtype: " + target_dtype);
        }
        
        // Wrap in external pointer
        auto tensor_unique = std::unique_ptr<TensorBase>(converted_tensor.release());
        XPtr<TensorBase> result_ptr(tensor_unique.release(), true);
        
        result_ptr.attr("class") = "gpuTensor";
        result_ptr.attr("dtype") = target_dtype;
        
        return result_ptr;
    } catch (const std::exception& e) {
        stop("Error converting tensor dtype: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_add_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        
        if (!a_tensor || !b_tensor) {
            stop("Invalid tensor pointer(s)");
        }
        
        // Check shapes compatibility
        if (a_tensor->shape() != b_tensor->shape()) {
            if (!a_tensor->shape().broadcastable_with(b_tensor->shape())) {
                stop("Tensor shapes are not broadcastable");
            }
            // TODO: Implement broadcasting
            stop("Broadcasting not yet implemented for unified interface");
        }
        
        // Determine result type
        DType result_type = TypePromotion::get_result_type(*a_tensor, *b_tensor);
        
        // Promote tensors to result type
        auto a_promoted = TypePromotion::promote_tensor(*a_tensor, result_type);
        auto b_promoted = TypePromotion::promote_tensor(*b_tensor, result_type);
        
        // Perform addition using simple type dispatch
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (result_type) {
            case DType::FLOAT16: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<half>*>(a_promoted.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<half>*>(b_promoted.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Type promotion failed for FLOAT16");
                }
                auto result = std::make_shared<gpuTensor<half>>(a_wrapper->tensor().shape());
                tensor_add_float16(result->data(), a_wrapper->tensor().data(), 
                                 b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_promoted.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_promoted.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Type promotion failed for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(a_wrapper->tensor().shape());
                tensor_add_float32(result->data(), a_wrapper->tensor().data(), 
                                 b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_promoted.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_promoted.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Type promotion failed for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(a_wrapper->tensor().shape());
                tensor_add_float64(result->data(), a_wrapper->tensor().data(), 
                                 b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            case DType::INT8: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<int8_t>*>(a_promoted.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<int8_t>*>(b_promoted.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Type promotion failed for INT8");
                }
                auto result = std::make_shared<gpuTensor<int8_t>>(a_wrapper->tensor().shape());
                tensor_add_int8(result->data(), a_wrapper->tensor().data(), 
                              b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<int8_t>>(result);
                break;
            }
            case DType::INT32: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<int32_t>*>(a_promoted.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<int32_t>*>(b_promoted.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Type promotion failed for INT32");
                }
                auto result = std::make_shared<gpuTensor<int32_t>>(a_wrapper->tensor().shape());
                tensor_add_int32(result->data(), a_wrapper->tensor().data(), 
                                b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<int32_t>>(result);
                break;
            }
            case DType::INT64: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<int64_t>*>(a_promoted.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<int64_t>*>(b_promoted.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Type promotion failed for INT64");
                }
                auto result = std::make_shared<gpuTensor<int64_t>>(a_wrapper->tensor().shape());
                tensor_add_int64(result->data(), a_wrapper->tensor().data(), 
                                b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<int64_t>>(result);
                break;
            }
            default:
                throw std::runtime_error("Addition not supported for dtype: " + dtype_to_string(result_type));
        }
        
        // Wrap result in external pointer
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(result_type);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor addition: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_scalar_mul_unified(SEXP tensor_ptr, double scalar) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT16: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<half>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT16");
                }
                auto result = std::make_shared<gpuTensor<half>>(tensor_wrapper->tensor().shape());
                tensor_scalar_mul_float16(result->data(), tensor_wrapper->tensor().data(), 
                                        static_cast<float>(scalar), result->size());
                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(tensor_wrapper->tensor().shape());
                tensor_scalar_mul_float32(result->data(), tensor_wrapper->tensor().data(), 
                                        static_cast<float>(scalar), result->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(tensor_wrapper->tensor().shape());
                tensor_scalar_mul_float64(result->data(), tensor_wrapper->tensor().data(), 
                                        scalar, result->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Scalar multiplication not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor scalar multiplication: " + std::string(e.what()));
    }
}

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
void tensor_synchronize_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> ptr(tensor_ptr);
        if (!ptr) {
            stop("Invalid tensor pointer");
        }
        ptr->synchronize();
    } catch (const std::exception& e) {
        stop("Error synchronizing unified tensor: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
bool tensor_is_contiguous_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> ptr(tensor_ptr);
        if (!ptr) {
            stop("Invalid tensor pointer");
        }
        return ptr->is_contiguous();
    } catch (const std::exception& e) {
        stop("Error checking tensor contiguity: " + std::string(e.what()));
    }
} 