#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

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
        
        // Set shape as attribute (only for non-scalar tensors)
        if (ptr->ndims() > 0) {
            IntegerVector shape_attr(ptr->ndims());
            auto shape = ptr->shape();
            for (size_t i = 0; i < ptr->ndims(); ++i) {
                shape_attr[i] = static_cast<int>(shape[i]);
            }
            result.attr("dim") = shape_attr;
        }
        
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