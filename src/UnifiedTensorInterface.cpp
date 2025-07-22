#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// Forward declarations of templated CUDA functions
extern "C" {
    // Addition
    void tensor_add_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_add_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_add_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_add_int8(int8_t* result, const int8_t* a, const int8_t* b, size_t n);
    void tensor_add_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t n);
    void tensor_add_int64(int64_t* result, const int64_t* a, const int64_t* b, size_t n);

    // Multiplication
    void tensor_mul_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_mul_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_mul_float64(double* result, const double* a, const double* b, size_t n);

    // Subtraction
    void tensor_sub_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_sub_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_sub_float64(double* result, const double* a, const double* b, size_t n);

    // Division
    void tensor_div_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_div_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_div_float64(double* result, const double* a, const double* b, size_t n);

    // Scalar multiplication
    void tensor_scalar_mul_float16(half* result, const half* input, float scalar, size_t n);
    void tensor_scalar_mul_float32(float* result, const float* input, float scalar, size_t n);
    void tensor_scalar_mul_float64(double* result, const double* input, double scalar, size_t n);

    // Scalar addition
    void tensor_scalar_add_float16(half* result, const half* input, float scalar, size_t n);
    void tensor_scalar_add_float32(float* result, const float* input, float scalar, size_t n);
    void tensor_scalar_add_float64(double* result, const double* input, double scalar, size_t n);

    // Matrix multiplication
    void tensor_matmul_float16(half* C, const half* A, const half* B, size_t M, size_t N, size_t K);
    void tensor_matmul_float32(float* C, const float* A, const float* B, size_t M, size_t N, size_t K);
    void tensor_matmul_float64(double* C, const double* A, const double* B, size_t M, size_t N, size_t K);

    // Sum reductions
    float tensor_sum_float16(const half* input, size_t n);
    float tensor_sum_float32(const float* input, size_t n);
    double tensor_sum_float64(const double* input, size_t n);
    int64_t tensor_sum_int64(const int64_t* input, size_t n);
    
    // Unary math operations
    void tensor_exp_float32(float* result, const float* input, size_t n);
    void tensor_exp_float64(double* result, const double* input, size_t n);
    void tensor_log_float32(float* result, const float* input, size_t n);
    void tensor_log_float64(double* result, const double* input, size_t n);
    void tensor_sqrt_float32(float* result, const float* input, size_t n);
    void tensor_sqrt_float64(double* result, const double* input, size_t n);
    
    // Reduction operations
    float tensor_max_float32(const float* input, size_t n);
    double tensor_max_float64(const double* input, size_t n);
    float tensor_min_float32(const float* input, size_t n);
    double tensor_min_float64(const double* input, size_t n);

    // Type conversions
    void convert_float32_to_float16(half* output, const float* input, size_t n);
    void convert_float16_to_float32(float* output, const half* input, size_t n);
    void convert_float64_to_float32(float* output, const double* input, size_t n);
    void convert_float32_to_float64(double* output, const float* input, size_t n);
    void convert_int32_to_float32(float* output, const int32_t* input, size_t n);
    void convert_float32_to_int32(int32_t* output, const float* input, size_t n);

    // Broadcast operations
    void tensor_add_broadcast_float32(
        float* result, const float* a, const float* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_add_broadcast_float64(
        double* result, const double* a, const double* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_mul_broadcast_float32(
        float* result, const float* a, const float* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_mul_broadcast_float64(
        double* result, const double* a, const double* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );

    // Strided copy operations
    void tensor_strided_copy_float32(float* dest, const float* src, const int* strides, const int* shape, int ndims, size_t total_elements);
    void tensor_strided_copy_float64(double* dest, const double* src, const int* strides, const int* shape, int ndims, size_t total_elements);
}

// Helper to compute broadcast strides
std::vector<int> compute_broadcast_strides(const Shape& tensor_shape, const Shape& broadcast_shape) {
    std::vector<int> strides(broadcast_shape.ndims(), 0);
    int tensor_dim = tensor_shape.ndims() - 1;
    
    for (int i = broadcast_shape.ndims() - 1; i >= 0; i--) {
        if (tensor_dim >= 0 && tensor_shape[tensor_dim] == broadcast_shape[i]) {
            // Normal dimension - compute stride
            int stride = 1;
            for (int j = tensor_dim + 1; j < (int)tensor_shape.ndims(); j++) {
                stride *= tensor_shape[j];
            }
            strides[i] = stride;
            tensor_dim--;
        } else if (tensor_dim >= 0 && tensor_shape[tensor_dim] == 1) {
            // Size-1 dimension broadcasts - stride is 0
            strides[i] = 0;
            tensor_dim--;
        } else if (tensor_dim < 0) {
            // Prepended dimension (size 1) - stride is 0
            strides[i] = 0;
        } else {
            throw std::runtime_error("Invalid broadcast - dimension mismatch");
        }
    }
    
    return strides;
}

// Forward declarations for functions defined later in this file (needed for calls inside other functions)
SEXP tensor_scalar_add_unified(SEXP tensor_ptr, double scalar);
SEXP tensor_scalar_mul_unified(SEXP tensor_ptr, double scalar); // already present but declare for clarity

template<typename KernelFunc, typename WrapperType, typename ScalarT>
std::unique_ptr<TensorBase> binary_elementwise_execute(const TensorBase& a_t, const TensorBase& b_t,
                                                       KernelFunc kernel) {
    const auto* a_wrap = dynamic_cast<const WrapperType*>(&a_t);
    const auto* b_wrap = dynamic_cast<const WrapperType*>(&b_t);
    if (!a_wrap || !b_wrap) {
        throw std::runtime_error("Type promotion wrapper mismatch");
    }

    // Get references to the original tensors
    const gpuTensor<ScalarT>& a_tensor = a_wrap->tensor();
    const gpuTensor<ScalarT>& b_tensor = b_wrap->tensor();

    // Ensure both tensors are contiguous to guarantee correct element order
    gpuTensor<ScalarT> a_contiguous = a_tensor.is_contiguous() ? a_tensor : a_tensor.contiguous();
    gpuTensor<ScalarT> b_contiguous = b_tensor.is_contiguous() ? b_tensor : b_tensor.contiguous();

    // Allocate result tensor
    auto result = std::make_shared<gpuTensor<ScalarT>>(a_contiguous.shape());

    // Launch CUDA kernel (expects contiguous memory)
    kernel(result->data(), a_contiguous.data(), b_contiguous.data(), result->size());

    return std::make_unique<TensorWrapper<ScalarT>>(result);
}

// helper macro to reduce duplication
#define HANDLE_BINARY_OP(dtype_enum, scalar_t, func_name) \
    case dtype_enum: { \
        result_tensor = binary_elementwise_execute<decltype(func_name), TensorWrapper<scalar_t>, scalar_t>(*a_promoted, *b_promoted, func_name); \
        break; }

static SEXP tensor_binary_template(SEXP a_ptr, SEXP b_ptr,
                                   std::function<void*(void)> dummy, // just for overload resolution
                                   bool is_sub) {
    XPtr<TensorBase> a_tensor(a_ptr);
    XPtr<TensorBase> b_tensor(b_ptr);

    if (!a_tensor || !b_tensor) stop("Invalid tensor pointer(s)");

    if (a_tensor->shape() != b_tensor->shape()) {
        stop("Shapes must match for this implementation");
    }

    DType res_type = TypePromotion::get_result_type(*a_tensor, *b_tensor);
    auto a_promoted = TypePromotion::promote_tensor(*a_tensor, res_type);
    auto b_promoted = TypePromotion::promote_tensor(*b_tensor, res_type);

    std::unique_ptr<TensorBase> result_tensor;

    switch (res_type) {
        HANDLE_BINARY_OP(DType::FLOAT16, half, is_sub ? tensor_sub_float16 : tensor_div_float16)
        HANDLE_BINARY_OP(DType::FLOAT32, float, is_sub ? tensor_sub_float32 : tensor_div_float32)
        HANDLE_BINARY_OP(DType::FLOAT64, double, is_sub ? tensor_sub_float64 : tensor_div_float64)
        default:
            stop("Operation not implemented for dtype: " + dtype_to_string(res_type));
    }

    auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
    XPtr<TensorBase> ptr(uniq.release(), true);
    ptr.attr("class") = "gpuTensor";
    ptr.attr("dtype") = dtype_to_string(res_type);
    return ptr;
}

// [[Rcpp::export]]
SEXP tensor_sub_unified(SEXP a_ptr, SEXP b_ptr) {
    return tensor_binary_template(a_ptr, b_ptr, [](){return nullptr;}, true);
}

// [[Rcpp::export]]
SEXP tensor_div_unified(SEXP a_ptr, SEXP b_ptr) {
    return tensor_binary_template(a_ptr, b_ptr, [](){return nullptr;}, false);
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
        bool shapes_equal = (a_tensor->shape() == b_tensor->shape());
        bool scalar_broadcast = (!shapes_equal) && (a_tensor->size() == 1 || b_tensor->size() == 1);
        bool needs_full_broadcast = !shapes_equal && !scalar_broadcast;

        if (needs_full_broadcast) {
            // Check if shapes are broadcastable using existing Shape methods
            if (!a_tensor->shape().broadcastable_with(b_tensor->shape())) {
                stop("Tensor shapes are not broadcastable");
            }
            
            // Compute broadcast shape and use broadcast kernel
            Shape broadcast_shape = a_tensor->shape().broadcast_with(b_tensor->shape());
            
            // Create result tensor - we need to use the existing pattern for tensor creation
            DType dtype_a = a_tensor->dtype();
            DType dtype_b = b_tensor->dtype();
            
            if (dtype_a != dtype_b) {
                stop("Cannot add tensors with different dtypes");
            }
            
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
            
            if (dtype_a == DType::FLOAT32) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(broadcast_shape);
                tensor_add_broadcast_float32(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
            } else if (dtype_a == DType::FLOAT64) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(broadcast_shape);
                tensor_add_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast addition only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype_a);
            
            return ptr;
        }

        // Handle tensor + scalar_tensor case by converting scalar operand to numeric and using scalar kernel
        if (scalar_broadcast) {
            const TensorBase* scalar_tensor = (a_tensor->size() == 1) ? a_tensor.get() : b_tensor.get();
            const TensorBase* other_tensor  = (a_tensor->size() == 1) ? b_tensor.get() : a_tensor.get();

            // Copy scalar to host (as double)
            double scalar_value = 0.0;
            scalar_tensor->copy_to_host_generic(&scalar_value);

            // Recurse to scalar addition kernel
            SEXP other_ptr = (a_tensor->size() == 1) ? b_ptr : a_ptr;
            return tensor_scalar_add_unified(other_ptr, scalar_value);
        }
        
        // Fast path for equal shapes - use existing type promotion logic
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

// [[Rcpp::export]]
SEXP tensor_mul_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);

        if (!a_tensor || !b_tensor) {
            stop("Invalid tensor pointer(s)");
        }

        // Check shapes compatibility
        bool shapes_equal = (a_tensor->shape() == b_tensor->shape());
        bool scalar_broadcast = (!shapes_equal) && (a_tensor->size() == 1 || b_tensor->size() == 1);
        bool needs_full_broadcast = !shapes_equal && !scalar_broadcast;

        if (needs_full_broadcast) {
            // Check if shapes are broadcastable using existing Shape methods
            if (!a_tensor->shape().broadcastable_with(b_tensor->shape())) {
                stop("Tensor shapes are not broadcastable");
            }
            
            // Compute broadcast shape and use broadcast kernel
            Shape broadcast_shape = a_tensor->shape().broadcast_with(b_tensor->shape());
            
            // Create result tensor
            DType dtype_a = a_tensor->dtype();
            DType dtype_b = b_tensor->dtype();
            
            if (dtype_a != dtype_b) {
                stop("Cannot multiply tensors with different dtypes");
            }
            
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
            
            if (dtype_a == DType::FLOAT32) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(broadcast_shape);
                tensor_mul_broadcast_float32(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
            } else if (dtype_a == DType::FLOAT64) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(broadcast_shape);
                tensor_mul_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast multiplication only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype_a);
            
            return ptr;
        }

        if (scalar_broadcast) {
            const TensorBase* scalar_tensor = (a_tensor->size() == 1) ? a_tensor.get() : b_tensor.get();
            const TensorBase* other_tensor  = (a_tensor->size() == 1) ? b_tensor.get() : a_tensor.get();

            double scalar_val = 0.0;
            scalar_tensor->copy_to_host_generic(&scalar_val);

            SEXP other_ptr = (a_tensor->size() == 1) ? b_ptr : a_ptr;
            return tensor_scalar_mul_unified(other_ptr, scalar_val);
        }
        
        // Fast path for equal shapes
        DType dtype_a = a_tensor->dtype();
        DType dtype_b = b_tensor->dtype();
        
        if (dtype_a != dtype_b) {
            stop("Cannot multiply tensors with different dtypes");
        }
        
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype_a) {
            case DType::FLOAT16: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<half>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<half>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT16");
                }
                auto result = std::make_shared<gpuTensor<half>>(a_wrapper->tensor().shape());
                tensor_mul_float16(result->data(), a_wrapper->tensor().data(), 
                                 b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(a_wrapper->tensor().shape());
                tensor_mul_float32(result->data(), a_wrapper->tensor().data(), 
                                 b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrapper = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT64");
                }
                auto result = std::make_shared<gpuTensor<double>>(a_wrapper->tensor().shape());
                tensor_mul_float64(result->data(), a_wrapper->tensor().data(), 
                                 b_wrapper->tensor().data(), result->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Element-wise multiplication not implemented for dtype: " + dtype_to_string(dtype_a));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype_a);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor element-wise multiplication: " + std::string(e.what()));
    }
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
                tensor_matmul_float16(result->data(), a_wrapper->tensor().data(), 
                                    b_wrapper->tensor().data(), M, N, K);
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
                tensor_matmul_float32(result->data(), a_wrapper->tensor().data(), 
                                    b_wrapper->tensor().data(), M, N, K);
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
                tensor_matmul_float64(result->data(), a_wrapper->tensor().data(), 
                                    b_wrapper->tensor().data(), M, N, K);
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
SEXP tensor_scalar_add_unified(SEXP tensor_ptr, double scalar) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        if (!tensor) {
            stop("Invalid tensor pointer");
        }

        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;

        switch (dtype) {
            case DType::FLOAT16: {
                auto tw = dynamic_cast<const TensorWrapper<half>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT16");
                auto result = std::make_shared<gpuTensor<half>>(tw->tensor().shape());
                tensor_scalar_add_float16(result->data(), tw->tensor().data(), static_cast<float>(scalar), result->size());
                result_tensor = std::make_unique<TensorWrapper<half>>(result);
                break;
            }
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto result = std::make_shared<gpuTensor<float>>(tw->tensor().shape());
                tensor_scalar_add_float32(result->data(), tw->tensor().data(), static_cast<float>(scalar), result->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto result = std::make_shared<gpuTensor<double>>(tw->tensor().shape());
                tensor_scalar_add_float64(result->data(), tw->tensor().data(), scalar, result->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Scalar addition not implemented for dtype: " + dtype_to_string(dtype));
        }

        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor scalar addition: " + std::string(e.what()));
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

// [[Rcpp::export]]
SEXP tensor_exp_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();
                
                auto result_gpu = gpuTensor<float>(input_tensor.shape(), input_tensor.device());
                tensor_exp_float32(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<float>>(
                    std::make_shared<gpuTensor<float>>(std::move(result_gpu))
                );
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();
                
                auto result_gpu = gpuTensor<double>(input_tensor.shape(), input_tensor.device());
                tensor_exp_float64(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<double>>(
                    std::make_shared<gpuTensor<double>>(std::move(result_gpu))
                );
                break;
            }
            default:
                stop("Exp not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor exp: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_log_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();
                
                auto result_gpu = gpuTensor<float>(input_tensor.shape(), input_tensor.device());
                tensor_log_float32(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<float>>(
                    std::make_shared<gpuTensor<float>>(std::move(result_gpu))
                );
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();
                
                auto result_gpu = gpuTensor<double>(input_tensor.shape(), input_tensor.device());
                tensor_log_float64(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<double>>(
                    std::make_shared<gpuTensor<double>>(std::move(result_gpu))
                );
                break;
            }
            default:
                stop("Log not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor log: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_sqrt_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();
                
                auto result_gpu = gpuTensor<float>(input_tensor.shape(), input_tensor.device());
                tensor_sqrt_float32(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<float>>(
                    std::make_shared<gpuTensor<float>>(std::move(result_gpu))
                );
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();
                
                auto result_gpu = gpuTensor<double>(input_tensor.shape(), input_tensor.device());
                tensor_sqrt_float64(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<double>>(
                    std::make_shared<gpuTensor<double>>(std::move(result_gpu))
                );
                break;
            }
            default:
                stop("Sqrt not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor sqrt: " + std::string(e.what()));
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