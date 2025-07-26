#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>
#include <functional>

using namespace Rcpp;

// Forward declarations of templated CUDA functions
extern "C" {
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

    // Subtraction operations
    void tensor_sub_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_sub_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_sub_float64(double* result, const double* a, const double* b, size_t n);

    // Division operations
    void tensor_div_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_div_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_div_float64(double* result, const double* a, const double* b, size_t n);

    // New binary element-wise operations for Phase 3.2
    void tensor_max_elemwise_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_max_elemwise_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_min_elemwise_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_min_elemwise_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_pow_elemwise_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_pow_elemwise_float64(double* result, const double* a, const double* b, size_t n);
    
    // Strided versions of new operations
    void tensor_max_elemwise_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                             const cuda_utils::TensorDescriptor& a_desc,
                                             const cuda_utils::TensorDescriptor& b_desc);
    void tensor_max_elemwise_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                             const cuda_utils::TensorDescriptor& a_desc,
                                             const cuda_utils::TensorDescriptor& b_desc);
    void tensor_min_elemwise_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                             const cuda_utils::TensorDescriptor& a_desc,
                                             const cuda_utils::TensorDescriptor& b_desc);
    void tensor_min_elemwise_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                             const cuda_utils::TensorDescriptor& a_desc,
                                             const cuda_utils::TensorDescriptor& b_desc);
    void tensor_pow_elemwise_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                             const cuda_utils::TensorDescriptor& a_desc,
                                             const cuda_utils::TensorDescriptor& b_desc);
    void tensor_pow_elemwise_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                             const cuda_utils::TensorDescriptor& a_desc,
                                             const cuda_utils::TensorDescriptor& b_desc);

    // Scalar operations
    void tensor_scalar_mul_float16(half* result, const half* input, float scalar, size_t n);
    void tensor_scalar_mul_float32(float* result, const float* input, float scalar, size_t n);
    void tensor_scalar_mul_float64(double* result, const double* input, double scalar, size_t n);
    void tensor_scalar_add_float16(half* result, const half* input, float scalar, size_t n);
    void tensor_scalar_add_float32(float* result, const float* input, float scalar, size_t n);
    void tensor_scalar_add_float64(double* result, const double* input, double scalar, size_t n);

    // New scalar operations for Phase 3.2
    void tensor_scalar_sub_float32(float* result, const float* input, float scalar, size_t n);
    void tensor_scalar_sub_float64(double* result, const double* input, double scalar, size_t n);
    void tensor_scalar_div_float32(float* result, const float* input, float scalar, size_t n);
    void tensor_scalar_div_float64(double* result, const double* input, double scalar, size_t n);

    // Strided operations for non-contiguous tensors
    void tensor_add_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_add_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_add_strided_float16(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_add_strided_int8(const cuda_utils::TensorDescriptor& out_desc,
                                const cuda_utils::TensorDescriptor& a_desc,
                                const cuda_utils::TensorDescriptor& b_desc);
    void tensor_mul_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_mul_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_mul_strided_float16(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_sub_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_sub_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_div_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_div_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_sub_strided_float16(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_div_strided_float16(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);

    // Strided scalar operations for non-contiguous tensors
    void tensor_scalar_add_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                           const cuda_utils::TensorDescriptor& in_desc,
                                           float scalar);
    void tensor_scalar_add_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                           const cuda_utils::TensorDescriptor& in_desc,
                                           double scalar);
    void tensor_scalar_mul_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                           const cuda_utils::TensorDescriptor& in_desc,
                                           float scalar);
    void tensor_scalar_mul_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                           const cuda_utils::TensorDescriptor& in_desc,
                                           double scalar);

    // New strided scalar operations for Phase 3.2
    void tensor_scalar_sub_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                           const cuda_utils::TensorDescriptor& in_desc,
                                           float scalar);
    void tensor_scalar_sub_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                           const cuda_utils::TensorDescriptor& in_desc,
                                           double scalar);
    void tensor_scalar_div_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                           const cuda_utils::TensorDescriptor& in_desc,
                                           float scalar);
    void tensor_scalar_div_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                           const cuda_utils::TensorDescriptor& in_desc,
                                           double scalar);

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
    void tensor_sub_broadcast_float32(
        float* result, const float* a, const float* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_sub_broadcast_float64(
        double* result, const double* a, const double* b,
        const int* a_strides, const int* b_strides, const int* result_strides,  
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_div_broadcast_float32(
        float* result, const float* a, const float* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_div_broadcast_float64(
        double* result, const double* a, const double* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_max_broadcast_float32(
        float* result, const float* a, const float* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_max_broadcast_float64(
        double* result, const double* a, const double* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_pow_broadcast_float32(
        float* result, const float* a, const float* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_pow_broadcast_float64(
        double* result, const double* a, const double* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_min_broadcast_float32(
        float* result, const float* a, const float* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
    void tensor_min_broadcast_float64(
        double* result, const double* a, const double* b,
        const int* a_strides, const int* b_strides, const int* result_strides,
        const int* shape, int ndims, size_t total_elements
    );
}

// Helper to compute broadcast strides
std::vector<int> compute_broadcast_strides(const Shape& tensor_shape, const Shape& broadcast_shape) {
     // Column-major: stride of first (fastest) dimension is 1 ----------------
    std::vector<int> strides(broadcast_shape.ndims(), 0);
 
     // Compute base strides for the original tensor (column-major) -----------
     std::vector<int> tensor_strides(tensor_shape.ndims());
     if (!tensor_strides.empty()) {
         tensor_strides[0] = 1;
         for (size_t i = 1; i < tensor_shape.ndims(); ++i) {
             tensor_strides[i] = tensor_strides[i - 1] * static_cast<int>(tensor_shape[i - 1]);
         }
     }

    size_t src_len = tensor_shape.ndims();
    size_t dst_len = broadcast_shape.ndims();

    // Try aligning tensor_shape at every possible offset inside broadcast_shape
    bool compatible = false;
    size_t chosen_offset = 0;
    for (size_t offset = 0; offset <= dst_len - src_len; ++offset) {
        bool ok = true;
        for (size_t i = 0; i < src_len; ++i) {
            size_t src_dim = tensor_shape[i];
            size_t dst_dim = broadcast_shape[i + offset];
            if (src_dim != dst_dim && src_dim != 1 && dst_dim != 1) {
                ok = false;
                break;
            }
        }
        if (ok) {
            compatible = true;
            chosen_offset = offset;
            break;
        }
    }

    if (!compatible) {
        throw std::runtime_error("Shape mismatch in broadcast stride calculation");
    }

    // Populate strides with chosen offset
    for (size_t i = 0; i < dst_len; ++i) {
        if (i < chosen_offset || i >= chosen_offset + src_len) {
            strides[i] = 0; // implicit dimension from broadcasting
        } else {
            size_t src_idx = i - chosen_offset;
            if (tensor_shape[src_idx] == 1) {
                strides[i] = 0;
            } else {
                strides[i] = tensor_strides[src_idx];
            }
        }
    }

    return strides;
}

// Forward declarations for functions defined later (needed for calls inside other functions)
SEXP tensor_scalar_add_unified(SEXP tensor_ptr, double scalar);
SEXP tensor_scalar_mul_unified(SEXP tensor_ptr, double scalar);
SEXP tensor_scalar_sub_unified(SEXP tensor_ptr, double scalar);
SEXP tensor_scalar_div_unified(SEXP tensor_ptr, double scalar);

template<typename ContiguousKernelFunc, typename StridedKernelFunc, typename WrapperType, typename ScalarT>
std::unique_ptr<TensorBase> binary_elementwise_execute_stride_aware(const TensorBase& a_t, const TensorBase& b_t,
                                                                   ContiguousKernelFunc contiguous_kernel,
                                                                   StridedKernelFunc strided_kernel) {
    const auto* a_wrap = dynamic_cast<const WrapperType*>(&a_t);
    const auto* b_wrap = dynamic_cast<const WrapperType*>(&b_t);
    if (!a_wrap || !b_wrap) {
        throw std::runtime_error("Type promotion wrapper mismatch");
    }

    // Get references to the original tensors
    const gpuTensor<ScalarT>& a_tensor = a_wrap->tensor();
    const gpuTensor<ScalarT>& b_tensor = b_wrap->tensor();

    // Allocate result tensor (always contiguous)
    auto result = std::make_shared<gpuTensor<ScalarT>>(a_tensor.shape());

    // Check if both tensors are contiguous - if so, use faster contiguous kernel
    if (a_tensor.is_contiguous() && b_tensor.is_contiguous()) {
        // Launch fast contiguous CUDA kernel
        contiguous_kernel(result->data(), a_tensor.data(), b_tensor.data(), result->size());
    } else {
        // Use stride-aware kernel for non-contiguous tensors
        auto a_desc = a_tensor.descriptor();
        auto b_desc = b_tensor.descriptor();
        auto out_desc = result->descriptor();
        
        strided_kernel(out_desc, a_desc, b_desc);
    }

    return std::make_unique<TensorWrapper<ScalarT>>(result);
}

// helper macro to reduce duplication
#define HANDLE_BINARY_OP(dtype_enum, scalar_t, contiguous_func, strided_func) \
    case dtype_enum: { \
        result_tensor = binary_elementwise_execute_stride_aware<decltype(contiguous_func), decltype(strided_func), TensorWrapper<scalar_t>, scalar_t>(*a_promoted, *b_promoted, contiguous_func, strided_func); \
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
        HANDLE_BINARY_OP(DType::FLOAT16, half, 
                         is_sub ? tensor_sub_float16 : tensor_div_float16,
                         is_sub ? tensor_sub_strided_float16 : tensor_div_strided_float16)
        HANDLE_BINARY_OP(DType::FLOAT32, float, 
                         is_sub ? tensor_sub_float32 : tensor_div_float32,
                         is_sub ? tensor_sub_strided_float32 : tensor_div_strided_float32)
        HANDLE_BINARY_OP(DType::FLOAT64, double, 
                         is_sub ? tensor_sub_float64 : tensor_div_float64,
                         is_sub ? tensor_sub_strided_float64 : tensor_div_strided_float64)
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
                stop("Cannot subtract tensors with different dtypes");
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
                tensor_sub_broadcast_float32(
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
                tensor_sub_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast subtraction only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype_a);
            
            return ptr;
        }

        // Handle tensor - scalar_tensor case by converting scalar operand to numeric and using scalar kernel
        if (scalar_broadcast) {
            const TensorBase* scalar_tensor = (a_tensor->size() == 1) ? a_tensor.get() : b_tensor.get();
            const TensorBase* other_tensor  = (a_tensor->size() == 1) ? b_tensor.get() : a_tensor.get();

            // Copy scalar to host (as double)
            double scalar_value = 0.0;
            scalar_tensor->copy_to_host_generic(&scalar_value);

            // Recurse to scalar subtraction kernel
            SEXP other_ptr = (a_tensor->size() == 1) ? b_ptr : a_ptr;
            if (a_tensor->size() == 1) {
                // scalar - tensor: need to negate and add
                auto neg_other = tensor_scalar_mul_unified(other_ptr, -1.0);
                return tensor_scalar_add_unified(neg_other, scalar_value);
            } else {
                // tensor - scalar: direct subtraction
                return tensor_scalar_sub_unified(other_ptr, scalar_value);
            }
        }
        
        // Fast path for equal shapes - use existing type promotion logic
        return tensor_binary_template(a_ptr, b_ptr, [](){return nullptr;}, true);
        
    } catch (const std::exception& e) {
        stop("Error in unified tensor subtraction: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_div_unified(SEXP a_ptr, SEXP b_ptr) {
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
                stop("Cannot divide tensors with different dtypes");
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
                tensor_div_broadcast_float32(
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
                tensor_div_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast division only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype_a);
            
            return ptr;
        }

        // Handle tensor / scalar_tensor case by converting scalar operand to numeric and using scalar kernel
        if (scalar_broadcast) {
            const TensorBase* scalar_tensor = (a_tensor->size() == 1) ? a_tensor.get() : b_tensor.get();
            const TensorBase* other_tensor  = (a_tensor->size() == 1) ? b_tensor.get() : a_tensor.get();

            // Copy scalar to host (as double)
            double scalar_value = 0.0;
            scalar_tensor->copy_to_host_generic(&scalar_value);

            // Recurse to scalar division kernel
            SEXP other_ptr = (a_tensor->size() == 1) ? b_ptr : a_ptr;
            if (a_tensor->size() == 1) {
                // scalar / tensor: need to compute scalar * (1/tensor)
                // For now, fallback to error - scalar/tensor broadcast is complex
                stop("Scalar / tensor broadcasting not yet implemented");
             } else {
                 // tensor / scalar: direct division
                 return tensor_scalar_div_unified(other_ptr, scalar_value);
             }
        }
        
        // Fast path for equal shapes - use existing type promotion logic
        return tensor_binary_template(a_ptr, b_ptr, [](){return nullptr;}, false);
        
    } catch (const std::exception& e) {
        stop("Error in unified tensor division: " + std::string(e.what()));
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
                
                // Choose kernel based on contiguity
                if (a_wrapper->tensor().is_contiguous() && b_wrapper->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_add_float16(result->data(), a_wrapper->tensor().data(), 
                                     b_wrapper->tensor().data(), result->size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto a_desc = a_wrapper->tensor().descriptor();
                    auto b_desc = b_wrapper->tensor().descriptor();
                    tensor_add_strided_float16(out_desc, a_desc, b_desc);
                }
                
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
                
                // Choose kernel based on contiguity
                if (a_wrapper->tensor().is_contiguous() && b_wrapper->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_add_float32(result->data(), a_wrapper->tensor().data(), 
                                       b_wrapper->tensor().data(), result->size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto a_desc = a_wrapper->tensor().descriptor();
                    auto b_desc = b_wrapper->tensor().descriptor();
                    tensor_add_strided_float32(out_desc, a_desc, b_desc);
                }
                
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
                
                // Choose kernel based on contiguity
                if (a_wrapper->tensor().is_contiguous() && b_wrapper->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_add_float64(result->data(), a_wrapper->tensor().data(), 
                                       b_wrapper->tensor().data(), result->size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto a_desc = a_wrapper->tensor().descriptor();
                    auto b_desc = b_wrapper->tensor().descriptor();
                    tensor_add_strided_float64(out_desc, a_desc, b_desc);
                }
                
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
                
                // Choose kernel based on contiguity
                if (a_wrapper->tensor().is_contiguous() && b_wrapper->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_add_int8(result->data(), a_wrapper->tensor().data(), 
                                  b_wrapper->tensor().data(), result->size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto a_desc = a_wrapper->tensor().descriptor();
                    auto b_desc = b_wrapper->tensor().descriptor();
                    tensor_add_strided_int8(out_desc, a_desc, b_desc);
                }
                
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
                
                // Choose kernel based on contiguity
                if (a_wrapper->tensor().is_contiguous() && b_wrapper->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_mul_float16(result->data(), a_wrapper->tensor().data(), 
                                     b_wrapper->tensor().data(), result->size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto a_desc = a_wrapper->tensor().descriptor();
                    auto b_desc = b_wrapper->tensor().descriptor();
                    tensor_mul_strided_float16(out_desc, a_desc, b_desc);
                }
                
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
                
                // Choose kernel based on contiguity
                if (a_wrapper->tensor().is_contiguous() && b_wrapper->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_mul_float32(result->data(), a_wrapper->tensor().data(), 
                                       b_wrapper->tensor().data(), result->size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto a_desc = a_wrapper->tensor().descriptor();
                    auto b_desc = b_wrapper->tensor().descriptor();
                    tensor_mul_strided_float32(out_desc, a_desc, b_desc);
                }
                
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
                
                // Choose kernel based on contiguity
                if (a_wrapper->tensor().is_contiguous() && b_wrapper->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_mul_float64(result->data(), a_wrapper->tensor().data(), 
                                       b_wrapper->tensor().data(), result->size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto a_desc = a_wrapper->tensor().descriptor();
                    auto b_desc = b_wrapper->tensor().descriptor();
                    tensor_mul_strided_float64(out_desc, a_desc, b_desc);
                }
                
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
                
                const gpuTensor<float>& input_tensor = tensor_wrapper->tensor();
                auto result = std::make_shared<gpuTensor<float>>(input_tensor.shape());
                
                // Use stride-aware kernels for efficiency - NO contiguous forcing!
                if (input_tensor.is_contiguous()) {
                    tensor_scalar_mul_float32(result->data(), input_tensor.data(), 
                                            static_cast<float>(scalar), result->size());
                } else {
                    // Use strided kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto in_desc = input_tensor.descriptor();
                    tensor_scalar_mul_strided_float32(out_desc, in_desc, static_cast<float>(scalar));
                }
                
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto tensor_wrapper = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tensor_wrapper) {
                    throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                }
                
                const gpuTensor<double>& input_tensor = tensor_wrapper->tensor();
                auto result = std::make_shared<gpuTensor<double>>(input_tensor.shape());
                
                // Use stride-aware kernels for efficiency - NO contiguous forcing!
                if (input_tensor.is_contiguous()) {
                    tensor_scalar_mul_float64(result->data(), input_tensor.data(), scalar, result->size());
                } else {
                    // Use strided kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto in_desc = input_tensor.descriptor();
                    tensor_scalar_mul_strided_float64(out_desc, in_desc, scalar);
                }
                
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
                
                const gpuTensor<float>& input_tensor = tw->tensor();
                auto result = std::make_shared<gpuTensor<float>>(input_tensor.shape());
                
                // Use stride-aware kernels for efficiency - NO contiguous forcing!
                if (input_tensor.is_contiguous()) {
                    tensor_scalar_add_float32(result->data(), input_tensor.data(), static_cast<float>(scalar), result->size());
                } else {
                    // Use strided kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto in_desc = input_tensor.descriptor();
                    tensor_scalar_add_strided_float32(out_desc, in_desc, static_cast<float>(scalar));
                }
                
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                const gpuTensor<double>& input_tensor = tw->tensor();
                auto result = std::make_shared<gpuTensor<double>>(input_tensor.shape());
                
                // Use stride-aware kernels for efficiency - NO contiguous forcing!
                if (input_tensor.is_contiguous()) {
                    tensor_scalar_add_float64(result->data(), input_tensor.data(), scalar, result->size());
                } else {
                    // Use strided kernel for non-contiguous tensors
                    auto out_desc = result->descriptor();
                    auto in_desc = input_tensor.descriptor();
                    tensor_scalar_add_strided_float64(out_desc, in_desc, scalar);
                }
                
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

// New binary element-wise operations for Phase 3.2

// [[Rcpp::export]]
SEXP tensor_max_elemwise_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);

        if (!a_tensor || !b_tensor) stop("Invalid tensor pointer(s)");

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
                stop("Cannot compute element-wise max of tensors with different dtypes");
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
                tensor_max_broadcast_float32(
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
                tensor_max_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast element-wise max only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype_a);
            
            return ptr;
        }

        // Handle scalar broadcast case (fallback to element-wise for now)
        if (scalar_broadcast) {
            // For now, fall back to shape check - scalar max broadcasting is less common
            stop("Scalar broadcasting not yet implemented for element-wise max");
        }
        
        // Fast path for equal shapes - use existing implementation
        DType dtype = a_tensor->dtype();
        if (dtype != b_tensor->dtype()) {
            stop("Tensors must have the same dtype");
        }

        std::unique_ptr<TensorBase> result_tensor;

        switch (dtype) {
            case DType::FLOAT32: {
                auto a_tw = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_tw = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_tw || !b_tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");

                auto result = std::make_shared<gpuTensor<float>>(a_tw->tensor().shape());

                if (a_tw->tensor().is_contiguous() && b_tw->tensor().is_contiguous()) {
                    tensor_max_elemwise_float32(result->data(), a_tw->tensor().data(), b_tw->tensor().data(), result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto a_desc = a_tw->tensor().descriptor();
                    auto b_desc = b_tw->tensor().descriptor();
                    tensor_max_elemwise_strided_float32(out_desc, a_desc, b_desc);
                }

                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_tw = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_tw = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_tw || !b_tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");

                auto result = std::make_shared<gpuTensor<double>>(a_tw->tensor().shape());

                if (a_tw->tensor().is_contiguous() && b_tw->tensor().is_contiguous()) {
                    tensor_max_elemwise_float64(result->data(), a_tw->tensor().data(), b_tw->tensor().data(), result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto a_desc = a_tw->tensor().descriptor();
                    auto b_desc = b_tw->tensor().descriptor();
                    tensor_max_elemwise_strided_float64(out_desc, a_desc, b_desc);
                }

                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Element-wise max not implemented for dtype: " + dtype_to_string(dtype));
        }

        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor element-wise max: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_min_elemwise_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);

        if (!a_tensor || !b_tensor) stop("Invalid tensor pointer(s)");

        // Check shapes compatibility
        bool shapes_equal = (a_tensor->shape() == b_tensor->shape());
        bool scalar_broadcast = (!shapes_equal) && (a_tensor->size() == 1 || b_tensor->size() == 1);
        bool needs_full_broadcast = !shapes_equal && !scalar_broadcast;

        if (needs_full_broadcast) {
            // Check if shapes are broadcastable
            if (!a_tensor->shape().broadcastable_with(b_tensor->shape())) {
                stop("Tensor shapes are not broadcastable");
            }
            
            Shape broadcast_shape = a_tensor->shape().broadcast_with(b_tensor->shape());
            DType dtype_a = a_tensor->dtype();
            DType dtype_b = b_tensor->dtype();
            
            if (dtype_a != dtype_b) {
                stop("Cannot compute element-wise min of tensors with different dtypes");
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
            
            std::vector<int> shape_int(broadcast_shape.dims.begin(), broadcast_shape.dims.end());
            
            if (dtype_a == DType::FLOAT32) {
                auto a_wrapper = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrapper = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrapper || !b_wrapper) {
                    throw std::runtime_error("Invalid tensor wrappers for FLOAT32");
                }
                auto result = std::make_shared<gpuTensor<float>>(broadcast_shape);
                tensor_min_broadcast_float32(
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
                tensor_min_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast element-wise min only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype_a);
            
            return ptr;
        }

        // Handle scalar broadcast case
        if (scalar_broadcast) {
            // For now, fall back to shape check - scalar min broadcasting is less common
            stop("Scalar broadcasting not yet implemented for element-wise min");
        }
        
        // Fast path for equal shapes - use existing implementation
        DType dtype = a_tensor->dtype();
        if (dtype != b_tensor->dtype()) {
            stop("Tensors must have the same dtype");
        }

        std::unique_ptr<TensorBase> result_tensor;

        switch (dtype) {
            case DType::FLOAT32: {
                auto a_tw = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_tw = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_tw || !b_tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");

                auto result = std::make_shared<gpuTensor<float>>(a_tw->tensor().shape());

                if (a_tw->tensor().is_contiguous() && b_tw->tensor().is_contiguous()) {
                    tensor_min_elemwise_float32(result->data(), a_tw->tensor().data(), b_tw->tensor().data(), result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto a_desc = a_tw->tensor().descriptor();
                    auto b_desc = b_tw->tensor().descriptor();
                    tensor_min_elemwise_strided_float32(out_desc, a_desc, b_desc);
                }

                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_tw = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_tw = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_tw || !b_tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");

                auto result = std::make_shared<gpuTensor<double>>(a_tw->tensor().shape());

                if (a_tw->tensor().is_contiguous() && b_tw->tensor().is_contiguous()) {
                    tensor_min_elemwise_float64(result->data(), a_tw->tensor().data(), b_tw->tensor().data(), result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto a_desc = a_tw->tensor().descriptor();
                    auto b_desc = b_tw->tensor().descriptor();
                    tensor_min_elemwise_strided_float64(out_desc, a_desc, b_desc);
                }

                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Element-wise min not implemented for dtype: " + dtype_to_string(dtype));
        }

        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor element-wise min: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_pow_elemwise_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);

        if (!a_tensor || !b_tensor) stop("Invalid tensor pointer(s)");

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
                stop("Cannot compute element-wise pow of tensors with different dtypes");
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
                tensor_pow_broadcast_float32(
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
                tensor_pow_broadcast_float64(
                    result->data(),
                    a_wrapper->tensor().data(),
                    b_wrapper->tensor().data(),
                    a_strides.data(), b_strides.data(), result_strides.data(),
                    shape_int.data(), broadcast_shape.ndims(), broadcast_shape.size()
                );
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
            } else {
                stop("Broadcast element-wise pow only supported for float and double dtypes");
            }
            
            auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
            XPtr<TensorBase> ptr(tensor_unique.release(), true);
            
            ptr.attr("class") = "gpuTensor";
            ptr.attr("dtype") = dtype_to_string(dtype_a);
            
            return ptr;
        }

        // Handle scalar broadcast case (fallback to element-wise for now)
        if (scalar_broadcast) {
            // For now, fall back to shape check - scalar pow broadcasting is less common
            stop("Scalar broadcasting not yet implemented for element-wise pow");
        }
        
        // Fast path for equal shapes - use existing implementation
        DType dtype = a_tensor->dtype();
        if (dtype != b_tensor->dtype()) {
            stop("Tensors must have the same dtype");
        }

        std::unique_ptr<TensorBase> result_tensor;

        switch (dtype) {
            case DType::FLOAT32: {
                auto a_tw = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_tw = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_tw || !b_tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");

                auto result = std::make_shared<gpuTensor<float>>(a_tw->tensor().shape());

                if (a_tw->tensor().is_contiguous() && b_tw->tensor().is_contiguous()) {
                    tensor_pow_elemwise_float32(result->data(), a_tw->tensor().data(), b_tw->tensor().data(), result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto a_desc = a_tw->tensor().descriptor();
                    auto b_desc = b_tw->tensor().descriptor();
                    tensor_pow_elemwise_strided_float32(out_desc, a_desc, b_desc);
                }

                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto a_tw = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_tw = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_tw || !b_tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");

                auto result = std::make_shared<gpuTensor<double>>(a_tw->tensor().shape());

                if (a_tw->tensor().is_contiguous() && b_tw->tensor().is_contiguous()) {
                    tensor_pow_elemwise_float64(result->data(), a_tw->tensor().data(), b_tw->tensor().data(), result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto a_desc = a_tw->tensor().descriptor();
                    auto b_desc = b_tw->tensor().descriptor();
                    tensor_pow_elemwise_strided_float64(out_desc, a_desc, b_desc);
                }

                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Element-wise pow not implemented for dtype: " + dtype_to_string(dtype));
        }

        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor element-wise pow: " + std::string(e.what()));
    }
} 

// New scalar operations for Phase 3.2

// [[Rcpp::export]]
SEXP tensor_scalar_sub_unified(SEXP tensor_ptr, double scalar) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        if (!tensor) stop("Invalid tensor pointer");

        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;

        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                
                const gpuTensor<float>& input_tensor = tw->tensor();
                auto result = std::make_shared<gpuTensor<float>>(input_tensor.shape());
                float scalar_f = static_cast<float>(scalar);
                
                if (input_tensor.is_contiguous()) {
                    tensor_scalar_sub_float32(result->data(), input_tensor.data(), scalar_f, result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto in_desc = input_tensor.descriptor();
                    tensor_scalar_sub_strided_float32(out_desc, in_desc, scalar_f);
                }
                
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                const gpuTensor<double>& input_tensor = tw->tensor();
                auto result = std::make_shared<gpuTensor<double>>(input_tensor.shape());
                
                if (input_tensor.is_contiguous()) {
                    tensor_scalar_sub_float64(result->data(), input_tensor.data(), scalar, result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto in_desc = input_tensor.descriptor();
                    tensor_scalar_sub_strided_float64(out_desc, in_desc, scalar);
                }
                
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Scalar subtraction not implemented for dtype: " + dtype_to_string(dtype));
        }

        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor scalar subtraction: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_scalar_div_unified(SEXP tensor_ptr, double scalar) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        if (!tensor) stop("Invalid tensor pointer");

        if (scalar == 0.0) {
            stop("Division by zero");
        }

        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;

        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                
                const gpuTensor<float>& input_tensor = tw->tensor();
                auto result = std::make_shared<gpuTensor<float>>(input_tensor.shape());
                float scalar_f = static_cast<float>(scalar);
                
                if (input_tensor.is_contiguous()) {
                    tensor_scalar_div_float32(result->data(), input_tensor.data(), scalar_f, result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto in_desc = input_tensor.descriptor();
                    tensor_scalar_div_strided_float32(out_desc, in_desc, scalar_f);
                }
                
                result_tensor = std::make_unique<TensorWrapper<float>>(result);
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                const gpuTensor<double>& input_tensor = tw->tensor();
                auto result = std::make_shared<gpuTensor<double>>(input_tensor.shape());
                
                if (input_tensor.is_contiguous()) {
                    tensor_scalar_div_float64(result->data(), input_tensor.data(), scalar, result->size());
                } else {
                    auto out_desc = result->descriptor();
                    auto in_desc = input_tensor.descriptor();
                    tensor_scalar_div_strided_float64(out_desc, in_desc, scalar);
                }
                
                result_tensor = std::make_unique<TensorWrapper<double>>(result);
                break;
            }
            default:
                stop("Scalar division not implemented for dtype: " + dtype_to_string(dtype));
        }

        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor scalar division: " + std::string(e.what()));
    }
} 