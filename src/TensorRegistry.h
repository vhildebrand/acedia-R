#ifndef TENSOR_REGISTRY_H
#define TENSOR_REGISTRY_H

#include "gpuTensor.h"
#include <memory>
#include <typeinfo>
#include <unordered_map>
#include <functional>

// Forward declarations of CUDA kernels (see tensor_ops.cu)
extern "C" {
    // Element-wise add
    void tensor_add_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_add_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_add_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_add_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_add_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    // Element-wise multiply
    void tensor_mul_float16(half* result, const half* a, const half* b, size_t n);
    void tensor_mul_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_mul_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_mul_strided_float32(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    void tensor_mul_strided_float64(const cuda_utils::TensorDescriptor& out_desc,
                                    const cuda_utils::TensorDescriptor& a_desc,
                                    const cuda_utils::TensorDescriptor& b_desc);
    // Scalar multiply
    void tensor_scalar_mul_float16(half* result, const half* input, float scalar, size_t n);
    void tensor_scalar_mul_float32(float* result, const float* input, float scalar, size_t n);
    void tensor_scalar_mul_float64(double* result, const double* input, double scalar, size_t n);
    // Matrix multiply
    void tensor_matmul_float16(half* C, const half* A, const half* B, size_t M, size_t N, size_t K);
    void tensor_matmul_float32(float* C, const float* A, const float* B, size_t M, size_t N, size_t K);
    void tensor_matmul_float64(double* C, const double* A, const double* B, size_t M, size_t N, size_t K);
    // Reductions
    float  tensor_sum_float16(const half* input, size_t n);
    float  tensor_sum_float32(const float* input, size_t n);
    double tensor_sum_float64(const double* input, size_t n);
}

// Helper for static_assert false in templated context
template<typename> struct always_false : std::false_type {};

/**
 * @brief Type-erased base class for tensors
 */
class TensorBase {
public:
    virtual ~TensorBase() = default;
    virtual Shape shape() const = 0;
    virtual DType dtype() const = 0;
    virtual Device device() const = 0;
    virtual size_t size() const = 0;
    virtual size_t ndims() const = 0;
    virtual bool requires_grad() const = 0;
    virtual bool is_contiguous() const = 0;
    virtual std::string info() const = 0;
    virtual void synchronize() = 0;
    
    // Type conversion methods
    virtual std::unique_ptr<TensorBase> to_float() const = 0;
    virtual std::unique_ptr<TensorBase> to_double() const = 0;
    virtual std::unique_ptr<TensorBase> to_half() const = 0;
    virtual std::unique_ptr<TensorBase> to_bfloat16() const = 0;
    virtual std::unique_ptr<TensorBase> to_int8() const = 0;
    virtual std::unique_ptr<TensorBase> to_int32() const = 0;
    virtual std::unique_ptr<TensorBase> to_int64() const = 0;
    
    // View operations
    virtual std::unique_ptr<TensorBase> view(const Shape& new_shape) const = 0;
    virtual std::unique_ptr<TensorBase> reshape(const Shape& new_shape) const = 0;
    
    // Data access
    virtual void copy_to_host_generic(void* host_ptr) const = 0;
    virtual void copy_from_host_generic(const void* host_ptr) = 0;
    
    // Operations
    virtual std::unique_ptr<TensorBase> add(const TensorBase& other) const = 0;
    virtual std::unique_ptr<TensorBase> mul(const TensorBase& other) const = 0;
    virtual std::unique_ptr<TensorBase> scalar_mul(double scalar) const = 0;
    virtual std::unique_ptr<TensorBase> matmul(const TensorBase& other) const = 0;
    virtual double sum() const = 0;
    
    // Clone operation
    virtual std::unique_ptr<TensorBase> clone() const = 0;
    
    // Get raw pointer (for type-specific operations)
    virtual void* get_data_ptr() const = 0;
};

/**
 * @brief Templated wrapper for gpuTensor
 */
template<typename T>
class TensorWrapper : public TensorBase {
private:
    std::shared_ptr<gpuTensor<T>> tensor_;

public:
    explicit TensorWrapper(std::shared_ptr<gpuTensor<T>> tensor) : tensor_(tensor) {}
    explicit TensorWrapper(const gpuTensor<T>& tensor) 
        : tensor_(std::make_shared<gpuTensor<T>>(tensor)) {}
    
    // Access to wrapped tensor
    gpuTensor<T>& tensor() { return *tensor_; }
    const gpuTensor<T>& tensor() const { return *tensor_; }
    std::shared_ptr<gpuTensor<T>> tensor_ptr() { return tensor_; }
    std::shared_ptr<const gpuTensor<T>> tensor_ptr() const { return tensor_; }
    
    // TensorBase interface implementation
    Shape shape() const override { return tensor_->shape(); }
    DType dtype() const override { return tensor_->dtype(); }
    Device device() const override { return tensor_->device(); }
    size_t size() const override { return tensor_->size(); }
    size_t ndims() const override { return tensor_->ndims(); }
    bool requires_grad() const override { return tensor_->requires_grad(); }
    bool is_contiguous() const override { return tensor_->is_contiguous(); }
    std::string info() const override { return tensor_->info(); }
    void synchronize() override { tensor_->synchronize(); }
    
    // Type conversion methods
    std::unique_ptr<TensorBase> to_float() const override {
        return std::make_unique<TensorWrapper<float>>(tensor_->to_float());
    }
    
    std::unique_ptr<TensorBase> to_double() const override {
        return std::make_unique<TensorWrapper<double>>(tensor_->to_double());
    }
    
    std::unique_ptr<TensorBase> to_half() const override {
        return std::make_unique<TensorWrapper<half>>(tensor_->to_half());
    }
    
    std::unique_ptr<TensorBase> to_bfloat16() const override {
        return std::make_unique<TensorWrapper<__nv_bfloat16>>(tensor_->to_bfloat16());
    }
    
    std::unique_ptr<TensorBase> to_int8() const override {
        return std::make_unique<TensorWrapper<int8_t>>(tensor_->template to<int8_t>());
    }
    
    std::unique_ptr<TensorBase> to_int32() const override {
        return std::make_unique<TensorWrapper<int32_t>>(tensor_->template to<int32_t>());
    }
    
    std::unique_ptr<TensorBase> to_int64() const override {
        return std::make_unique<TensorWrapper<int64_t>>(tensor_->template to<int64_t>());
    }
    
    // View operations
    std::unique_ptr<TensorBase> view(const Shape& new_shape) const override {
        // CRITICAL FIX: Create view that shares the same underlying gpuTensor instance
        // instead of creating a copy
        auto view_tensor = tensor_->view(new_shape);
        return std::make_unique<TensorWrapper<T>>(
            std::make_shared<gpuTensor<T>>(std::move(view_tensor))
        );
    }
    
    std::unique_ptr<TensorBase> reshape(const Shape& new_shape) const override {
        // CRITICAL FIX: Create reshape that shares storage when possible
        auto reshaped_tensor = tensor_->reshape(new_shape);
        return std::make_unique<TensorWrapper<T>>(
            std::make_shared<gpuTensor<T>>(std::move(reshaped_tensor))
        );
    }
    
    // Data access
    void copy_to_host_generic(void* host_ptr) const override {
        tensor_->copy_to_host(static_cast<T*>(host_ptr));
    }
    
    void copy_from_host_generic(const void* host_ptr) override {
        tensor_->copy_from_host(static_cast<const T*>(host_ptr));
    }
    
    // Operations (previously TODO) – now implemented with CUDA kernels
    std::unique_ptr<TensorBase> add(const TensorBase& other) const override {
        // Ensure same dtype for now (type-promotion handled elsewhere)
        if (other.dtype() != tensor_->dtype()) {
            throw std::runtime_error("add: dtype mismatch – promotion not yet implemented");
        }
        const auto* other_wrap = dynamic_cast<const TensorWrapper<T>*>(&other);
        if (!other_wrap) {
            throw std::runtime_error("add: dynamic_cast failed");
        }
        if (tensor_->shape() != other_wrap->tensor().shape()) {
            throw std::runtime_error("add: shape mismatch");
        }
        // Ensure contiguous fast path or strided for non-contiguous (float32/64)
        bool a_contig_flag = tensor_->is_contiguous();
        bool b_contig_flag = other_wrap->tensor().is_contiguous();

        auto result = std::make_shared<gpuTensor<T>>(tensor_->shape());

        if constexpr (std::is_same_v<T, float>) {
            if (a_contig_flag && b_contig_flag) {
                tensor_add_float32(result->data(), tensor_->data(), other_wrap->tensor().data(), result->size());
            } else {
                tensor_add_strided_float32(result->descriptor(), tensor_->descriptor(), other_wrap->tensor().descriptor());
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if (a_contig_flag && b_contig_flag) {
                tensor_add_float64(result->data(), tensor_->data(), other_wrap->tensor().data(), result->size());
            } else {
                tensor_add_strided_float64(result->descriptor(), tensor_->descriptor(), other_wrap->tensor().descriptor());
            }
        } else if constexpr (std::is_same_v<T, half>) {
            // No strided kernel yet – fall back to contiguous copy path
            gpuTensor<T> a_contig = a_contig_flag ? *tensor_ : tensor_->contiguous();
            gpuTensor<T> b_contig = b_contig_flag ? other_wrap->tensor() : other_wrap->tensor().contiguous();
            tensor_add_float16(result->data(), a_contig.data(), b_contig.data(), result->size());
        } else {
            throw std::runtime_error("add not implemented for this type");
        }
        return std::make_unique<TensorWrapper<T>>(result);
    }

    std::unique_ptr<TensorBase> mul(const TensorBase& other) const override {
        if (other.dtype() != tensor_->dtype()) {
            throw std::runtime_error("mul: dtype mismatch – promotion not yet implemented");
        }
        const auto* other_wrap = dynamic_cast<const TensorWrapper<T>*>(&other);
        if (!other_wrap) {
            throw std::runtime_error("mul: dynamic_cast failed");
        }
        if (tensor_->shape() != other_wrap->tensor().shape()) {
            throw std::runtime_error("mul: shape mismatch");
        }
        bool a_contig_flag = tensor_->is_contiguous();
        bool b_contig_flag = other_wrap->tensor().is_contiguous();

        auto result = std::make_shared<gpuTensor<T>>(tensor_->shape());

        if constexpr (std::is_same_v<T, float>) {
            if (a_contig_flag && b_contig_flag) {
                tensor_mul_float32(result->data(), tensor_->data(), other_wrap->tensor().data(), result->size());
            } else {
                tensor_mul_strided_float32(result->descriptor(), tensor_->descriptor(), other_wrap->tensor().descriptor());
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if (a_contig_flag && b_contig_flag) {
                tensor_mul_float64(result->data(), tensor_->data(), other_wrap->tensor().data(), result->size());
            } else {
                tensor_mul_strided_float64(result->descriptor(), tensor_->descriptor(), other_wrap->tensor().descriptor());
            }
        } else if constexpr (std::is_same_v<T, half>) {
            gpuTensor<T> a_contig = a_contig_flag ? *tensor_ : tensor_->contiguous();
            gpuTensor<T> b_contig = b_contig_flag ? other_wrap->tensor() : other_wrap->tensor().contiguous();
            tensor_mul_float16(result->data(), a_contig.data(), b_contig.data(), result->size());
        } else {
            throw std::runtime_error("mul not implemented for this type");
        }
        return std::make_unique<TensorWrapper<T>>(result);
    }

    std::unique_ptr<TensorBase> scalar_mul(double scalar) const override {
        gpuTensor<T> a_contig = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        auto result = std::make_shared<gpuTensor<T>>(tensor_->shape());
        if constexpr (std::is_same_v<T, float>) {
            tensor_scalar_mul_float32(result->data(), a_contig.data(), static_cast<float>(scalar), result->size());
        } else if constexpr (std::is_same_v<T, double>) {
            tensor_scalar_mul_float64(result->data(), a_contig.data(), scalar, result->size());
        } else if constexpr (std::is_same_v<T, half>) {
            tensor_scalar_mul_float16(result->data(), a_contig.data(), static_cast<float>(scalar), result->size());
        } else {
            throw std::runtime_error("scalar_mul not implemented for this type");
        }
        return std::make_unique<TensorWrapper<T>>(result);
    }

    std::unique_ptr<TensorBase> matmul(const TensorBase& other) const override {
        if (other.dtype() != tensor_->dtype()) {
            throw std::runtime_error("matmul: dtype mismatch – promotion not yet implemented");
        }
        const auto* other_wrap = dynamic_cast<const TensorWrapper<T>*>(&other);
        if (!other_wrap) {
            throw std::runtime_error("matmul: dynamic_cast failed");
        }
        // Expect 2-D matrices with matching inner dim
        if (tensor_->ndims() != 2 || other_wrap->tensor().ndims() != 2) {
            throw std::runtime_error("matmul: requires 2-D tensors");
        }
        size_t M = tensor_->shape()[0];
        size_t K = tensor_->shape()[1];
        size_t K2 = other_wrap->tensor().shape()[0];
        size_t N = other_wrap->tensor().shape()[1];
        if (K != K2) {
            throw std::runtime_error("matmul: inner dimensions do not match");
        }
        gpuTensor<T> A = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        gpuTensor<T> B = other_wrap->tensor().is_contiguous() ? other_wrap->tensor() : other_wrap->tensor().contiguous();
        Shape result_shape({M, N});
        auto result = std::make_shared<gpuTensor<T>>(result_shape);
        if constexpr (std::is_same_v<T, float>) {
            tensor_matmul_float32(result->data(), A.data(), B.data(), M, N, K);
        } else if constexpr (std::is_same_v<T, double>) {
            tensor_matmul_float64(result->data(), A.data(), B.data(), M, N, K);
        } else if constexpr (std::is_same_v<T, half>) {
            tensor_matmul_float16(result->data(), A.data(), B.data(), M, N, K);
        } else {
            throw std::runtime_error("matmul not implemented for this type");
        }
        return std::make_unique<TensorWrapper<T>>(result);
    }

    double sum() const override {
        gpuTensor<T> a_contig = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        if constexpr (std::is_same_v<T, float>) {
            return static_cast<double>(tensor_sum_float32(a_contig.data(), a_contig.size()));
        } else if constexpr (std::is_same_v<T, double>) {
            return tensor_sum_float64(a_contig.data(), a_contig.size());
        } else if constexpr (std::is_same_v<T, half>) {
            return static_cast<double>(tensor_sum_float16(a_contig.data(), a_contig.size()));
        } else {
            throw std::runtime_error("sum not implemented for this type");
        }
    }
    
    std::unique_ptr<TensorBase> clone() const override {
        return std::make_unique<TensorWrapper<T>>(*tensor_);
    }
    
    void* get_data_ptr() const override {
        return tensor_->data();
    }
};

/**
 * @brief Factory functions for creating tensors of different types
 */
class TensorFactory {
public:
    // Create tensor from R data
    template<typename T>
    static std::unique_ptr<TensorBase> create_tensor(const std::vector<double>& data, const Shape& shape) {
        std::vector<T> typed_data;
        typed_data.reserve(data.size());
        
        for (double val : data) {
            typed_data.push_back(static_cast<T>(val));
        }
        
        auto tensor = std::make_shared<gpuTensor<T>>(typed_data, shape);
        return std::make_unique<TensorWrapper<T>>(tensor);
    }
    
    // Create empty tensor
    template<typename T>
    static std::unique_ptr<TensorBase> create_empty_tensor(const Shape& shape) {
        auto tensor = std::make_shared<gpuTensor<T>>(shape);
        return std::make_unique<TensorWrapper<T>>(tensor);
    }
    
    // Create tensor by dtype string
    static std::unique_ptr<TensorBase> create_tensor_by_dtype(
        const std::vector<double>& data, const Shape& shape, const std::string& dtype_str) {
        
        DType dtype = string_to_dtype(dtype_str);
        return create_tensor_by_dtype_enum(data, shape, dtype);
    }
    
    // Create tensor by DType enum
    static std::unique_ptr<TensorBase> create_tensor_by_dtype_enum(
        const std::vector<double>& data, const Shape& shape, DType dtype) {
        
        switch (dtype) {
            case DType::FLOAT16:
                return create_tensor<half>(data, shape);
            case DType::FLOAT32:
                return create_tensor<float>(data, shape);
            case DType::FLOAT64:
                return create_tensor<double>(data, shape);
            case DType::INT8:
                return create_tensor<int8_t>(data, shape);
            case DType::INT32:
                return create_tensor<int32_t>(data, shape);
            case DType::INT64:
                return create_tensor<int64_t>(data, shape);
            default:
                throw std::runtime_error("Unsupported dtype: " + dtype_to_string(dtype));
        }
    }
    
    // Create empty tensor by dtype
    static std::unique_ptr<TensorBase> create_empty_tensor_by_dtype(
        const Shape& shape, const std::string& dtype_str) {
        
        DType dtype = string_to_dtype(dtype_str);
        return create_empty_tensor_by_dtype_enum(shape, dtype);
    }
    
    // Create empty tensor by DType enum
    static std::unique_ptr<TensorBase> create_empty_tensor_by_dtype_enum(
        const Shape& shape, DType dtype) {
        
        switch (dtype) {
            case DType::FLOAT16:
                return create_empty_tensor<half>(shape);
            case DType::FLOAT32:
                return create_empty_tensor<float>(shape);
            case DType::FLOAT64:
                return create_empty_tensor<double>(shape);
            case DType::INT8:
                return create_empty_tensor<int8_t>(shape);
            case DType::INT32:
                return create_empty_tensor<int32_t>(shape);
            case DType::INT64:
                return create_empty_tensor<int64_t>(shape);
            default:
                throw std::runtime_error("Unsupported dtype: " + dtype_to_string(dtype));
        }
    }
};

/**
 * @brief Type promotion utilities
 */
class TypePromotion {
public:
    // Determine result type for binary operations
    static DType get_result_type(const TensorBase& a, const TensorBase& b) {
        return promote_types(a.dtype(), b.dtype());
    }
    
    // Promote tensor to target type
    static std::unique_ptr<TensorBase> promote_tensor(const TensorBase& tensor, DType target_dtype) {
        if (tensor.dtype() == target_dtype) {
            return tensor.clone();
        }
        
        switch (target_dtype) {
            case DType::FLOAT64:
                return tensor.to_double();
            case DType::FLOAT32:
                return tensor.to_float();
            case DType::FLOAT16:
                return tensor.to_half();
            case DType::BFLOAT16:
                return tensor.to_bfloat16();
            case DType::INT64:
                return tensor.to_int64();
            case DType::INT32:
                return tensor.to_int32();
            case DType::INT8:
                return tensor.to_int8();
            default:
                throw std::runtime_error("Unsupported target dtype for promotion");
        }
    }
    
    // Perform binary operation with automatic type promotion
    static std::unique_ptr<TensorBase> binary_op_with_promotion(
        const TensorBase& a, const TensorBase& b, 
        std::function<std::unique_ptr<TensorBase>(const TensorBase&, const TensorBase&)> op) {
        
        DType result_type = get_result_type(a, b);
        
        auto a_promoted = promote_tensor(a, result_type);
        auto b_promoted = promote_tensor(b, result_type);
        
        return op(*a_promoted, *b_promoted);
    }
};

/**
 * @brief Helper macros for type dispatching
 */
#define DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
    switch (TYPE) { \
        case DType::FLOAT16: { \
            using scalar_t = half; \
            __VA_ARGS__(); \
            break; \
        } \
        case DType::FLOAT32: { \
            using scalar_t = float; \
            __VA_ARGS__(); \
            break; \
        } \
        case DType::FLOAT64: { \
            using scalar_t = double; \
            __VA_ARGS__(); \
            break; \
        } \
        case DType::INT8: { \
            using scalar_t = int8_t; \
            __VA_ARGS__(); \
            break; \
        } \
        case DType::INT32: { \
            using scalar_t = int32_t; \
            __VA_ARGS__(); \
            break; \
        } \
        case DType::INT64: { \
            using scalar_t = int64_t; \
            __VA_ARGS__(); \
            break; \
        } \
        default: \
            throw std::runtime_error("Unsupported dtype in " NAME ": " + dtype_to_string(TYPE)); \
    }

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    switch (TYPE) { \
        case DType::FLOAT16: { \
            using scalar_t = half; \
            __VA_ARGS__(); \
            break; \
        } \
        case DType::BFLOAT16: { \
            using scalar_t = __nv_bfloat16; \
            __VA_ARGS__(); \
            break; \
        } \
        case DType::FLOAT32: { \
            using scalar_t = float; \
            __VA_ARGS__(); \
            break; \
        } \
        case DType::FLOAT64: { \
            using scalar_t = double; \
            __VA_ARGS__(); \
            break; \
        } \
        default: \
            throw std::runtime_error("Operation " NAME " only supports floating point types"); \
    }

#endif // TENSOR_REGISTRY_H 