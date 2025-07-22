#ifndef TENSOR_REGISTRY_H
#define TENSOR_REGISTRY_H

#include "gpuTensor.h"
#include <memory>
#include <typeinfo>
#include <unordered_map>
#include <functional>

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
        return std::make_unique<TensorWrapper<T>>(
            std::make_shared<gpuTensor<T>>(tensor_->view(new_shape))
        );
    }
    
    std::unique_ptr<TensorBase> reshape(const Shape& new_shape) const override {
        return std::make_unique<TensorWrapper<T>>(
            std::make_shared<gpuTensor<T>>(tensor_->reshape(new_shape))
        );
    }
    
    // Data access
    void copy_to_host_generic(void* host_ptr) const override {
        tensor_->copy_to_host(static_cast<T*>(host_ptr));
    }
    
    void copy_from_host_generic(const void* host_ptr) override {
        tensor_->copy_from_host(static_cast<const T*>(host_ptr));
    }
    
    // Operations (to be implemented with templated kernels)
    std::unique_ptr<TensorBase> add(const TensorBase& other) const override {
        // Check if other is the same type
        if (other.dtype() == tensor_->dtype()) {
            const TensorWrapper<T>* other_wrapper = dynamic_cast<const TensorWrapper<T>*>(&other);
            if (other_wrapper) {
                // TODO: Implement templated addition
                throw std::runtime_error("Templated tensor addition not yet implemented");
            }
        }
        
        // Type promotion case
        throw std::runtime_error("Mixed-type tensor operations not yet implemented");
    }
    
    std::unique_ptr<TensorBase> mul(const TensorBase& other) const override {
        // Similar to add, but for multiplication
        throw std::runtime_error("Templated tensor multiplication not yet implemented");
    }
    
    std::unique_ptr<TensorBase> scalar_mul(double scalar) const override {
        // TODO: Implement templated scalar multiplication
        throw std::runtime_error("Templated scalar multiplication not yet implemented");
    }
    
    std::unique_ptr<TensorBase> matmul(const TensorBase& other) const override {
        // TODO: Implement templated matrix multiplication
        throw std::runtime_error("Templated matrix multiplication not yet implemented");
    }
    
    double sum() const override {
        // TODO: Implement templated sum reduction
        throw std::runtime_error("Templated sum reduction not yet implemented");
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