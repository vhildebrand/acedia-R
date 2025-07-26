#ifndef TENSOR_REGISTRY_H
#define TENSOR_REGISTRY_H

#include "gpuTensor.h"
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <typeinfo>
#include <unordered_map>
#include <functional>
#include <set>  // For axis validation helper methods

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
    
    // Global argmax/argmin
    int64_t tensor_argmax_float32(const float* input, size_t n);
    int64_t tensor_argmax_float64(const double* input, size_t n);
    int64_t tensor_argmin_float32(const float* input, size_t n);
    int64_t tensor_argmin_float64(const double* input, size_t n);

    // Axis-aware reductions
    void tensor_axis_sum_float32(float* result, const float* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_sum_float64(double* result, const double* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware mean
    void tensor_axis_mean_float32(float* result, const float* input,
                                 const int* input_strides, const int* input_shape,
                                 const int* result_strides, const int* reduction_axes,
                                 int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_mean_float64(double* result, const double* input,
                                 const int* input_strides, const int* input_shape,
                                 const int* result_strides, const int* reduction_axes,
                                 int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware max
    void tensor_axis_max_float32(float* result, const float* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_max_float64(double* result, const double* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware min
    void tensor_axis_min_float32(float* result, const float* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_min_float64(double* result, const double* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware prod
    void tensor_axis_prod_float32(float* result, const float* input,
                                 const int* input_strides, const int* input_shape,
                                 const int* result_strides, const int* reduction_axes,
                                 int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_prod_float64(double* result, const double* input,
                                 const int* input_strides, const int* input_shape,
                                 const int* result_strides, const int* reduction_axes,
                                 int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware var
    void tensor_axis_var_float32(float* result, const float* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_var_float64(double* result, const double* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    
    // NEW: Axis-aware argmax/argmin
    void tensor_axis_argmax_float32(int64_t* idx_out, const float* input,
                                   const int* in_strides, const int* in_shape, 
                                   const int* out_strides, const int* axes, 
                                   int n_axes, int ndims, size_t out_elems);
    void tensor_axis_argmax_float64(int64_t* idx_out, const double* input,
                                   const int* in_strides, const int* in_shape, 
                                   const int* out_strides, const int* axes, 
                                   int n_axes, int ndims, size_t out_elems);
    void tensor_axis_argmin_float32(int64_t* idx_out, const float* input,
                                   const int* in_strides, const int* in_shape, 
                                   const int* out_strides, const int* axes, 
                                   int n_axes, int ndims, size_t out_elems);
    void tensor_axis_argmin_float64(int64_t* idx_out, const double* input,
                                   const int* in_strides, const int* in_shape, 
                                   const int* out_strides, const int* axes, 
                                   int n_axes, int ndims, size_t out_elems);
}

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
    
    // NEW: Axis-aware reduction kernels
    // Axis-aware sum
    void tensor_axis_sum_float32(float* result, const float* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_sum_float64(double* result, const double* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware mean
    void tensor_axis_mean_float32(float* result, const float* input,
                                 const int* input_strides, const int* input_shape,
                                 const int* result_strides, const int* reduction_axes,
                                 int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_mean_float64(double* result, const double* input,
                                 const int* input_strides, const int* input_shape,
                                 const int* result_strides, const int* reduction_axes,
                                 int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware max
    void tensor_axis_max_float32(float* result, const float* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_max_float64(double* result, const double* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware min
    void tensor_axis_min_float32(float* result, const float* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_min_float64(double* result, const double* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware prod
    void tensor_axis_prod_float32(float* result, const float* input,
                                 const int* input_strides, const int* input_shape,
                                 const int* result_strides, const int* reduction_axes,
                                 int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_prod_float64(double* result, const double* input,
                                 const int* input_strides, const int* input_shape,
                                 const int* result_strides, const int* reduction_axes,
                                 int num_reduction_axes, int ndims, size_t output_size);
    // Axis-aware var
    void tensor_axis_var_float32(float* result, const float* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
    void tensor_axis_var_float64(double* result, const double* input,
                                const int* input_strides, const int* input_shape,
                                const int* result_strides, const int* reduction_axes,
                                int num_reduction_axes, int ndims, size_t output_size);
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
    virtual std::unique_ptr<TensorBase> sum() const = 0;  // Return tensor instead of scalar
    virtual std::unique_ptr<TensorBase> mean() const = 0;  // Add mean
    virtual std::unique_ptr<TensorBase> max() const = 0;   // Add max
    virtual std::unique_ptr<TensorBase> min() const = 0;   // Add min
    virtual std::unique_ptr<TensorBase> prod() const = 0;  // Add prod
    virtual std::unique_ptr<TensorBase> var() const = 0;   // Add var
    
    // NEW: Axis-aware reduction methods
    virtual std::unique_ptr<TensorBase> sum(const std::vector<int>& axis, bool keep_dims = false) const = 0;
    virtual std::unique_ptr<TensorBase> mean(const std::vector<int>& axis, bool keep_dims = false) const = 0;
    virtual std::unique_ptr<TensorBase> max(const std::vector<int>& axis, bool keep_dims = false) const = 0;
    virtual std::unique_ptr<TensorBase> min(const std::vector<int>& axis, bool keep_dims = false) const = 0;
    virtual std::unique_ptr<TensorBase> prod(const std::vector<int>& axis, bool keep_dims = false) const = 0;
    virtual std::unique_ptr<TensorBase> var(const std::vector<int>& axis, bool keep_dims = false) const = 0;
    
    // Argmax/argmin operations
    virtual std::unique_ptr<TensorBase> argmax() const = 0;  // Global argmax
    virtual std::unique_ptr<TensorBase> argmin() const = 0;  // Global argmin
    virtual std::unique_ptr<TensorBase> argmax(int axis, bool keep_dims = false) const = 0;  // Axis-aware argmax
    virtual std::unique_ptr<TensorBase> argmin(int axis, bool keep_dims = false) const = 0;  // Axis-aware argmin
    
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

    std::unique_ptr<TensorBase> sum() const override {
        gpuTensor<T> a_contig = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        if constexpr (std::is_same_v<T, float>) {
            float result_val = tensor_sum_float32(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<float>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(float), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<float>>(result);
        } else if constexpr (std::is_same_v<T, double>) {
            double result_val = tensor_sum_float64(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<double>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(double), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<double>>(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float result_val = tensor_sum_float16(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<half>>(Shape{1});
            half half_val = __float2half(result_val);
            cudaMemcpy(result->data(), &half_val, sizeof(half), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<half>>(result);
        } else {
            throw std::runtime_error("sum not implemented for this type");
        }
    }

    std::unique_ptr<TensorBase> mean() const override {
        gpuTensor<T> a_contig = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        if constexpr (std::is_same_v<T, float>) {
            float sum_val = tensor_sum_float32(a_contig.data(), a_contig.size());
            float result_val = sum_val / static_cast<float>(a_contig.size());
            auto result = std::make_shared<gpuTensor<float>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(float), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<float>>(result);
        } else if constexpr (std::is_same_v<T, double>) {
            double sum_val = tensor_sum_float64(a_contig.data(), a_contig.size());
            double result_val = sum_val / static_cast<double>(a_contig.size());
            auto result = std::make_shared<gpuTensor<double>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(double), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<double>>(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float sum_val = tensor_sum_float16(a_contig.data(), a_contig.size());
            float result_val = sum_val / static_cast<float>(a_contig.size());
            auto result = std::make_shared<gpuTensor<half>>(Shape{1});
            half half_val = __float2half(result_val);
            cudaMemcpy(result->data(), &half_val, sizeof(half), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<half>>(result);
        } else {
            throw std::runtime_error("mean not implemented for this type");
        }
    }

    std::unique_ptr<TensorBase> max() const override {
        gpuTensor<T> a_contig = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        if constexpr (std::is_same_v<T, float>) {
            float result_val = tensor_max_float32(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<float>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(float), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<float>>(result);
        } else if constexpr (std::is_same_v<T, double>) {
            double result_val = tensor_max_float64(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<double>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(double), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<double>>(result);
        } else {
            throw std::runtime_error("max not implemented for this type");
        }
    }

    std::unique_ptr<TensorBase> min() const override {
        gpuTensor<T> a_contig = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        if constexpr (std::is_same_v<T, float>) {
            float result_val = tensor_min_float32(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<float>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(float), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<float>>(result);
        } else if constexpr (std::is_same_v<T, double>) {
            double result_val = tensor_min_float64(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<double>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(double), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<double>>(result);
        } else {
            throw std::runtime_error("min not implemented for this type");
        }
    }

    std::unique_ptr<TensorBase> prod() const override {
        gpuTensor<T> a_contig = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        if constexpr (std::is_same_v<T, float>) {
            float result_val = tensor_prod_float32(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<float>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(float), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<float>>(result);
        } else if constexpr (std::is_same_v<T, double>) {
            double result_val = tensor_prod_float64(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<double>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(double), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<double>>(result);
        } else {
            throw std::runtime_error("prod not implemented for this type");
        }
    }

    std::unique_ptr<TensorBase> var() const override {
        gpuTensor<T> a_contig = tensor_->is_contiguous() ? *tensor_ : tensor_->contiguous();
        if constexpr (std::is_same_v<T, float>) {
            double result_val = tensor_var_float32(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<float>>(Shape{1});
            float float_val = static_cast<float>(result_val);
            cudaMemcpy(result->data(), &float_val, sizeof(float), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<float>>(result);
        } else if constexpr (std::is_same_v<T, double>) {
            double result_val = tensor_var_float64(a_contig.data(), a_contig.size());
            auto result = std::make_shared<gpuTensor<double>>(Shape{1});
            cudaMemcpy(result->data(), &result_val, sizeof(double), cudaMemcpyHostToDevice);
            return std::make_unique<TensorWrapper<double>>(result);
        } else {
            throw std::runtime_error("var not implemented for this type");
        }
    }
    
    // NEW: Axis-aware reduction method implementations
    std::unique_ptr<TensorBase> sum(const std::vector<int>& axis, bool keep_dims = false) const override {
        // Validate and normalize axis values
        std::vector<int> normalized_axis = validate_and_normalize_axis(axis);
        
        // For now, fall back to global reduction if all axes are being reduced
        if (normalized_axis.size() == tensor_->ndims()) {
            return sum();  // Global reduction
        }
        
        // Call axis-aware reduction kernel
        return perform_axis_reduction("sum", normalized_axis, keep_dims);
    }
    
    std::unique_ptr<TensorBase> mean(const std::vector<int>& axis, bool keep_dims = false) const override {
        std::vector<int> normalized_axis = validate_and_normalize_axis(axis);
        if (normalized_axis.size() == tensor_->ndims()) {
            return mean();  // Global reduction
        }
        return perform_axis_reduction("mean", normalized_axis, keep_dims);
    }
    
    std::unique_ptr<TensorBase> max(const std::vector<int>& axis, bool keep_dims = false) const override {
        std::vector<int> normalized_axis = validate_and_normalize_axis(axis);
        if (normalized_axis.size() == tensor_->ndims()) {
            return max();  // Global reduction
        }
        return perform_axis_reduction("max", normalized_axis, keep_dims);
    }
    
    std::unique_ptr<TensorBase> min(const std::vector<int>& axis, bool keep_dims = false) const override {
        std::vector<int> normalized_axis = validate_and_normalize_axis(axis);
        if (normalized_axis.size() == tensor_->ndims()) {
            return min();  // Global reduction
        }
        return perform_axis_reduction("min", normalized_axis, keep_dims);
    }
    
    std::unique_ptr<TensorBase> prod(const std::vector<int>& axis, bool keep_dims = false) const override {
        std::vector<int> normalized_axis = validate_and_normalize_axis(axis);
        if (normalized_axis.size() == tensor_->ndims()) {
            return prod();  // Global reduction
        }
        return perform_axis_reduction("prod", normalized_axis, keep_dims);
    }
    
    std::unique_ptr<TensorBase> var(const std::vector<int>& axis, bool keep_dims = false) const override {
        std::vector<int> normalized_axis = validate_and_normalize_axis(axis);
        if (normalized_axis.size() == tensor_->ndims()) {
            return var();  // Global reduction
        }
        return perform_axis_reduction("var", normalized_axis, keep_dims);
    }
    
    // Argmax/argmin implementations
    std::unique_ptr<TensorBase> argmax() const override {
        // Global argmax - return single index as int64 tensor
        auto result = std::make_shared<gpuTensor<int64_t>>(Shape{1});
        
        if constexpr (std::is_same_v<T, float>) {
            int64_t index;
            if (tensor_->is_contiguous()) {
                index = tensor_argmax_float32(tensor_->data(), tensor_->size());
            } else {
                // For non-contiguous tensors, make contiguous first
                auto contiguous_tensor = tensor_->contiguous();
                index = tensor_argmax_float32(contiguous_tensor.data(), contiguous_tensor.size());
            }
            cudaError_t err = cudaMemcpy(result->data(), &index, sizeof(int64_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy argmax result to device: " + std::string(cudaGetErrorString(err)));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            int64_t index;
            if (tensor_->is_contiguous()) {
                index = tensor_argmax_float64(tensor_->data(), tensor_->size());
            } else {
                auto contiguous_tensor = tensor_->contiguous();
                index = tensor_argmax_float64(contiguous_tensor.data(), contiguous_tensor.size());
            }
            cudaError_t err = cudaMemcpy(result->data(), &index, sizeof(int64_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy argmax result to device: " + std::string(cudaGetErrorString(err)));
            }
        } else {
            throw std::runtime_error("argmax not implemented for this dtype: " + dtype_to_string(tensor_->dtype()));
        }
        
        return std::make_unique<TensorWrapper<int64_t>>(result);
    }
    
    std::unique_ptr<TensorBase> argmin() const override {
        // Global argmin - return single index as int64 tensor
        auto result = std::make_shared<gpuTensor<int64_t>>(Shape{1});
        
        if constexpr (std::is_same_v<T, float>) {
            int64_t index;
            if (tensor_->is_contiguous()) {
                index = tensor_argmin_float32(tensor_->data(), tensor_->size());
            } else {
                auto contiguous_tensor = tensor_->contiguous();
                index = tensor_argmin_float32(contiguous_tensor.data(), contiguous_tensor.size());
            }
            cudaError_t err = cudaMemcpy(result->data(), &index, sizeof(int64_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy argmin result to device: " + std::string(cudaGetErrorString(err)));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            int64_t index;
            if (tensor_->is_contiguous()) {
                index = tensor_argmin_float64(tensor_->data(), tensor_->size());
            } else {
                auto contiguous_tensor = tensor_->contiguous();
                index = tensor_argmin_float64(contiguous_tensor.data(), contiguous_tensor.size());
            }
            cudaError_t err = cudaMemcpy(result->data(), &index, sizeof(int64_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy argmin result to device: " + std::string(cudaGetErrorString(err)));
            }
        } else {
            throw std::runtime_error("argmin not implemented for this dtype: " + dtype_to_string(tensor_->dtype()));
        }
        
        return std::make_unique<TensorWrapper<int64_t>>(result);
    }
    
    std::unique_ptr<TensorBase> argmax(int axis, bool keep_dims = false) const override {
        int normalized_axis = normalize_single_axis(axis);
        
        // Use perform_axis_reduction like other reductions, but handle int64_t output
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            // Calculate output shape
            Shape result_shape = calculate_reduction_shape({normalized_axis}, keep_dims);
            auto result_tensor = std::make_shared<gpuTensor<int64_t>>(result_shape);
            
            // Handle keep_dims workaround similar to perform_axis_reduction
            if (keep_dims) {
                auto tmp_result = argmax(axis, false);
                Shape final_shape = calculate_reduction_shape({normalized_axis}, true);
                return tmp_result->reshape(final_shape);
            }
            
            // Get input descriptor
            cuda_utils::TensorDescriptor input_desc = tensor_->descriptor();
            cuda_utils::TensorDescriptor output_desc = result_tensor->descriptor();
            
            // Allocate device memory arrays
            int* d_input_shape;
            int* d_input_strides;
            int* d_output_strides;
            int* d_reduction_axes;
            
            cudaMalloc(&d_input_shape, input_desc.ndims * sizeof(int));
            cudaMalloc(&d_input_strides, input_desc.ndims * sizeof(int));
            cudaMalloc(&d_output_strides, std::max(1, output_desc.ndims) * sizeof(int));
            cudaMalloc(&d_reduction_axes, sizeof(int));
            
            // Copy data to device
            cudaMemcpy(d_input_shape, input_desc.shape, input_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_strides, input_desc.strides, input_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_reduction_axes, &normalized_axis, sizeof(int), cudaMemcpyHostToDevice);
            
            if (output_desc.ndims > 0) {
                cudaMemcpy(d_output_strides, output_desc.strides, output_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
            } else {
                int dummy_stride = 1;
                cudaMemcpy(d_output_strides, &dummy_stride, sizeof(int), cudaMemcpyHostToDevice);
            }
            
            // Call kernel with proper error handling
            try {
                if constexpr (std::is_same_v<T, float>) {
                    tensor_axis_argmax_float32(
                        result_tensor->data(), tensor_->data(),
                        d_input_strides, d_input_shape, d_output_strides, d_reduction_axes,
                        1, input_desc.ndims, result_tensor->size()
                    );
                } else {
                    tensor_axis_argmax_float64(
                        result_tensor->data(), tensor_->data(),
                        d_input_strides, d_input_shape, d_output_strides, d_reduction_axes,
                        1, input_desc.ndims, result_tensor->size()
                    );
                }
            } catch (...) {
                // Clean up on error
                cudaFree(d_reduction_axes);
                cudaFree(d_input_shape);
                cudaFree(d_input_strides);
                cudaFree(d_output_strides);
                throw;
            }
            
            // Clean up device memory
            cudaFree(d_reduction_axes);
            cudaFree(d_input_shape);
            cudaFree(d_input_strides);
            cudaFree(d_output_strides);
            
            return std::make_unique<TensorWrapper<int64_t>>(result_tensor);
        } else {
            // Fallback: compile for any T but raise at runtime if invoked.
            throw std::runtime_error("Axis-aware argmax not implemented for this dtype: " + dtype_to_string(tensor_->dtype()));
        }
    }
    
    std::unique_ptr<TensorBase> argmin(int axis, bool keep_dims = false) const override {
        int normalized_axis = normalize_single_axis(axis);
        
        // Use perform_axis_reduction like other reductions, but handle int64_t output
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            // Calculate output shape
            Shape result_shape = calculate_reduction_shape({normalized_axis}, keep_dims);
            auto result_tensor = std::make_shared<gpuTensor<int64_t>>(result_shape);
            
            // Handle keep_dims workaround similar to perform_axis_reduction
            if (keep_dims) {
                auto tmp_result = argmin(axis, false);
                Shape final_shape = calculate_reduction_shape({normalized_axis}, true);
                return tmp_result->reshape(final_shape);
            }
            
            // Get input descriptor
            cuda_utils::TensorDescriptor input_desc = tensor_->descriptor();
            cuda_utils::TensorDescriptor output_desc = result_tensor->descriptor();
            
            // Allocate device memory arrays
            int* d_input_shape;
            int* d_input_strides;
            int* d_output_strides;
            int* d_reduction_axes;
            
            cudaMalloc(&d_input_shape, input_desc.ndims * sizeof(int));
            cudaMalloc(&d_input_strides, input_desc.ndims * sizeof(int));
            cudaMalloc(&d_output_strides, std::max(1, output_desc.ndims) * sizeof(int));
            cudaMalloc(&d_reduction_axes, sizeof(int));
            
            // Copy data to device
            cudaMemcpy(d_input_shape, input_desc.shape, input_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_strides, input_desc.strides, input_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_reduction_axes, &normalized_axis, sizeof(int), cudaMemcpyHostToDevice);
            
            if (output_desc.ndims > 0) {
                cudaMemcpy(d_output_strides, output_desc.strides, output_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
            } else {
                int dummy_stride = 1;
                cudaMemcpy(d_output_strides, &dummy_stride, sizeof(int), cudaMemcpyHostToDevice);
            }
            
            // Call kernel with proper error handling
            try {
                if constexpr (std::is_same_v<T, float>) {
                    tensor_axis_argmin_float32(
                        result_tensor->data(), tensor_->data(),
                        d_input_strides, d_input_shape, d_output_strides, d_reduction_axes,
                        1, input_desc.ndims, result_tensor->size()
                    );
                } else {
                    tensor_axis_argmin_float64(
                        result_tensor->data(), tensor_->data(),
                        d_input_strides, d_input_shape, d_output_strides, d_reduction_axes,
                        1, input_desc.ndims, result_tensor->size()
                    );
                }
            } catch (...) {
                // Clean up on error
                cudaFree(d_reduction_axes);
                cudaFree(d_input_shape);
                cudaFree(d_input_strides);
                cudaFree(d_output_strides);
                throw;
            }
            
            // Clean up device memory
            cudaFree(d_reduction_axes);
            cudaFree(d_input_shape);
            cudaFree(d_input_strides);
            cudaFree(d_output_strides);
            
            return std::make_unique<TensorWrapper<int64_t>>(result_tensor);
        } else {
            // Fallback: compile for any T but raise at runtime if invoked.
            throw std::runtime_error("Axis-aware argmin not implemented for this dtype: " + dtype_to_string(tensor_->dtype()));
        }
    }
    
    // Clone operation
    std::unique_ptr<TensorBase> clone() const override {
        return std::make_unique<TensorWrapper<T>>(*tensor_);
    }
    
    // Get raw pointer (for type-specific operations)
    void* get_data_ptr() const override {
        return tensor_->data();
    }

private:
    // Helper method to validate and normalize axis values
    std::vector<int> validate_and_normalize_axis(const std::vector<int>& axis) const {
        std::vector<int> normalized;
        int ndims = static_cast<int>(tensor_->ndims());
        
        for (int ax : axis) {
            // Handle negative indexing (Python-style)
            int norm_ax = ax < 0 ? ndims + ax : ax;
            
            // Validate range
            if (norm_ax < 0 || norm_ax >= ndims) {
                throw std::runtime_error("axis " + std::to_string(ax) + " is out of bounds for tensor with " + 
                                       std::to_string(ndims) + " dimensions");
            }
            
            normalized.push_back(norm_ax);
        }
        
        // Check for duplicates
        std::sort(normalized.begin(), normalized.end());
        auto last = std::unique(normalized.begin(), normalized.end());
        if (last != normalized.end()) {
            throw std::runtime_error("repeated axis in reduction");
        }
        
        return normalized;
    }
    
    // Helper method to normalize single axis  
    int normalize_single_axis(int axis) const {
        int ndims = static_cast<int>(tensor_->ndims());
        int norm_ax = axis < 0 ? ndims + axis : axis;
        
        if (norm_ax < 0 || norm_ax >= ndims) {
            throw std::runtime_error("axis " + std::to_string(axis) + " is out of bounds for tensor with " + 
                                   std::to_string(ndims) + " dimensions");
        }
        
        return norm_ax;
    }
    
    // Core method that performs axis-aware reduction using CUDA kernels
    std::unique_ptr<TensorBase> perform_axis_reduction(const std::string& op_name, 
                                                      const std::vector<int>& axis, 
                                                      bool keep_dims) const {
        // ------------------------------------------------------------------
        // WORK-AROUND: kernel expects squeezed output (no keep-dims).  If the
        // user requests keep_dims=TRUE we first execute the reduction with
        // keep_dims == FALSE, then reshape the returned tensor to reinstate
        // the singleton dimensions on the host.  This avoids the illegal
        // memory access triggered by the kernel's internal shape logic.
        // ------------------------------------------------------------------
        if (keep_dims) {
            // 1) run reduction without keeping dims
            auto tmp_result = perform_axis_reduction(op_name, axis, /*keep_dims*/ false);
            // 2) compute final shape with singleton dims and reshape
            Shape final_shape = calculate_reduction_shape(axis, /*keep_dims*/ true);
            return tmp_result->reshape(final_shape);
        }

        // Calculate output shape
        Shape result_shape = calculate_reduction_shape(axis, /*keep_dims*/ false);
        
        // Allocate device memory for the result tensor
        auto result_tensor = std::make_shared<gpuTensor<T>>(result_shape);
        
        // Use stride-aware axis reduction kernels
        // Get input and output descriptors
        cuda_utils::TensorDescriptor input_desc = tensor_->descriptor();
        cuda_utils::TensorDescriptor output_desc = result_tensor->descriptor();
        
        // Get reduction axes as device array with proper error checking
        int num_reduction_axes = static_cast<int>(axis.size());
        int* d_reduction_axes = nullptr;
        int* d_input_shape = nullptr;
        int* d_input_strides = nullptr;
        int* d_output_strides = nullptr;
        
        // Allocate and copy reduction axes
        cudaError_t err = cudaMalloc(&d_reduction_axes, num_reduction_axes * sizeof(int));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for reduction axes: " + std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMemcpy(d_reduction_axes, axis.data(), num_reduction_axes * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_reduction_axes);
            throw std::runtime_error("Failed to copy reduction axes to device: " + std::string(cudaGetErrorString(err)));
        }
        
        // Allocate device memory for shape and strides arrays with error checking
        err = cudaMalloc(&d_input_shape, input_desc.ndims * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_reduction_axes);
            throw std::runtime_error("Failed to allocate device memory for input shape: " + std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMalloc(&d_input_strides, input_desc.ndims * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_reduction_axes);
            cudaFree(d_input_shape);
            throw std::runtime_error("Failed to allocate device memory for input strides: " + std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMalloc(&d_output_strides, output_desc.ndims * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_reduction_axes);
            cudaFree(d_input_shape);
            cudaFree(d_input_strides);
            throw std::runtime_error("Failed to allocate device memory for output strides: " + std::string(cudaGetErrorString(err)));
        }
        
        // Copy data to device with error checking
        err = cudaMemcpy(d_input_shape, input_desc.shape, input_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_reduction_axes);
            cudaFree(d_input_shape);
            cudaFree(d_input_strides);
            cudaFree(d_output_strides);
            throw std::runtime_error("Failed to copy input shape to device: " + std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMemcpy(d_input_strides, input_desc.strides, input_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_reduction_axes);
            cudaFree(d_input_shape);
            cudaFree(d_input_strides);
            cudaFree(d_output_strides);
            throw std::runtime_error("Failed to copy input strides to device: " + std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMemcpy(d_output_strides, output_desc.strides, output_desc.ndims * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_reduction_axes);
            cudaFree(d_input_shape);
            cudaFree(d_input_strides);
            cudaFree(d_output_strides);
            throw std::runtime_error("Failed to copy output strides to device: " + std::string(cudaGetErrorString(err)));
        }
        
        // Call the appropriate kernel based on operation and data type with error checking
        try {
            if constexpr (std::is_same_v<T, float>) {
                if (op_name == "sum") {
                    tensor_axis_sum_float32(result_tensor->data(), tensor_->data(),
                                            d_input_strides, d_input_shape,
                                            d_output_strides, d_reduction_axes,
                                            num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "mean") {
                    tensor_axis_mean_float32(result_tensor->data(), tensor_->data(),
                                             d_input_strides, d_input_shape,
                                             d_output_strides, d_reduction_axes,
                                             num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "max") {
                    tensor_axis_max_float32(result_tensor->data(), tensor_->data(),
                                             d_input_strides, d_input_shape,
                                             d_output_strides, d_reduction_axes,
                                             num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "min") {
                    tensor_axis_min_float32(result_tensor->data(), tensor_->data(),
                                             d_input_strides, d_input_shape,
                                             d_output_strides, d_reduction_axes,
                                             num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "prod") {
                    tensor_axis_prod_float32(result_tensor->data(), tensor_->data(),
                                              d_input_strides, d_input_shape,
                                              d_output_strides, d_reduction_axes,
                                              num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "var") {
                    tensor_axis_var_float32(result_tensor->data(), tensor_->data(),
                                             d_input_strides, d_input_shape,
                                             d_output_strides, d_reduction_axes,
                                             num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else {
                    throw std::runtime_error("Unknown reduction operation: " + op_name);
                }
            } else if constexpr (std::is_same_v<T, double>) {
                if (op_name == "sum") {
                    tensor_axis_sum_float64(result_tensor->data(), tensor_->data(),
                                             d_input_strides, d_input_shape,
                                             d_output_strides, d_reduction_axes,
                                             num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "mean") {
                    tensor_axis_mean_float64(result_tensor->data(), tensor_->data(),
                                              d_input_strides, d_input_shape,
                                              d_output_strides, d_reduction_axes,
                                              num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "max") {
                    tensor_axis_max_float64(result_tensor->data(), tensor_->data(),
                                             d_input_strides, d_input_shape,
                                             d_output_strides, d_reduction_axes,
                                             num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "min") {
                    tensor_axis_min_float64(result_tensor->data(), tensor_->data(),
                                              d_input_strides, d_input_shape,
                                              d_output_strides, d_reduction_axes,
                                              num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "prod") {
                    tensor_axis_prod_float64(result_tensor->data(), tensor_->data(),
                                               d_input_strides, d_input_shape,
                                               d_output_strides, d_reduction_axes,
                                               num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else if (op_name == "var") {
                    tensor_axis_var_float64(result_tensor->data(), tensor_->data(),
                                              d_input_strides, d_input_shape,
                                              d_output_strides, d_reduction_axes,
                                              num_reduction_axes, input_desc.ndims, result_tensor->size());
                } else {
                    throw std::runtime_error("Unknown reduction operation: " + op_name);
                }
            } else {
                // Fallback: compile for any T but raise at runtime if invoked.
                throw std::runtime_error("Axis-aware reduction not implemented for this dtype: " + dtype_to_string(tensor_->dtype()));
            }
            
            // Check for kernel execution errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Kernel execution failed for " + op_name + ": " + std::string(cudaGetErrorString(err)));
            }
            
            // Synchronize to ensure completion
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                throw std::runtime_error("Device synchronization failed for " + op_name + ": " + std::string(cudaGetErrorString(err)));
            }
        } catch (...) {
            // Ensure cleanup on any error
            cudaFree(d_reduction_axes);
            cudaFree(d_input_shape);
            cudaFree(d_input_strides);
            cudaFree(d_output_strides);
            throw; // Re-throw the exception
        }

        // Clean up device memory
        cudaFree(d_reduction_axes);
        cudaFree(d_input_shape);
        cudaFree(d_input_strides);
        cudaFree(d_output_strides);

        return std::make_unique<TensorWrapper<T>>(result_tensor);
    }
    
    // Helper to calculate output shape after reduction
    Shape calculate_reduction_shape(const std::vector<int>& axis, bool keep_dims) const {
        const Shape& input_shape = tensor_->shape();
        std::vector<size_t> result_dims;
        
        // Create set for fast lookup
        std::set<int> axis_set(axis.begin(), axis.end());
        
        for (int i = 0; i < static_cast<int>(input_shape.ndims()); ++i) {
            if (axis_set.find(i) != axis_set.end()) {
                // This dimension is being reduced
                if (keep_dims) {
                    result_dims.push_back(1);  // Keep as size 1
                }
                // Otherwise, remove this dimension
            } else {
                // This dimension is not being reduced
                result_dims.push_back(input_shape[i]);
            }
        }
        
        // If all dimensions were reduced and keep_dims=false, return scalar (shape [1])
        if (result_dims.empty()) {
            result_dims.push_back(1);
        }
        
        return Shape(result_dims);
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
            case DType::BOOL:
                return create_tensor<bool>(data, shape);
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
            case DType::BOOL:
                return create_empty_tensor<bool>(shape);
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