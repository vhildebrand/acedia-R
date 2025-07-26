#ifndef GPUTENSOR_H
#define GPUTENSOR_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half precision support
#include <cuda_bf16.h>  // For bfloat16 support
#include <stdexcept>
#include <vector>
#include <memory>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <typeinfo>

#ifndef __CUDA_ARCH__
#include <iostream>
#include <string>
#include <sstream>
#endif

#include <memory>
#include <vector>
#include <iostream>
#include <cassert>
#include <type_traits>
#include <cstring>
#include "cuda_utils.h"

// Forward declarations for CUDA kernels
extern "C" {
    void tensor_strided_copy_float32(float* dest, const float* src, const int* strides, const int* shape, int ndims, size_t total_elements);
    void tensor_strided_copy_float64(double* dest, const double* src, const int* strides, const int* shape, int ndims, size_t total_elements);
}

// Forward declarations
template<typename T> class gpuTensor;
template<typename T> class TensorView;
class ComputationGraph;

/**
 * @brief Device type enumeration
 */
enum class Device {
    CPU,
    CUDA
};

// BEGIN Column-major layout enforcement
enum class MemoryLayout { ColumnMajor, RowMajor };
static constexpr MemoryLayout kDefaultMemoryLayout = MemoryLayout::ColumnMajor;
static_assert(kDefaultMemoryLayout == MemoryLayout::ColumnMajor,
              "acediaR currently supports only column-major tensor layout; compile-time enforcement triggered.");
// END Column-major layout enforcement

/**
 * @brief Comprehensive data type enumeration for tensors
 */
enum class DType {
    FLOAT16,    // Half precision floating point
    BFLOAT16,   // Brain floating point (16-bit)
    FLOAT32,    // Single precision floating point
    FLOAT64,    // Double precision floating point
    INT8,       // 8-bit signed integer
    UINT8,      // 8-bit unsigned integer
    INT16,      // 16-bit signed integer
    UINT16,     // 16-bit unsigned integer
    INT32,      // 32-bit signed integer
    UINT32,     // 32-bit unsigned integer
    INT64,      // 64-bit signed integer
    UINT64,     // 64-bit unsigned integer
    BOOL        // Boolean type
};

/**
 * @brief Type traits for mapping C++ types to DType enums
 */
template<typename T>
struct dtype_traits;

template<> struct dtype_traits<half> { static constexpr DType value = DType::FLOAT16; };
template<> struct dtype_traits<__nv_bfloat16> { static constexpr DType value = DType::BFLOAT16; };
template<> struct dtype_traits<float> { static constexpr DType value = DType::FLOAT32; };
template<> struct dtype_traits<double> { static constexpr DType value = DType::FLOAT64; };
template<> struct dtype_traits<int8_t> { static constexpr DType value = DType::INT8; };
template<> struct dtype_traits<uint8_t> { static constexpr DType value = DType::UINT8; };
template<> struct dtype_traits<int16_t> { static constexpr DType value = DType::INT16; };
template<> struct dtype_traits<uint16_t> { static constexpr DType value = DType::UINT16; };
template<> struct dtype_traits<int32_t> { static constexpr DType value = DType::INT32; };
template<> struct dtype_traits<uint32_t> { static constexpr DType value = DType::UINT32; };
template<> struct dtype_traits<int64_t> { static constexpr DType value = DType::INT64; };
template<> struct dtype_traits<uint64_t> { static constexpr DType value = DType::UINT64; };
template<> struct dtype_traits<bool> { static constexpr DType value = DType::BOOL; };

/**
 * @brief Type traits for checking if a type is floating point
 */
template<typename T>
struct is_floating_point : std::false_type {};

template<> struct is_floating_point<half> : std::true_type {};
template<> struct is_floating_point<__nv_bfloat16> : std::true_type {};
template<> struct is_floating_point<float> : std::true_type {};
template<> struct is_floating_point<double> : std::true_type {};

/**
 * @brief Type traits for checking if a type is integer
 */
template<typename T>
struct is_integer_type : std::false_type {};

template<> struct is_integer_type<int8_t> : std::true_type {};
template<> struct is_integer_type<uint8_t> : std::true_type {};
template<> struct is_integer_type<int16_t> : std::true_type {};
template<> struct is_integer_type<uint16_t> : std::true_type {};
template<> struct is_integer_type<int32_t> : std::true_type {};
template<> struct is_integer_type<uint32_t> : std::true_type {};
template<> struct is_integer_type<int64_t> : std::true_type {};
template<> struct is_integer_type<uint64_t> : std::true_type {};

/**
 * @brief Get size in bytes for a data type
 */
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT16: return sizeof(half);
        case DType::BFLOAT16: return sizeof(__nv_bfloat16);
        case DType::FLOAT32: return sizeof(float);
        case DType::FLOAT64: return sizeof(double);
        case DType::INT8: return sizeof(int8_t);
        case DType::UINT8: return sizeof(uint8_t);
        case DType::INT16: return sizeof(int16_t);
        case DType::UINT16: return sizeof(uint16_t);
        case DType::INT32: return sizeof(int32_t);
        case DType::UINT32: return sizeof(uint32_t);
        case DType::INT64: return sizeof(int64_t);
        case DType::UINT64: return sizeof(uint64_t);
        case DType::BOOL: return sizeof(bool);
        default: return 0;
    }
}

/**
 * @brief Type promotion rules for mixed operations
 */
inline DType promote_types(DType a, DType b) {
    // If types are the same, return that type
    if (a == b) return a;
    
    // Floating point types take precedence
    if (a == DType::FLOAT64 || b == DType::FLOAT64) return DType::FLOAT64;
    if (a == DType::FLOAT32 || b == DType::FLOAT32) return DType::FLOAT32;
    if (a == DType::BFLOAT16 || b == DType::BFLOAT16) return DType::BFLOAT16;
    if (a == DType::FLOAT16 || b == DType::FLOAT16) return DType::FLOAT16;
    
    // Integer type promotion (promote to larger type)
    std::vector<DType> int_hierarchy = {
        DType::BOOL, DType::INT8, DType::UINT8, DType::INT16, DType::UINT16,
        DType::INT32, DType::UINT32, DType::INT64, DType::UINT64
    };
    
    auto pos_a = std::find(int_hierarchy.begin(), int_hierarchy.end(), a);
    auto pos_b = std::find(int_hierarchy.begin(), int_hierarchy.end(), b);
    
    if (pos_a != int_hierarchy.end() && pos_b != int_hierarchy.end()) {
        return std::max(a, b, [&](DType x, DType y) {
            auto pos_x = std::find(int_hierarchy.begin(), int_hierarchy.end(), x);
            auto pos_y = std::find(int_hierarchy.begin(), int_hierarchy.end(), y);
            return pos_x < pos_y;
        });
    }
    
    // Default to float32 for mixed operations
    return DType::FLOAT32;
}

#ifndef __CUDA_ARCH__
/**
 * @brief Get string representation of DType
 */
inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::FLOAT16: return "float16";
        case DType::BFLOAT16: return "bfloat16";
        case DType::FLOAT32: return "float";    // R standard name
        case DType::FLOAT64: return "double";   // R standard name
        case DType::INT8: return "int8";
        case DType::UINT8: return "uint8";
        case DType::INT16: return "int16";
        case DType::UINT16: return "uint16";
        case DType::INT32: return "int32";
        case DType::UINT32: return "uint32";
        case DType::INT64: return "int64";
        case DType::UINT64: return "uint64";
        case DType::BOOL: return "bool";
        default: return "unknown";
    }
}

/**
 * @brief Parse string to DType
 */
inline DType string_to_dtype(const std::string& str) {
    static std::unordered_map<std::string, DType> dtype_map = {
        {"float16", DType::FLOAT16}, {"half", DType::FLOAT16},
        {"bfloat16", DType::BFLOAT16}, {"bf16", DType::BFLOAT16},
        {"float32", DType::FLOAT32}, {"float", DType::FLOAT32},
        {"float64", DType::FLOAT64}, {"double", DType::FLOAT64},
        {"int8", DType::INT8}, {"uint8", DType::UINT8},
        {"int16", DType::INT16}, {"uint16", DType::UINT16},
        {"int32", DType::INT32}, {"uint32", DType::UINT32},
        {"int64", DType::INT64}, {"uint64", DType::UINT64},
        {"bool", DType::BOOL}
    };
    
    auto it = dtype_map.find(str);
    if (it != dtype_map.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown dtype string: " + str);
}
#endif

/**
 * @brief Shape class for tensor dimensions (unchanged)
 */
class Shape {
public:
    std::vector<size_t> dims;
    
    Shape() = default;
    Shape(std::initializer_list<size_t> shape) : dims(shape) {}
    Shape(const std::vector<size_t>& shape) : dims(shape) {}
    
    size_t ndims() const { return dims.size(); }
    size_t size() const { 
        return std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<size_t>());
    }
    
    size_t operator[](size_t idx) const { return dims[idx]; }
    size_t& operator[](size_t idx) { return dims[idx]; }
    
    bool operator==(const Shape& other) const { return dims == other.dims; }
    bool operator!=(const Shape& other) const { return dims != other.dims; }
    
    bool broadcastable_with(const Shape& other) const {
        const Shape& a = *this;
        const Shape& b = other;

        size_t len_a = a.ndims();
        size_t len_b = b.ndims();
        size_t max_len = std::max(len_a, len_b);

        // Helper lambda that checks compatibility given an offset for the shorter tensor
        auto is_compatible = [&](const Shape& shorter, const Shape& longer, size_t offset) {
            for (size_t i = 0; i < max_len; ++i) {
                size_t dim_short = 1;
                if (i >= offset && i < offset + shorter.ndims()) {
                    dim_short = shorter.dims[i - offset];
                }
                size_t dim_long = longer.dims[i];

                if (dim_short != dim_long && dim_short != 1 && dim_long != 1) return false;
            }
            return true;
        };

        // If equal length, simple element-wise check
        if (len_a == len_b) {
            return is_compatible(a, b, 0); // either order works
        }

        // Identify shorter and longer
        const Shape& shorter = (len_a < len_b) ? a : b;
        const Shape& longer  = (len_a < len_b) ? b : a;
        size_t len_short = shorter.ndims();

        // Try every possible offset where the shorter shape could align inside longer
        for (size_t offset = 0; offset <= max_len - len_short; ++offset) {
            if (is_compatible(shorter, longer, offset)) return true;
        }

        return false;
    }
 
    Shape broadcast_with(const Shape& other) const {
        if (!broadcastable_with(other)) {
            throw std::runtime_error("Shapes are not broadcastable");
        }

        size_t max_len = std::max(ndims(), other.ndims());
        std::vector<size_t> out_dims(max_len, 1);

        // We'll compute using the offset that works (prefer the first that matches)
        const Shape& a = *this;
        const Shape& b = other;

        size_t len_a = a.ndims();
        size_t len_b = b.ndims();

        auto fill_dims = [&](const Shape& shorter, const Shape& longer, size_t offset) {
            for (size_t i = 0; i < max_len; ++i) {
                size_t dim_short = 1;
                if (i >= offset && i < offset + shorter.ndims()) {
                    dim_short = shorter.dims[i - offset];
                }
                size_t dim_long = longer.dims[i];
                out_dims[i] = std::max(dim_short, dim_long);
            }
        };

        if (len_a == len_b) {
            fill_dims(a, b, 0);
        } else if (len_a < len_b) {
            // Sweep offsets for a inside b
            for (size_t off = 0; off <= max_len - len_a; ++off) {
                bool ok = true;
                for (size_t i = 0; i < max_len; ++i) {
                    size_t dim_a = (i >= off && i < off + len_a) ? a.dims[i - off] : 1;
                    size_t dim_b = b.dims[i];
                    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) { ok = false; break; }
                }
                if (ok) { fill_dims(a, b, off); break; }
            }
        } else { // len_b < len_a
            for (size_t off = 0; off <= max_len - len_b; ++off) {
                bool ok = true;
                for (size_t i = 0; i < max_len; ++i) {
                    size_t dim_b = (i >= off && i < off + len_b) ? b.dims[i - off] : 1;
                    size_t dim_a = a.dims[i];
                    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) { ok = false; break; }
                }
                if (ok) { fill_dims(b, a, off); break; }
            }
        }

        return Shape(out_dims);
    }

#ifndef __CUDA_ARCH__
    std::string to_string() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << dims[i];
        }
        oss << "]";
        return oss.str();
    }
#endif
};

/**
 * @brief Compute strides for column-major (R/Fortran-style) memory layout
 */
inline std::vector<size_t> compute_strides(const Shape& shape) {
    if (shape.ndims() == 0) return {};
    
    std::vector<size_t> strides(shape.ndims());
    strides[0] = 1;
    
    for (size_t i = 1; i < shape.ndims(); ++i) {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    
    return strides;
}

/**
 * @brief CUDA Stream wrapper for async operations (unchanged)
 */
class CudaStream {
private:
    cudaStream_t stream_;
    bool owns_stream_;

public:
    CudaStream() : owns_stream_(true) {
        cudaError_t err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
        }
    }
    
    explicit CudaStream(cudaStream_t stream) : stream_(stream), owns_stream_(false) {}
    
    ~CudaStream() {
        if (owns_stream_ && stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    CudaStream(CudaStream&& other) noexcept 
        : stream_(other.stream_), owns_stream_(other.owns_stream_) {
        other.stream_ = nullptr;
        other.owns_stream_ = false;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (owns_stream_ && stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            owns_stream_ = other.owns_stream_;
            other.stream_ = nullptr;
            other.owns_stream_ = false;
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    
    void synchronize() {
        cudaError_t err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Stream synchronization failed: " + std::string(cudaGetErrorString(err)));
        }
    }
};

/**
 * @brief Main GPU Tensor class - fully templated for any numeric type
 */
template<typename T>
class gpuTensor {
private:
    T* data_;                              // Device pointer to data
    Shape shape_;                          // Tensor shape
    std::vector<size_t> strides_;         // Memory strides
    Device device_;                        // Device location
    DType dtype_;                          // Data type
    size_t offset_;                        // Offset for views
    std::shared_ptr<T> storage_;          // Shared storage for views
    size_t storage_size_;                 // Total storage size
    std::shared_ptr<CudaStream> stream_;  // CUDA stream for async ops
    
    // Autograd support
    bool requires_grad_flag_;
    std::shared_ptr<ComputationGraph> grad_fn_;
    std::shared_ptr<gpuTensor<T>> grad_;

    void cudaSafeCall(cudaError_t err, const char* msg) const {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    }
    
    void allocate_storage() {
        if (device_ == Device::CPU) {
            throw std::runtime_error("CPU tensors not yet implemented");
        }
        
        T* raw_ptr;
        cudaError_t err = cudaMalloc(&raw_ptr, storage_size_ * sizeof(T));
        cudaSafeCall(err, "Failed to allocate GPU memory in gpuTensor");
        
        storage_ = std::shared_ptr<T>(raw_ptr, [](T* ptr) {
            if (ptr) cudaFree(ptr);
        });
        
        data_ = storage_.get() + offset_;
    }

public:
    // Constructors
    explicit gpuTensor(const Shape& shape, Device device = Device::CUDA)
        : shape_(shape), device_(device), dtype_(dtype_traits<T>::value), offset_(0), 
          requires_grad_flag_(false), storage_size_(shape.size()) {
        
        strides_ = compute_strides(shape_);
        stream_ = std::make_shared<CudaStream>();
        allocate_storage();
    }
    
    // Constructor from host data
    gpuTensor(const T* host_data, const Shape& shape, Device device = Device::CUDA)
        : gpuTensor(shape, device) {
        
        if (host_data && shape.size() > 0) {
            cudaError_t err = cudaMemcpy(data_, host_data, shape.size() * sizeof(T), cudaMemcpyHostToDevice);
            cudaSafeCall(err, "Failed to copy data to GPU in gpuTensor constructor");
        }
    }
    
    // Constructor from std::vector
    gpuTensor(const std::vector<T>& host_data, const Shape& shape, Device device = Device::CUDA)
        : gpuTensor(get_vector_data(host_data), shape, device) {
        
        if (host_data.size() != shape.size()) {
            throw std::runtime_error("Data size doesn't match shape size");
        }
    }
    
private:
    // Helper to handle std::vector<bool> special case
    static const T* get_vector_data(const std::vector<T>& vec) {
        if constexpr (std::is_same_v<T, bool>) {
            // std::vector<bool> is special and doesn't have data()
            // For now, we'll need a workaround - convert to regular bool array
            static_assert(std::is_same_v<T, bool>, "This path should only be taken for bool");
            throw std::runtime_error("std::vector<bool> is not directly supported. Use std::vector<uint8_t> instead.");
        } else {
            return vec.data();
        }
    }
    
public:
    
    // View constructor (shares storage)
    gpuTensor(std::shared_ptr<T> storage, const Shape& shape, const std::vector<size_t>& strides, 
              size_t offset, Device device, std::shared_ptr<CudaStream> stream)
        : storage_(storage), shape_(shape), strides_(strides), offset_(offset), 
          device_(device), dtype_(dtype_traits<T>::value), stream_(stream), 
          requires_grad_flag_(false), storage_size_(0) {
        data_ = storage_.get() + offset_;
    }
    
    // Copy constructor (deep copy)
    gpuTensor(const gpuTensor& other) 
        : shape_(other.shape_), strides_(other.strides_), device_(other.device_), 
          dtype_(other.dtype_), offset_(0), requires_grad_flag_(other.requires_grad_flag_),
          storage_size_(other.shape_.size()) {
        
        stream_ = std::make_shared<CudaStream>();
        allocate_storage();
        
        if (other.data_ && shape_.size() > 0) {
            cudaError_t err = cudaMemcpy(data_, other.data_, shape_.size() * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaSafeCall(err, "Failed to copy GPU data in gpuTensor copy constructor");
        }
    }
    
    // Move constructor
    gpuTensor(gpuTensor&& other) noexcept
        : data_(other.data_), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
          device_(other.device_), dtype_(other.dtype_), offset_(other.offset_),
          storage_(std::move(other.storage_)), storage_size_(other.storage_size_),
          stream_(std::move(other.stream_)), requires_grad_flag_(other.requires_grad_flag_),
          grad_fn_(std::move(other.grad_fn_)), grad_(std::move(other.grad_)) {
        
        other.data_ = nullptr;
        other.storage_size_ = 0;
        other.offset_ = 0;
    }
    
    // Assignment operators
    gpuTensor& operator=(const gpuTensor& other) {
        if (this != &other) {
            *this = gpuTensor(other);
        }
        return *this;
    }
    
    gpuTensor& operator=(gpuTensor&& other) noexcept {
        if (this != &other) {
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            device_ = other.device_;
            dtype_ = other.dtype_;
            offset_ = other.offset_;
            storage_ = std::move(other.storage_);
            storage_size_ = other.storage_size_;
            stream_ = std::move(other.stream_);
            requires_grad_flag_ = other.requires_grad_flag_;
            grad_fn_ = std::move(other.grad_fn_);
            grad_ = std::move(other.grad_);
            
            other.data_ = nullptr;
            other.storage_size_ = 0;
            other.offset_ = 0;
        }
        return *this;
    }
    
    // Accessors
    const Shape& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    Device device() const { return device_; }
    DType dtype() const { return dtype_; }
    size_t size() const { return shape_.size(); }
    size_t ndims() const { return shape_.ndims(); }
    T* data() { return data_; }
    const T* data() const { return data_; }
    bool requires_grad() const { return requires_grad_flag_; }
    
    // Get TensorDescriptor for stride-aware operations
    cuda_utils::TensorDescriptor descriptor() const {
        cuda_utils::TensorDescriptor desc;
        desc.data = static_cast<void*>(data_);
        desc.ndims = static_cast<int>(ndims());
        desc.offset = offset_;
        desc.total_size = size();
        
        // Copy shape and strides (convert size_t to int)
        for (int i = 0; i < desc.ndims && i < 8; ++i) {
            desc.shape[i] = static_cast<int>(shape_.dims[i]);
            desc.strides[i] = static_cast<int>(strides_[i]);
        }
        
        return desc;
    }
    
    // Enable gradient computation
    gpuTensor& requires_grad_(bool requires = true) {
        requires_grad_flag_ = requires;
        return *this;
    }
    
    // Type conversion methods
    template<typename U>
    gpuTensor<U> to() const {
        gpuTensor<U> result(shape_, device_);
        // TODO: Implement type conversion kernel
        return result;
    }
    
    // Convenience type conversion methods
    gpuTensor<float> to_float() const { return to<float>(); }
    gpuTensor<double> to_double() const { return to<double>(); }
    gpuTensor<half> to_half() const { return to<half>(); }
    gpuTensor<__nv_bfloat16> to_bfloat16() const { return to<__nv_bfloat16>(); }
    
    // Check if tensor is contiguous
    bool is_contiguous() const {
        auto expected_strides = compute_strides(shape_);
        return strides_ == expected_strides;
    }
    
    // Create a view with new shape - SUPPORTS NON-CONTIGUOUS TENSORS
    gpuTensor view(const Shape& new_shape) {
        if (new_shape.size() != shape_.size()) {
            throw std::runtime_error("View shape size must match original size");
        }
        
        // For non-contiguous tensors, create contiguous copy first, then view
        if (!is_contiguous()) {
            gpuTensor contiguous_copy = contiguous(); 
            auto new_strides = compute_strides(new_shape);
            return gpuTensor(contiguous_copy.storage_, new_shape, new_strides, 
                           contiguous_copy.offset_, device_, stream_);
        }
        
        // For contiguous tensors, create view directly
        auto new_strides = compute_strides(new_shape);
        return gpuTensor(storage_, new_shape, new_strides, offset_, device_, stream_);
    }
    
    // Create a reshaped tensor
    gpuTensor reshape(const Shape& new_shape) {
        if (new_shape.size() != shape_.size()) {
            throw std::runtime_error("Reshape size must match original size");
        }
        
        if (is_contiguous()) {
            return view(new_shape);
        } else {
            gpuTensor contiguous_copy = contiguous();
            return contiguous_copy.view(new_shape);
        }
    }
    
    // Transpose (2D only for now) - EFFICIENT VIEW IMPLEMENTATION
    gpuTensor transpose() const {
        if (ndims() != 2) {
            throw std::runtime_error("Transpose currently supports 2D tensors only");
        }
        
        // Create transpose as a VIEW by swapping dimensions and strides
        size_t rows = shape_[0];
        size_t cols = shape_[1]; 
        
        // New shape: swap dimensions
        Shape new_shape({cols, rows});
        
        // New strides: swap strides to achieve transpose effect
        std::vector<size_t> new_strides(2);
        new_strides[0] = strides_[1];  // First dim gets old second dim stride
        new_strides[1] = strides_[0];  // Second dim gets old first dim stride
        
        // Create view that shares storage but has transposed layout
        return gpuTensor(storage_, new_shape, new_strides, offset_, device_, stream_);
    }
    
    // Permute dimensions as VIEW (PyTorch-like behavior)
    gpuTensor permute(const std::vector<int>& dims) const {
        if (dims.size() != ndims()) {
            throw std::runtime_error("Permute dimensions must match tensor dimensions");
        }
        
        // Validate dimensions
        std::vector<bool> used(ndims(), false);
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] < 0 || dims[i] >= (int)ndims() || used[dims[i]]) {
                throw std::runtime_error("Invalid permutation dimensions");
            }
            used[dims[i]] = true;
        }
        
        // Create permuted view by rearranging shape and strides
        std::vector<size_t> new_shape_dims(ndims());
        std::vector<size_t> new_strides(ndims());
        
        for (size_t i = 0; i < ndims(); ++i) {
            new_shape_dims[i] = shape_[dims[i]];  // New shape from permuted dimensions
            new_strides[i] = strides_[dims[i]];   // New strides from permuted dimensions
        }
        
        Shape new_shape(new_shape_dims);
        
        // Create view that shares storage but has permuted layout
        return gpuTensor(storage_, new_shape, new_strides, offset_, device_, stream_);
    }
    
    // Create a contiguous copy
    gpuTensor contiguous() const {
        if (is_contiguous()) {
            return *this;
        }

#ifndef __CUDA_ARCH__
        // GPU-native strided copy - much faster than host round-trip
        gpuTensor result(shape_, device_);
        
        // Convert strides from size_t to int for CUDA kernel
        std::vector<int> strides_int(strides_.begin(), strides_.end());
        std::vector<int> shape_int(shape_.dims.begin(), shape_.dims.end());
        
        // Call appropriate strided copy kernel based on type
        if constexpr (std::is_same_v<T, float>) {
            tensor_strided_copy_float32(
                result.data(), data_, 
                strides_int.data(), shape_int.data(), 
                shape_.ndims(), shape_.size()
            );
        } else if constexpr (std::is_same_v<T, double>) {
            tensor_strided_copy_float64(
                result.data(), data_, 
                strides_int.data(), shape_int.data(), 
                shape_.ndims(), shape_.size()
            );
        } else {
            // Fallback to host round-trip for unsupported types
            std::vector<T> host_buf(shape_.size());
            copy_to_host(host_buf.data());
            result.copy_from_host(host_buf.data());
        }
        
        return result;
#else
        // Device code – cannot allocate device memory, throw for now
        assert(false && "contiguous() not supported in device code");
        return *this;
#endif
    }
    
    // Data transfer methods
    void copy_to_host(T* host_ptr) const {
        if (!data_ || shape_.size() == 0) return;
        
        if (is_contiguous()) {
            // Fast path for contiguous tensors
            cudaError_t err = cudaMemcpy(host_ptr, data_, shape_.size() * sizeof(T), cudaMemcpyDeviceToHost);
            cudaSafeCall(err, "Failed to copy contiguous data from GPU to host");
        } else {
            // Handle non-contiguous tensors by first making them contiguous
            gpuTensor<T> contiguous_copy = contiguous();
            contiguous_copy.copy_to_host(host_ptr);
        }
    }
    
    void copy_from_host(const T* host_ptr) {
        if (!data_ || shape_.size() == 0) return;
        
        if (!is_contiguous()) {
            throw std::runtime_error("Cannot copy to non-contiguous tensor from host. Use contiguous() first.");
        }
        
        cudaError_t err = cudaMemcpy(data_, host_ptr, shape_.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaSafeCall(err, "Failed to copy data from host to GPU");
    }
    
    // Convert to host vector (host-only function)
    std::vector<T> to_host() const {
#ifdef __CUDA_ARCH__
        // This should never be called from device code
        return std::vector<T>();
#else
        if (shape_.size() == 0) return std::vector<T>();
        
        std::vector<T> result(shape_.size());
        copy_to_host(result.data());
        return result;
#endif
    }
    

    
    // Print tensor info (always available)
    std::string info() const {
#ifdef __CUDA_ARCH__
        // Device code version - simplified
        return "gpuTensor(device_code)";
#else
        std::ostringstream oss;
        oss << "gpuTensor<" << dtype_to_string(dtype_) << ">(shape=" << shape_.to_string() 
            << ", device=" << (device_ == Device::CUDA ? "CUDA" : "CPU")
            << ", requires_grad=" << (requires_grad_flag_ ? "True" : "False") << ")";
        return oss.str();
#endif
    }

    // Stream operations
    std::shared_ptr<CudaStream> get_stream() const { return stream_; }
    
    void synchronize() {
        if (stream_) {
            stream_->synchronize();
        }
    }
};

// Type aliases for common tensor types
using Float16Tensor = gpuTensor<half>;
using BFloat16Tensor = gpuTensor<__nv_bfloat16>;
using FloatTensor = gpuTensor<float>;
using DoubleTensor = gpuTensor<double>;
using Int8Tensor = gpuTensor<int8_t>;
using UInt8Tensor = gpuTensor<uint8_t>;
using Int16Tensor = gpuTensor<int16_t>;
using UInt16Tensor = gpuTensor<uint16_t>;
using IntTensor = gpuTensor<int32_t>;
using UIntTensor = gpuTensor<uint32_t>;
using LongTensor = gpuTensor<int64_t>;
using ULongTensor = gpuTensor<uint64_t>;
using BoolTensor = gpuTensor<bool>;

#endif // GPUTENSOR_H 