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
        case DType::FLOAT32: return "float32";
        case DType::FLOAT64: return "float64";
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
        size_t max_dims = std::max(ndims(), other.ndims());
        for (size_t i = 0; i < max_dims; ++i) {
            size_t dim1 = (i < ndims()) ? dims[ndims() - 1 - i] : 1;
            size_t dim2 = (i < other.ndims()) ? other.dims[other.ndims() - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                return false;
            }
        }
        return true;
    }
    
    Shape broadcast_with(const Shape& other) const {
        if (!broadcastable_with(other)) {
            throw std::runtime_error("Shapes are not broadcastable");
        }
        
        size_t max_dims = std::max(ndims(), other.ndims());
        std::vector<size_t> result_dims(max_dims);
        
        for (size_t i = 0; i < max_dims; ++i) {
            size_t dim1 = (i < ndims()) ? dims[ndims() - 1 - i] : 1;
            size_t dim2 = (i < other.ndims()) ? other.dims[other.ndims() - 1 - i] : 1;
            result_dims[max_dims - 1 - i] = std::max(dim1, dim2);
        }
        
        return Shape(result_dims);
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
 * @brief Compute strides for row-major (C-style) memory layout (unchanged)
 */
inline std::vector<size_t> compute_strides(const Shape& shape) {
    if (shape.ndims() == 0) return {};
    
    std::vector<size_t> strides(shape.ndims());
    strides[shape.ndims() - 1] = 1;
    
    for (int i = static_cast<int>(shape.ndims()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
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
    
    // Create a view with new shape
    gpuTensor view(const Shape& new_shape) {
        if (new_shape.size() != shape_.size()) {
            throw std::runtime_error("View shape size must match original size");
        }
        
        if (!is_contiguous()) {
            throw std::runtime_error("View requires contiguous tensor");
        }
        
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
    
    // Create a contiguous copy
    gpuTensor contiguous() const {
        if (is_contiguous()) {
            return *this;
        }
        
        gpuTensor result(shape_, device_);
        // TODO: Implement strided copy kernel
        throw std::runtime_error("Non-contiguous tensor copying not yet implemented");
        return result;
    }
    
    // Data transfer methods
    void copy_to_host(T* host_ptr) const {
        if (!data_ || shape_.size() == 0) return;
        
        if (!is_contiguous()) {
            throw std::runtime_error("Cannot copy non-contiguous tensor to host");
        }
        
        cudaError_t err = cudaMemcpy(host_ptr, data_, shape_.size() * sizeof(T), cudaMemcpyDeviceToHost);
        cudaSafeCall(err, "Failed to copy data from GPU to host");
    }
    
    void copy_from_host(const T* host_ptr) {
        if (!data_ || shape_.size() == 0) return;
        
        if (!is_contiguous()) {
            throw std::runtime_error("Cannot copy to non-contiguous tensor from host");
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