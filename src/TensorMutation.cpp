#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"

using namespace Rcpp;

// Forward declarations of the CUDA wrappers (defined in tensor_slice_update.cu)
extern "C" {
    void tensor_slice_add_scalar_float32(
        float* data, double scalar,
        const int* start, const int* slice_shape,
        const int* strides, int ndims, size_t total_elements);
    void tensor_slice_add_scalar_float64(
        double* data, double scalar,
        const int* start, const int* slice_shape,
        const int* strides, int ndims, size_t total_elements);
}

// Utility: convert Shape to std::vector<int> strides (column-major for R compatibility)
static std::vector<int> shape_to_strides_int(const Shape& shape) {
    if (shape.ndims() == 0) return {};
    
    std::vector<int> strides_int(shape.ndims());
    strides_int[0] = 1;
    
    // Column-major: stride[i] = stride[i-1] * dim[i-1]
    for (size_t i = 1; i < shape.ndims(); ++i) {
        strides_int[i] = strides_int[i - 1] * static_cast<int>(shape.dims[i - 1]);
    }
    
    return strides_int;
}

// [[Rcpp::export]]
void tensor_slice_add_scalar_unified(SEXP tensor_ptr,
                                     IntegerVector start_indices,
                                     IntegerVector slice_shape,
                                     double scalar) {
    if (start_indices.size() != slice_shape.size()) {
        stop("start_indices and slice_shape must have the same length");
    }
    int ndims = start_indices.size();

    // Copy start and shape to std::vector<int>
    std::vector<int> start(ndims);
    std::vector<int> slice(ndims);
    size_t total_elements = 1;
    for (int i = 0; i < ndims; ++i) {
        // convert R 1-based to 0-based
        start[i]  = static_cast<int>(start_indices[i] - 1);
        slice[i]  = static_cast<int>(slice_shape[i]);
        if (slice[i] <= 0) {
            stop("slice_shape entries must be positive");
        }
        total_elements *= static_cast<size_t>(slice[i]);
    }

    try {
        XPtr<TensorBase> base_ptr(tensor_ptr);
        if (!base_ptr) {
            stop("Invalid tensor pointer");
        }

        // Derive strides
        auto strides_int = shape_to_strides_int(base_ptr->shape());

        if (base_ptr->dtype() == DType::FLOAT32) {
            auto* typed = dynamic_cast<TensorWrapper<float>*>(base_ptr.get());
            if (!typed) stop("Type cast failed for float32 tensor");
            FloatTensor& tensor = typed->tensor();
            tensor_slice_add_scalar_float32(
                tensor.data(), scalar,
                start.data(), slice.data(), strides_int.data(), ndims, total_elements);
        } else if (base_ptr->dtype() == DType::FLOAT64) {
            auto* typed = dynamic_cast<TensorWrapper<double>*>(base_ptr.get());
            if (!typed) stop("Type cast failed for float64 tensor");
            DoubleTensor& tensor = typed->tensor();
            tensor_slice_add_scalar_float64(
                tensor.data(), scalar,
                start.data(), slice.data(), strides_int.data(), ndims, total_elements);
        } else {
            stop("tensor_slice_add_scalar_unified currently supports float32 and float64 tensors only");
        }
    } catch (const std::exception& e) {
        stop(std::string("Error in tensor_slice_add_scalar_unified: ") + e.what());
    }
} 