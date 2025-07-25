#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <vector>

using namespace Rcpp;

// Forward declarations of CUDA launch functions (defined in tensor_ops.cu)
extern "C" {
    // Concat launch functions
    void launch_concat_float32(float* result, const float** inputs, const int* input_sizes, int num_tensors,
                               const int* result_strides, const int* input_strides_list, 
                               const int* shape, int ndims, int concat_axis, size_t total_elements);
         void launch_concat_float64(double* result, const double** inputs, const int* input_sizes, int num_tensors,
                               const int* result_strides, const int* input_strides_list, 
                               const int* shape, int ndims, int concat_axis, size_t total_elements);
    
    // Stack launch functions
    void launch_stack_float32(float* result, const float** inputs, int num_tensors,
                              const int* input_strides, const int* result_shape, int ndims,
                              int stack_axis, size_t total_elements);
    void launch_stack_float64(double* result, const double** inputs, int num_tensors,
                              const int* input_strides, const int* result_shape, int ndims,
                              int stack_axis, size_t total_elements);
    
    // Repeat launch functions
    void launch_repeat_float32(float* result, const float* input,
                               const int* input_strides, const int* repeat_counts,
                               const int* input_shape, const int* result_shape, int ndims,
                               size_t total_elements);
    void launch_repeat_float64(double* result, const double* input,
                               const int* input_strides, const int* repeat_counts,
                               const int* input_shape, const int* result_shape, int ndims,
                               size_t total_elements);
    
    // Pad launch functions
    void launch_pad_float32(float* result, const float* input,
                            const int* input_strides, const int* input_shape,
                            const int* pad_before, const int* pad_after,
                            const int* result_shape, int ndims, float pad_value, int pad_mode,
                            size_t total_elements);
    void launch_pad_float64(double* result, const double* input,
                            const int* input_strides, const int* input_shape,
                            const int* pad_before, const int* pad_after,
                            const int* result_shape, int ndims, double pad_value, int pad_mode,
                            size_t total_elements);
}

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
    
    // New slice assignment functions
    void tensor_slice_set_scalar_float32(
        float* data, double scalar,
        const int* start, const int* slice_shape,
        const int* strides, int ndims, size_t total_elements);
    void tensor_slice_set_scalar_float64(
        double* data, double scalar,
        const int* start, const int* slice_shape,
        const int* strides, int ndims, size_t total_elements);
    
    void tensor_slice_set_tensor_float32(
        float* dest_data, const float* src_data,
        const int* start, const int* slice_shape,
        const int* dest_strides, const int* src_strides,
        int ndims, size_t total_elements);
    void tensor_slice_set_tensor_float64(
        double* dest_data, const double* src_data,
        const int* start, const int* slice_shape,
        const int* dest_strides, const int* src_strides,
        int ndims, size_t total_elements);
    
    // Boolean mask assignment functions
    void tensor_mask_set_scalar_float32(
        float* data, const bool* mask, float value, size_t total_elements);
    void tensor_mask_set_scalar_float64(
        double* data, const bool* mask, double value, size_t total_elements);
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

// Utility helper: convert size_t vector to int vector
static std::vector<int> to_int_vec(const std::vector<size_t>& v) {
    std::vector<int> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<int>(v[i]);
    return out;
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

// ======================= CONCAT ======================= //
// [[Rcpp::export]]
SEXP tensor_concat_unified(List tensor_list, int axis = 1) {
    if (tensor_list.size() < 2) stop("Need at least two tensors to concatenate");
    axis--; // convert to 0-indexed

    // Validate tensors and gather pointers
    std::vector<XPtr<TensorBase>> tensors;
    tensors.reserve(tensor_list.size());
    for (SEXP t : tensor_list) {
        tensors.emplace_back(t);
        if (!tensors.back()) stop("Invalid tensor pointer in list");
    }

    // Ensure same dtype
    DType dtype = tensors[0]->dtype();
    for (auto& tp : tensors) {
        if (tp->dtype() != dtype) stop("All tensors must have same dtype for concat");
    }

    // Determine result shape
    Shape base_shape = tensors[0]->shape();
    if (axis < 0 || axis >= (int)base_shape.ndims())
        stop("axis out of range");
    size_t concat_dim = 0;
    for (auto& tp : tensors) {
        Shape s = tp->shape();
        if (s.ndims() != base_shape.ndims()) stop("All tensors must have same number of dims");
        for (size_t d = 0; d < s.ndims(); ++d) {
            if ((int)d == axis) continue;
            if (s[d] != base_shape[d]) stop("Non-concat dimensions must match");
        }
        concat_dim += s[axis];
    }
    std::vector<size_t> res_dims(base_shape.dims);
    res_dims[axis] = concat_dim;
    Shape result_shape(res_dims);

    // Create result tensor
    std::unique_ptr<TensorBase> result_tensor;

    size_t total_elements = result_shape.size();
    int num_tensors = tensors.size();

    // Gather strides and device pointers
    if (dtype == DType::FLOAT32) {
        std::vector<const float*> d_ptrs(num_tensors);
        std::vector<std::vector<int>> input_strides_list;
        std::vector<int> input_sizes(num_tensors);
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto tw = dynamic_cast<const TensorWrapper<float>*>(tensors[i].get());
            if (!tw) throw std::runtime_error("Wrapper cast failed");
            d_ptrs[i] = tw->tensor().data();
            input_strides_list.push_back(to_int_vec(tw->tensor().strides()));
            input_sizes[i] = static_cast<int>(tw->tensor().shape()[axis]);
        }
        
        // Flatten input strides for C function
        std::vector<int> flattened_strides;
        for (const auto& strides : input_strides_list) {
            flattened_strides.insert(flattened_strides.end(), strides.begin(), strides.end());
        }
        
        auto res_gpu = std::make_shared<gpuTensor<float>>(result_shape);
        auto result_strides = to_int_vec(res_gpu->strides());
        auto shape_vec = to_int_vec(res_dims);
        
        launch_concat_float32(res_gpu->data(), d_ptrs.data(), input_sizes.data(), num_tensors,
                             result_strides.data(), flattened_strides.data(),
                             shape_vec.data(), base_shape.ndims(), axis, total_elements);
        result_tensor = std::make_unique<TensorWrapper<float>>(res_gpu);
    } else if (dtype == DType::FLOAT64) {
        std::vector<const double*> d_ptrs(num_tensors);
        std::vector<std::vector<int>> input_strides_list;
        std::vector<int> input_sizes(num_tensors);
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto tw = dynamic_cast<const TensorWrapper<double>*>(tensors[i].get());
            if (!tw) throw std::runtime_error("Wrapper cast failed");
            d_ptrs[i] = tw->tensor().data();
            input_strides_list.push_back(to_int_vec(tw->tensor().strides()));
            input_sizes[i] = static_cast<int>(tw->tensor().shape()[axis]);
        }
        
        // Flatten input strides for C function
        std::vector<int> flattened_strides;
        for (const auto& strides : input_strides_list) {
            flattened_strides.insert(flattened_strides.end(), strides.begin(), strides.end());
        }
        
        auto res_gpu = std::make_shared<gpuTensor<double>>(result_shape);
        auto result_strides = to_int_vec(res_gpu->strides());
        auto shape_vec = to_int_vec(res_dims);
        
        launch_concat_float64(res_gpu->data(), d_ptrs.data(), input_sizes.data(), num_tensors,
                             result_strides.data(), flattened_strides.data(),
                             shape_vec.data(), base_shape.ndims(), axis, total_elements);
        result_tensor = std::make_unique<TensorWrapper<double>>(res_gpu);
    } else {
        stop("concat currently supports float & double tensors only");
    }

    auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
    XPtr<TensorBase> ptr(uniq.release(), true);
    ptr.attr("class") = "gpuTensor";
    ptr.attr("dtype") = dtype_to_string(dtype);
    return ptr;
}

// ======================= STACK ======================= //
// [[Rcpp::export]]
SEXP tensor_stack_unified(List tensor_list, int axis = 1) {
    if (tensor_list.size() < 2) stop("Need at least two tensors to stack");
    axis--; // new dim index 0-based
    // Validate tensors
    std::vector<XPtr<TensorBase>> tensors;
    for (SEXP t : tensor_list) tensors.emplace_back(t);
    DType dtype = tensors[0]->dtype();
    Shape base_shape = tensors[0]->shape();
    for (auto& tp : tensors) {
        if (tp->dtype() != dtype) stop("All tensors must have same dtype");
        if (tp->shape() != base_shape) stop("All tensors must have identical shape for stack");
    }
    // Result shape: insert new dim of size num
    std::vector<size_t> res_dims = base_shape.dims;
    res_dims.insert(res_dims.begin() + axis, tensors.size());
    Shape result_shape(res_dims);

    size_t total_elements = result_shape.size();
    int num_tensors = tensors.size();

    if (dtype == DType::FLOAT32) {
        std::vector<const float*> d_ptrs(num_tensors);
        for (int i=0;i<num_tensors;++i) {
            auto tw = dynamic_cast<const TensorWrapper<float>*>(tensors[i].get());
            d_ptrs[i]=tw->tensor().data();
        }
        // Use the original input strides (length = base_ndims). The kernel receives
        // (ndims-1) entries which correspond exactly to the input tensor dimensions.
        auto one_stride = to_int_vec(dynamic_cast<const TensorWrapper<float>*>(tensors[0].get())->tensor().strides());
        auto res_gpu = std::make_shared<gpuTensor<float>>(result_shape);
        auto result_shape_vec = to_int_vec(res_dims);
        
        launch_stack_float32(res_gpu->data(), d_ptrs.data(), num_tensors,
                            one_stride.data(), result_shape_vec.data(), result_shape.ndims(), axis, total_elements);
        auto result_tensor = std::make_unique<TensorWrapper<float>>(res_gpu);
        XPtr<TensorBase> ptr(result_tensor.release(), true);
        ptr.attr("class")="gpuTensor"; ptr.attr("dtype")=dtype_to_string(dtype); return ptr;
    } else if (dtype==DType::FLOAT64) {
        std::vector<const double*> d_ptrs(num_tensors);
        for (int i=0;i<num_tensors;++i) {
            auto tw = dynamic_cast<const TensorWrapper<double>*>(tensors[i].get());
            d_ptrs[i]=tw->tensor().data();
        }
        auto one_stride = to_int_vec(dynamic_cast<const TensorWrapper<double>*>(tensors[0].get())->tensor().strides());
        auto res_gpu = std::make_shared<gpuTensor<double>>(result_shape);
        auto result_shape_vec = to_int_vec(res_dims);
        
        launch_stack_float64(res_gpu->data(), d_ptrs.data(), num_tensors,
                            one_stride.data(), result_shape_vec.data(), result_shape.ndims(), axis, total_elements);
        auto result_tensor = std::make_unique<TensorWrapper<double>>(res_gpu);
        XPtr<TensorBase> ptr(result_tensor.release(), true);
        ptr.attr("class")="gpuTensor"; ptr.attr("dtype")=dtype_to_string(dtype); return ptr;
    }
    stop("stack currently supports float/double only");
}

// ======================= REPEAT ======================= //
// [[Rcpp::export]]
SEXP tensor_repeat_unified(SEXP tensor_ptr, IntegerVector repeats) {
    XPtr<TensorBase> tensor(tensor_ptr);
    if (!tensor) stop("Invalid tensor pointer");

    int ndims = tensor->ndims();
    if (repeats.size() != ndims) stop("repeats length must match tensor dims");

    std::vector<int> repeat_counts(repeats.size());
    for (int i=0;i<repeats.size();++i) {
        if (repeats[i]<=0) stop("repeat counts must be positive");
        repeat_counts[i]=repeats[i];
    }

    // Result shape
    std::vector<size_t> res_dims;
    for (size_t d=0; d<tensor->shape().ndims(); ++d) {
        res_dims.push_back(tensor->shape()[d]*static_cast<size_t>(repeat_counts[d]));
    }
    Shape result_shape(res_dims);
    size_t total_elements = result_shape.size();

    DType dtype = tensor->dtype();
    if (dtype==DType::FLOAT32) {
        auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
        auto res_gpu = std::make_shared<gpuTensor<float>>(result_shape);
        auto input_strides_vec = to_int_vec(tw->tensor().strides());
        auto input_shape_vec = to_int_vec(tw->tensor().shape().dims);
        auto result_shape_vec = to_int_vec(res_dims);
        
        launch_repeat_float32(res_gpu->data(), tw->tensor().data(),
                             input_strides_vec.data(), repeat_counts.data(),
                             input_shape_vec.data(), result_shape_vec.data(), ndims, total_elements);
        auto result_tensor = std::make_unique<TensorWrapper<float>>(res_gpu);
        XPtr<TensorBase> ptr(result_tensor.release(), true);
        ptr.attr("class")="gpuTensor"; ptr.attr("dtype")="float"; return ptr;
    } else if (dtype==DType::FLOAT64) {
        auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
        auto res_gpu = std::make_shared<gpuTensor<double>>(result_shape);
        auto input_strides_vec = to_int_vec(tw->tensor().strides());
        auto input_shape_vec = to_int_vec(tw->tensor().shape().dims);
        auto result_shape_vec = to_int_vec(res_dims);
        
        launch_repeat_float64(res_gpu->data(), tw->tensor().data(),
                             input_strides_vec.data(), repeat_counts.data(),
                             input_shape_vec.data(), result_shape_vec.data(), ndims, total_elements);
        auto result_tensor = std::make_unique<TensorWrapper<double>>(res_gpu);
        XPtr<TensorBase> ptr(result_tensor.release(), true);
        ptr.attr("class")="gpuTensor"; ptr.attr("dtype")="double"; return ptr;
    }
    stop("repeat_tensor currently supports float/double only");
}

// ======================= PAD (constant only) ======================= //
// [[Rcpp::export]]
SEXP tensor_pad_unified(SEXP tensor_ptr, IntegerMatrix pad_width, std::string mode="constant", double value=0) {
    if (mode!="constant" && mode!="reflect" && mode!="replicate")
        stop("Unsupported pad mode");
    XPtr<TensorBase> tensor(tensor_ptr);
    if(!tensor) stop("Invalid tensor pointer");
    int ndims=tensor->ndims();
    if (pad_width.nrow()!=ndims || pad_width.ncol()!=2) stop("pad_width must be matrix of ndims x 2");
    std::vector<int> pad_before(ndims), pad_after(ndims);
    std::vector<size_t> res_dims= tensor->shape().dims;
    for(int i=0;i<ndims;++i){
        pad_before[i]=pad_width(i,0); pad_after[i]=pad_width(i,1);
        if (pad_before[i]<0 || pad_after[i]<0) stop("padding must be non-negative");
        res_dims[i]+= pad_before[i]+pad_after[i];
    }
    Shape result_shape(res_dims);
    size_t total_elements=result_shape.size();

    int pad_mode = (mode=="constant"?0: (mode=="reflect"?1:2));
    DType dtype=tensor->dtype();
    if (dtype==DType::FLOAT32){
        auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
        auto res_gpu=std::make_shared<gpuTensor<float>>(result_shape);
        auto input_strides_vec = to_int_vec(tw->tensor().strides());
        auto input_shape_vec = to_int_vec(tw->tensor().shape().dims);
        auto result_shape_vec = to_int_vec(res_dims);
        
        launch_pad_float32(res_gpu->data(), tw->tensor().data(),
                          input_strides_vec.data(), input_shape_vec.data(),
                          pad_before.data(), pad_after.data(), result_shape_vec.data(), ndims, static_cast<float>(value), pad_mode, total_elements);
        auto result_tensor=std::make_unique<TensorWrapper<float>>(res_gpu);
        XPtr<TensorBase> ptr(result_tensor.release(), true);
        ptr.attr("class")="gpuTensor"; ptr.attr("dtype")="float"; return ptr;
    } else if (dtype==DType::FLOAT64){
        auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
        auto res_gpu=std::make_shared<gpuTensor<double>>(result_shape);
        auto input_strides_vec = to_int_vec(tw->tensor().strides());
        auto input_shape_vec = to_int_vec(tw->tensor().shape().dims);
        auto result_shape_vec = to_int_vec(res_dims);
        
        launch_pad_float64(res_gpu->data(), tw->tensor().data(),
                          input_strides_vec.data(), input_shape_vec.data(),
                          pad_before.data(), pad_after.data(), result_shape_vec.data(), ndims, value, pad_mode, total_elements);
        auto result_tensor=std::make_unique<TensorWrapper<double>>(res_gpu);
        XPtr<TensorBase> ptr(result_tensor.release(), true);
        ptr.attr("class")="gpuTensor"; ptr.attr("dtype")="double"; return ptr;
    }
    stop("pad currently supports float/double only");
}

// ======================= GENERAL SLICE ASSIGNMENT ======================= //

// [[Rcpp::export]]
void tensor_slice_set_scalar_unified(SEXP tensor_ptr,
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
            tensor_slice_set_scalar_float32(
                tensor.data(), scalar,
                start.data(), slice.data(), strides_int.data(), ndims, total_elements);
        } else if (base_ptr->dtype() == DType::FLOAT64) {
            auto* typed = dynamic_cast<TensorWrapper<double>*>(base_ptr.get());
            if (!typed) stop("Type cast failed for float64 tensor");
            DoubleTensor& tensor = typed->tensor();
            tensor_slice_set_scalar_float64(
                tensor.data(), scalar,
                start.data(), slice.data(), strides_int.data(), ndims, total_elements);
        } else {
            stop("tensor_slice_set_scalar_unified currently supports float32 and float64 tensors only");
        }
    } catch (const std::exception& e) {
        stop(std::string("Error in tensor_slice_set_scalar_unified: ") + e.what());
    }
}

// [[Rcpp::export]]
void tensor_slice_set_tensor_unified(SEXP dest_tensor_ptr,
                                     IntegerVector start_indices,
                                     IntegerVector slice_shape,
                                     SEXP src_tensor_ptr) {
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
        XPtr<TensorBase> dest_ptr(dest_tensor_ptr);
        XPtr<TensorBase> src_ptr(src_tensor_ptr);
        if (!dest_ptr || !src_ptr) {
            stop("Invalid tensor pointer");
        }

        if (dest_ptr->dtype() != src_ptr->dtype()) {
            stop("Source and destination tensors must have the same dtype");
        }

        // Verify source tensor shape matches slice shape
        if (src_ptr->ndims() != ndims) {
            stop("Source tensor dimensions must match slice dimensions");
        }
        for (int i = 0; i < ndims; ++i) {
            if (src_ptr->shape()[i] != static_cast<size_t>(slice[i])) {
                stop("Source tensor shape must match slice shape");
            }
        }

        // Derive strides
        auto dest_strides_int = shape_to_strides_int(dest_ptr->shape());
        auto src_strides_int = shape_to_strides_int(src_ptr->shape());

        if (dest_ptr->dtype() == DType::FLOAT32) {
            auto* dest_typed = dynamic_cast<TensorWrapper<float>*>(dest_ptr.get());
            auto* src_typed = dynamic_cast<const TensorWrapper<float>*>(src_ptr.get());
            if (!dest_typed || !src_typed) stop("Type cast failed for float32 tensors");
            
            FloatTensor& dest_tensor = dest_typed->tensor();
            const FloatTensor& src_tensor = src_typed->tensor();
            
            tensor_slice_set_tensor_float32(
                dest_tensor.data(), src_tensor.data(),
                start.data(), slice.data(), 
                dest_strides_int.data(), src_strides_int.data(), 
                ndims, total_elements);
        } else if (dest_ptr->dtype() == DType::FLOAT64) {
            auto* dest_typed = dynamic_cast<TensorWrapper<double>*>(dest_ptr.get());
            auto* src_typed = dynamic_cast<const TensorWrapper<double>*>(src_ptr.get());
            if (!dest_typed || !src_typed) stop("Type cast failed for float64 tensors");
            
            DoubleTensor& dest_tensor = dest_typed->tensor();
            const DoubleTensor& src_tensor = src_typed->tensor();
            
            tensor_slice_set_tensor_float64(
                dest_tensor.data(), src_tensor.data(),
                start.data(), slice.data(), 
                dest_strides_int.data(), src_strides_int.data(), 
                ndims, total_elements);
        } else {
            stop("tensor_slice_set_tensor_unified currently supports float32 and float64 tensors only");
        }
    } catch (const std::exception& e) {
        stop(std::string("Error in tensor_slice_set_tensor_unified: ") + e.what());
    }
}

// [[Rcpp::export]]
void tensor_mask_set_scalar_unified(SEXP tensor_ptr, SEXP mask_ptr, double scalar) {
    try {
        XPtr<TensorBase> tensor_base(tensor_ptr);
        XPtr<TensorBase> mask_base(mask_ptr);
        if (!tensor_base || !mask_base) {
            stop("Invalid tensor pointer");
        }

        // Verify tensors have same shape
        if (tensor_base->shape() != mask_base->shape()) {
            stop("Tensor and mask must have the same shape");
        }

        // Verify mask is boolean type
        if (mask_base->dtype() != DType::BOOL) {
            stop("Mask tensor must be boolean type");
        }

        size_t total_elements = tensor_base->size();

        if (tensor_base->dtype() == DType::FLOAT32) {
            auto* tensor_typed = dynamic_cast<TensorWrapper<float>*>(tensor_base.get());
            auto* mask_typed = dynamic_cast<const TensorWrapper<bool>*>(mask_base.get());
            if (!tensor_typed || !mask_typed) stop("Type cast failed");
            
            FloatTensor& tensor = tensor_typed->tensor();
            const BoolTensor& mask = mask_typed->tensor();
            
            tensor_mask_set_scalar_float32(
                tensor.data(), mask.data(), static_cast<float>(scalar), total_elements);
        } else if (tensor_base->dtype() == DType::FLOAT64) {
            auto* tensor_typed = dynamic_cast<TensorWrapper<double>*>(tensor_base.get());
            auto* mask_typed = dynamic_cast<const TensorWrapper<bool>*>(mask_base.get());
            if (!tensor_typed || !mask_typed) stop("Type cast failed");
            
            DoubleTensor& tensor = tensor_typed->tensor();
            const BoolTensor& mask = mask_typed->tensor();
            
            tensor_mask_set_scalar_float64(
                tensor.data(), mask.data(), scalar, total_elements);
        } else {
            stop("tensor_mask_set_scalar_unified currently supports float32 and float64 tensors only");
        }
    } catch (const std::exception& e) {
        stop(std::string("Error in tensor_mask_set_scalar_unified: ") + e.what());
    }
} 