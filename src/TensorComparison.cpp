#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// Forward declarations of CUDA wrappers (defined in tensor_ops.cu)
extern "C" {
    void tensor_gt_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_gt_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_lt_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_lt_float64(double* result, const double* a, const double* b, size_t n);
    void tensor_eq_float32(float* result, const float* a, const float* b, size_t n);
    void tensor_eq_float64(double* result, const double* a, const double* b, size_t n);
}

// [[Rcpp::export]]
SEXP tensor_gt_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        if (!a_tensor || !b_tensor) stop("Invalid tensor pointer(s)");
        if (a_tensor->dtype() != b_tensor->dtype()) {
            stop("Comparison requires tensors with identical dtypes");
        }
        if (a_tensor->shape() != b_tensor->shape()) {
            stop("Comparison currently requires tensors with identical shapes");
        }
        DType dtype = a_tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        switch (dtype) {
            case DType::FLOAT32: {
                auto a_wrap = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<float>>(a_contig.shape());
                tensor_gt_float32(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result_gpu);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrap = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<double>>(a_contig.shape());
                tensor_gt_float64(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result_gpu);
                break;
            }
            default:
                stop("Comparison not yet implemented for dtype");
        }
        auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(uniq.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop(std::string("Error in comparison function: ") + e.what());
    }
}

// [[Rcpp::export]]
SEXP tensor_lt_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        if (!a_tensor || !b_tensor) stop("Invalid tensor pointer(s)");
        if (a_tensor->dtype() != b_tensor->dtype()) {
            stop("Comparison requires tensors with identical dtypes");
        }
        if (a_tensor->shape() != b_tensor->shape()) {
            stop("Comparison currently requires tensors with identical shapes");
        }
        DType dtype = a_tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        switch (dtype) {
            case DType::FLOAT32: {
                auto a_wrap = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<float>>(a_contig.shape());
                tensor_lt_float32(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result_gpu);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrap = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<double>>(a_contig.shape());
                tensor_lt_float64(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result_gpu);
                break;
            }
            default:
                stop("Comparison not yet implemented for dtype");
        }
        auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(uniq.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop(std::string("Error in comparison function: ") + e.what());
    }
}

// [[Rcpp::export]]
SEXP tensor_eq_unified(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<TensorBase> a_tensor(a_ptr);
        XPtr<TensorBase> b_tensor(b_ptr);
        if (!a_tensor || !b_tensor) stop("Invalid tensor pointer(s)");
        if (a_tensor->dtype() != b_tensor->dtype()) {
            stop("Comparison requires tensors with identical dtypes");
        }
        if (a_tensor->shape() != b_tensor->shape()) {
            stop("Comparison currently requires tensors with identical shapes");
        }
        DType dtype = a_tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        switch (dtype) {
            case DType::FLOAT32: {
                auto a_wrap = dynamic_cast<const TensorWrapper<float>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<float>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<float>>(a_contig.shape());
                tensor_eq_float32(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<float>>(result_gpu);
                break;
            }
            case DType::FLOAT64: {
                auto a_wrap = dynamic_cast<const TensorWrapper<double>*>(a_tensor.get());
                auto b_wrap = dynamic_cast<const TensorWrapper<double>*>(b_tensor.get());
                if (!a_wrap || !b_wrap) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                auto a_contig = a_wrap->tensor().is_contiguous()? a_wrap->tensor() : a_wrap->tensor().contiguous();
                auto b_contig = b_wrap->tensor().is_contiguous()? b_wrap->tensor() : b_wrap->tensor().contiguous();
                auto result_gpu = std::make_shared<gpuTensor<double>>(a_contig.shape());
                tensor_eq_float64(result_gpu->data(), a_contig.data(), b_contig.data(), result_gpu->size());
                result_tensor = std::make_unique<TensorWrapper<double>>(result_gpu);
                break;
            }
            default:
                stop("Comparison not yet implemented for dtype");
        }
        auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(uniq.release(), true);
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = dtype_to_string(dtype);
        return ptr;
    } catch (const std::exception& e) {
        stop(std::string("Error in comparison function: ") + e.what());
    }
} 