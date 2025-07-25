#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include <memory>

using namespace Rcpp;

extern "C" {
    void tensor_softmax_float32(float* output, const float* input, size_t n);
    void tensor_softmax_float64(double* output, const double* input, size_t n);
    int64_t tensor_argmax_float32(const float* input, size_t n);
    int64_t tensor_argmax_float64(const double* input, size_t n);
    
    // New activation functions
    void tensor_tanh_float32(float* result, const float* input, size_t n);
    void tensor_tanh_float64(double* result, const double* input, size_t n);
    void tensor_sigmoid_float32(float* result, const float* input, size_t n);
    void tensor_sigmoid_float64(double* result, const double* input, size_t n);
    void tensor_relu_float32(float* result, const float* input, size_t n);
    void tensor_relu_float64(double* result, const double* input, size_t n);
    void tensor_sin_float32(float* result, const float* input, size_t n);
    void tensor_sin_float64(double* result, const double* input, size_t n);
    void tensor_cos_float32(float* result, const float* input, size_t n);
    void tensor_cos_float64(double* result, const double* input, size_t n);
}

// [[Rcpp::export]]
SEXP tensor_softmax_unified(SEXP tensor_ptr) {
    XPtr<TensorBase> tensor(tensor_ptr);
    if (!tensor) stop("Invalid tensor pointer");
    DType dtype = tensor->dtype();
    std::unique_ptr<TensorBase> result_tensor;
    switch(dtype){
        case DType::FLOAT32: {
            auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
            auto contig = tw->tensor().is_contiguous()? tw->tensor(): tw->tensor().contiguous();
            auto res_gpu = std::make_shared<gpuTensor<float>>(contig.shape());
            tensor_softmax_float32(res_gpu->data(), contig.data(), contig.size());
            result_tensor = std::make_unique<TensorWrapper<float>>(res_gpu);
            break;}
        case DType::FLOAT64: {
            auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
            auto contig = tw->tensor().is_contiguous()? tw->tensor(): tw->tensor().contiguous();
            auto res_gpu = std::make_shared<gpuTensor<double>>(contig.shape());
            tensor_softmax_float64(res_gpu->data(), contig.data(), contig.size());
            result_tensor = std::make_unique<TensorWrapper<double>>(res_gpu);
            break;}
        default:
            stop("Softmax currently supports float/double tensors only");
    }
    auto uniq=std::unique_ptr<TensorBase>(result_tensor.release());
    XPtr<TensorBase> ptr(uniq.release(), true);
    ptr.attr("class")="gpuTensor";
    ptr.attr("dtype")=dtype_to_string(dtype);
    return ptr;
}

// Removed tensor_argmax_unified - now implemented in TensorReduction.cpp with full axis support

// Helper function for unary activation functions
SEXP create_unary_activation_unified(SEXP tensor_ptr, 
                                   void (*cuda_func_float32)(float*, const float*, size_t),
                                   void (*cuda_func_float64)(double*, const double*, size_t),
                                   const std::string& op_name) {
    XPtr<TensorBase> tensor(tensor_ptr);
    if (!tensor) stop("Invalid tensor pointer");
    
    DType dtype = tensor->dtype();
    std::unique_ptr<TensorBase> result_tensor;
    
    switch(dtype) {
        case DType::FLOAT32: {
            auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
            if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
            auto contig = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();
            auto res_gpu = std::make_shared<gpuTensor<float>>(contig.shape());
            cuda_func_float32(res_gpu->data(), contig.data(), contig.size());
            result_tensor = std::make_unique<TensorWrapper<float>>(res_gpu);
            break;
        }
        case DType::FLOAT64: {
            auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
            if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
            auto contig = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();
            auto res_gpu = std::make_shared<gpuTensor<double>>(contig.shape());
            cuda_func_float64(res_gpu->data(), contig.data(), contig.size());
            result_tensor = std::make_unique<TensorWrapper<double>>(res_gpu);
            break;
        }
        default:
            stop(op_name + " currently supports float/double tensors only");
    }
    
    auto uniq = std::unique_ptr<TensorBase>(result_tensor.release());
    XPtr<TensorBase> ptr(uniq.release(), true);
    ptr.attr("class") = "gpuTensor";
    ptr.attr("dtype") = dtype_to_string(dtype);
    return ptr;
}

// [[Rcpp::export]]
SEXP tensor_tanh_unified(SEXP tensor_ptr) {
    return create_unary_activation_unified(tensor_ptr, tensor_tanh_float32, tensor_tanh_float64, "Tanh");
}

// [[Rcpp::export]]
SEXP tensor_sigmoid_unified(SEXP tensor_ptr) {
    return create_unary_activation_unified(tensor_ptr, tensor_sigmoid_float32, tensor_sigmoid_float64, "Sigmoid");
}

// [[Rcpp::export]]
SEXP tensor_relu_unified(SEXP tensor_ptr) {
    return create_unary_activation_unified(tensor_ptr, tensor_relu_float32, tensor_relu_float64, "ReLU");
}

// [[Rcpp::export]]
SEXP tensor_sin_unified(SEXP tensor_ptr) {
    return create_unary_activation_unified(tensor_ptr, tensor_sin_float32, tensor_sin_float64, "Sin");
}

// [[Rcpp::export]]
SEXP tensor_cos_unified(SEXP tensor_ptr) {
    return create_unary_activation_unified(tensor_ptr, tensor_cos_float32, tensor_cos_float64, "Cos");
} 