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

// [[Rcpp::export]]
int64_t tensor_argmax_unified(SEXP tensor_ptr) {
    XPtr<TensorBase> tensor(tensor_ptr);
    if(!tensor) stop("Invalid tensor pointer");
    if (tensor->size()==0) stop("Empty tensor");
    DType dtype=tensor->dtype();
    switch(dtype){
        case DType::FLOAT32: {
            auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
            return tensor_argmax_float32(tw->tensor().data(), tensor->size()); }
        case DType::FLOAT64: {
            auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
            return tensor_argmax_float64(tw->tensor().data(), tensor->size()); }
        default:
            stop("Argmax currently supports float/double tensors only");
    }
    return -1;
} 