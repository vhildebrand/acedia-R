#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// Forward declarations for math operations
extern "C" {
    // Unary math operations
    void tensor_exp_float32(float* result, const float* input, size_t n);
    void tensor_exp_float64(double* result, const double* input, size_t n);
    void tensor_log_float32(float* result, const float* input, size_t n);
    void tensor_log_float64(double* result, const double* input, size_t n);
    void tensor_sqrt_float32(float* result, const float* input, size_t n);
    void tensor_sqrt_float64(double* result, const double* input, size_t n);
    
    // Strided unary operations (for non-contiguous tensors)
    void tensor_exp_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                    const cuda_utils::TensorDescriptor& in_desc);
    void tensor_exp_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                    const cuda_utils::TensorDescriptor& in_desc);
    void tensor_log_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                    const cuda_utils::TensorDescriptor& in_desc);
    void tensor_log_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                    const cuda_utils::TensorDescriptor& in_desc);
    void tensor_sqrt_strided_float32(const cuda_utils::TensorDescriptor& out_desc, 
                                     const cuda_utils::TensorDescriptor& in_desc);
    void tensor_sqrt_strided_float64(const cuda_utils::TensorDescriptor& out_desc, 
                                     const cuda_utils::TensorDescriptor& in_desc);
}

// Declare the tensor_dtype_unified function (defined in TensorDataAccess.cpp)
std::string tensor_dtype_unified(SEXP tensor_ptr);

// [[Rcpp::export]]
SEXP tensor_exp_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                
                auto result_gpu = gpuTensor<float>(tw->tensor().shape(), tw->tensor().device());
                
                // Choose kernel based on contiguity
                if (tw->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_exp_float32(result_gpu.data(), tw->tensor().data(), tw->tensor().size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result_gpu.descriptor();
                    auto in_desc = tw->tensor().descriptor();
                    tensor_exp_strided_float32(out_desc, in_desc);
                }
                
                result_tensor = std::make_unique<TensorWrapper<float>>(
                    std::make_shared<gpuTensor<float>>(std::move(result_gpu))
                );
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                auto result_gpu = gpuTensor<double>(tw->tensor().shape(), tw->tensor().device());
                
                // Choose kernel based on contiguity
                if (tw->tensor().is_contiguous()) {
                    // Fast path: use flat kernel for contiguous tensors
                    tensor_exp_float64(result_gpu.data(), tw->tensor().data(), tw->tensor().size());
                } else {
                    // Strided path: use descriptor-based kernel for non-contiguous tensors
                    auto out_desc = result_gpu.descriptor();
                    auto in_desc = tw->tensor().descriptor();
                    tensor_exp_strided_float64(out_desc, in_desc);
                }
                
                result_tensor = std::make_unique<TensorWrapper<double>>(
                    std::make_shared<gpuTensor<double>>(std::move(result_gpu))
                );
                break;
            }
            default:
                stop("Exp not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor exp: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_log_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();

                // Input validation: log undefined for non-positive values
                std::vector<float> host_data(input_tensor.size());
                input_tensor.copy_to_host(host_data.data());
                for (float v : host_data) {
                    if (v <= 0.0f) {
                        stop("log() domain error: input contains non-positive values");
                    }
                }
                
                auto result_gpu = gpuTensor<float>(input_tensor.shape(), input_tensor.device());
                tensor_log_float32(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<float>>(
                    std::make_shared<gpuTensor<float>>(std::move(result_gpu))
                );
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();

                // Input validation: log undefined for non-positive values
                std::vector<double> host_data(input_tensor.size());
                input_tensor.copy_to_host(host_data.data());
                for (double v : host_data) {
                    if (v <= 0.0) {
                        stop("log() domain error: input contains non-positive values");
                    }
                }
                
                auto result_gpu = gpuTensor<double>(input_tensor.shape(), input_tensor.device());
                tensor_log_float64(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<double>>(
                    std::make_shared<gpuTensor<double>>(std::move(result_gpu))
                );
                break;
            }
            default:
                stop("Log not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor log: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP tensor_sqrt_unified(SEXP tensor_ptr) {
    try {
        XPtr<TensorBase> tensor(tensor_ptr);
        
        if (!tensor) {
            stop("Invalid tensor pointer");
        }
        
        DType dtype = tensor->dtype();
        std::unique_ptr<TensorBase> result_tensor;
        
        switch (dtype) {
            case DType::FLOAT32: {
                auto tw = dynamic_cast<const TensorWrapper<float>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT32");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();

                // Input validation: sqrt undefined for negative values
                std::vector<float> host_data(input_tensor.size());
                input_tensor.copy_to_host(host_data.data());
                for (float v : host_data) {
                    if (v < 0.0f) {
                        stop("sqrt() domain error: input contains negative values");
                    }
                }
                
                auto result_gpu = gpuTensor<float>(input_tensor.shape(), input_tensor.device());
                tensor_sqrt_float32(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<float>>(
                    std::make_shared<gpuTensor<float>>(std::move(result_gpu))
                );
                break;
            }
            case DType::FLOAT64: {
                auto tw = dynamic_cast<const TensorWrapper<double>*>(tensor.get());
                if (!tw) throw std::runtime_error("Invalid tensor wrapper for FLOAT64");
                
                // Make tensor contiguous if needed for proper memory layout
                auto input_tensor = tw->tensor().is_contiguous() ? tw->tensor() : tw->tensor().contiguous();

                // Input validation: sqrt undefined for negative values
                std::vector<double> host_data(input_tensor.size());
                input_tensor.copy_to_host(host_data.data());
                for (double v : host_data) {
                    if (v < 0.0) {
                        stop("sqrt() domain error: input contains negative values");
                    }
                }
                
                auto result_gpu = gpuTensor<double>(input_tensor.shape(), input_tensor.device());
                tensor_sqrt_float64(result_gpu.data(), input_tensor.data(), input_tensor.size());
                
                result_tensor = std::make_unique<TensorWrapper<double>>(
                    std::make_shared<gpuTensor<double>>(std::move(result_gpu))
                );
                break;
            }
            default:
                stop("Sqrt not yet implemented for dtype: " + dtype_to_string(dtype));
        }
        
        auto tensor_unique = std::unique_ptr<TensorBase>(result_tensor.release());
        XPtr<TensorBase> ptr(tensor_unique.release(), true);
        
        ptr.attr("class") = "gpuTensor";
        ptr.attr("dtype") = tensor_dtype_unified(tensor_ptr);
        
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in unified tensor sqrt: " + std::string(e.what()));
    }
} 