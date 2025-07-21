#include <Rcpp.h>
#include "gpuVector.h"

using namespace Rcpp;

// Forward declaration of CUDA function for vector addition
extern "C" void addVectorsOnGpu_gpuVector(gpuVector& result, const gpuVector& a, const gpuVector& b);

// [[Rcpp::export]]
SEXP as_gpuVector(NumericVector x) {
    try {
        gpuVector* gpu_vec;
        if (x.size() == 0) {
            gpu_vec = new gpuVector(0);
        } else {
            gpu_vec = new gpuVector(x.begin(), x.size());
        }
        
        // Create external pointer
        XPtr<gpuVector> ptr(gpu_vec, true); // true means R will delete the object
        ptr.attr("class") = "gpuVector";
        return ptr;
    } catch (const std::exception& e) {
        stop("Error creating gpuVector: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
NumericVector as_vector_gpuVector(SEXP gpu_vec_ptr) {
    try {
        XPtr<gpuVector> ptr(gpu_vec_ptr);
        if (!ptr) {
            stop("Invalid gpuVector pointer");
        }
        
        if (ptr->empty()) {
            return NumericVector();
        }
        
        NumericVector result(ptr->size());
        ptr->copyToHost(result.begin());
        return result;
    } catch (const std::exception& e) {
        stop("Error converting gpuVector to R vector: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
SEXP gpu_add_rcpp(SEXP a_ptr, SEXP b_ptr) {
    try {
        XPtr<gpuVector> a_vec(a_ptr);
        XPtr<gpuVector> b_vec(b_ptr);
        
        if (!a_vec || !b_vec) {
            stop("Invalid gpuVector pointer(s)");
        }
        
        if (a_vec->size() != b_vec->size()) {
            stop("gpuVector sizes must match for addition");
        }
        
        gpuVector* result = new gpuVector(a_vec->size());
        
        // Call the CUDA function
        addVectorsOnGpu_gpuVector(*result, *a_vec, *b_vec);
        
        // Create external pointer for result
        XPtr<gpuVector> ptr(result, true);
        ptr.attr("class") = "gpuVector";
        return ptr;
    } catch (const std::exception& e) {
        stop("Error in GPU vector addition: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
void print_gpuVector(SEXP gpu_vec_ptr) {
    try {
        XPtr<gpuVector> ptr(gpu_vec_ptr);
        
        if (!ptr) {
            Rcout << "Invalid gpuVector pointer" << std::endl;
            return;
        }
        
        Rcout << "gpuVector of length " << ptr->size();
        
        if (ptr->empty()) {
            Rcout << " (empty)" << std::endl;
            return;
        }
        
        // Show first few elements for preview
        const size_t preview_size = std::min(static_cast<size_t>(6), ptr->size());
        NumericVector preview_data(preview_size);
        
        // Copy a small portion to CPU for display
        if (preview_size > 0) {
            gpuVector temp_vec(preview_size);
            cudaMemcpy(temp_vec.data(), ptr->data(), 
                       preview_size * sizeof(double), cudaMemcpyDeviceToDevice);
            temp_vec.copyToHost(preview_data.begin());
            
            Rcout << ": ";
            for (size_t i = 0; i < preview_size; ++i) {
                Rcout << preview_data[i];
                if (i < preview_size - 1) Rcout << ", ";
            }
            
            if (ptr->size() > preview_size) {
                Rcout << ", ...";
            }
        }
        
        Rcout << std::endl;
    } catch (const std::exception& e) {
        Rcout << "Error printing gpuVector: " << e.what() << std::endl;
    }
}

// [[Rcpp::export]]
size_t gpuVector_size(SEXP gpu_vec_ptr) {
    try {
        XPtr<gpuVector> ptr(gpu_vec_ptr);
        if (!ptr) {
            stop("Invalid gpuVector pointer");
        }
        return ptr->size();
    } catch (const std::exception& e) {
        stop("Error getting gpuVector size: " + std::string(e.what()));
    }
}

// [[Rcpp::export]]
bool gpuVector_empty(SEXP gpu_vec_ptr) {
    try {
        XPtr<gpuVector> ptr(gpu_vec_ptr);
        if (!ptr) {
            stop("Invalid gpuVector pointer");
        }
        return ptr->empty();
    } catch (const std::exception& e) {
        stop("Error checking if gpuVector is empty: " + std::string(e.what()));
    }
} 