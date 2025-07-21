// R_interface.cpp

#include <R.h>
#include <Rinternals.h>

// forward declartion of CUDA host function from vector_add.cu

extern "C" void addVectorsonGpu(double *h_c, const double *h_a, const double *h_b, int n);

extern "C" SEXP r_gpu_add(SEXP a, SEXP b) {
    // check that inputs are valid numeric vectors
    if (!isReal(a) || !isReal(b)) {
        error("Inputs must be numeric vectors");
    }

    // get length of vectors
    int n = length(a);
    if (length(b) != n) {
        error("Input vectors must have the same length");
    }

    // get C pointer to underlying data of R vectors
    const double *h_a = REAL(a);
    const double *h_b = REAL(b);

    // allocate on R vector to store the result
    SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
    double *h_c = REAL(result_sexp);

    // call CUDA host wrapper function
    addVectorsonGpu(h_c, h_a, h_b, n);

    UNPROTECT(1);
    return result_sexp;
}