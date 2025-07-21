#' @details
#' The acediaR package provides GPU-accelerated computing capabilities for R using CUDA.
#' It includes high-performance implementations of common mathematical operations
#' including vector arithmetic, matrix operations, and statistical computations.
#' 
#' The package provides two main interfaces:
#' \itemize{
#'   \item Low-level functions that transfer data to GPU, perform operations, and return to CPU
#'   \item High-level gpuVector objects that maintain data on GPU for efficient chained operations
#' }
#'
#' @keywords internal
"_PACKAGE"

#' @useDynLib acediaR, .registration=TRUE
#' @importFrom Rcpp evalCpp sourceCpp
#' @import methods
#' @exportPattern "^[[:alpha:]]+"
NULL

# Package initialization
.onLoad <- function(libname, pkgname) {
  # Any package initialization code can go here
} 