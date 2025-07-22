#' acediaR: R Interface for CUDA-Accelerated Linear Algebra
#'
#' This package provides CUDA-accelerated linear algebra operations for R,
#' enabling high-performance GPU computation with familiar R syntax.
#'
#' The package offers:
#'   \item High-level functions like \code{gpu_add()}, \code{gpu_multiply()} for simple operations
#'   \item Advanced gpuTensor objects for complex multi-dimensional operations with mixed precision support
#'   \item Automatic CPU fallback when GPU is not available
#'   \item Memory-efficient tensor operations that minimize data transfers
#'
#' @docType package
#' @name acediaR
#' @useDynLib acediaR, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

# Package initialization
.onLoad <- function(libname, pkgname) {
  # Any package initialization code can go here
} 