#' GPU-accelerated Element-wise Vector Multiplication
#'
#' Performs element-wise multiplication of two numeric vectors using CUDA on the GPU.
#' This function provides a GPU-accelerated alternative to the standard R `*` operator
#' for large vectors where the computational overhead of GPU memory transfer is 
#' justified by the parallel processing benefits.
#'
#' @param a A numeric vector
#' @param b A numeric vector of the same length as \code{a}
#' @param force_cpu Logical. If TRUE, forces CPU implementation (for testing/fallback)
#' @param warn_fallback Logical. If TRUE, warns when falling back to CPU implementation
#' 
#' @return A numeric vector containing the element-wise product of \code{a} and \code{b}
#' 
#' @details
#' This function transfers the input vectors to GPU memory, performs the multiplication
#' using a CUDA kernel with parallel threads, and transfers the result back to CPU.
#' For small vectors (< 10^4 elements), the CPU version may be faster due to 
#' memory transfer overhead.
#' 
#' If GPU is not available or GPU operations fail, the function automatically
#' falls back to CPU computation with an optional warning.
#' 
#' For chained GPU operations or advanced workflows, consider using \code{as.gpuVector()}
#' to create GPU-resident objects and use the \code{*} operator directly:
#' \code{gpu_a * gpu_b} where \code{gpu_a} and \code{gpu_b} are gpuVector objects.
#' 
#' @examples
#' \dontrun{
#' # Multiply two large vectors on GPU (with automatic fallback)
#' n <- 1e6
#' a <- runif(n)
#' b <- runif(n)
#' result <- gpu_multiply(a, b)
#' 
#' # Force CPU implementation for testing
#' result_cpu <- gpu_multiply(a, b, force_cpu = TRUE)
#' 
#' # For chained operations, use gpuVector objects:
#' gpu_a <- as.gpuVector(a)
#' gpu_b <- as.gpuVector(b) 
#' gpu_result <- gpu_a * gpu_b  # Stays on GPU
#' result2 <- as.vector(gpu_result)  # Transfer back when needed
#' 
#' # Verify correctness against CPU
#' all.equal(result, a * b)
#' all.equal(result2, a * b)
#' }
#' 
#' @export
gpu_multiply <- function(a, b, force_cpu = FALSE, warn_fallback = TRUE) {
  # Input validation
  if (!is.numeric(a) || !is.numeric(b)) {
    stop("Both arguments must be numeric vectors")
  }
  
  if (length(a) != length(b)) {
    stop("Input vectors must have the same length")
  }
  
  if (length(a) == 0) {
    return(numeric(0))
  }
  
  # Check if we should use GPU
  should_use_gpu <- !force_cpu && gpu_available()
  
  if (should_use_gpu) {
    # Try GPU implementation first
    tryCatch({
      # Create gpuVector objects and perform multiplication
      gpu_a <- as.gpuVector(a)
      gpu_b <- as.gpuVector(b)
      gpu_result <- gpu_multiply_vectors(gpu_a, gpu_b)
      as.vector(gpu_result)
    }, error = function(e) {
      # GPU failed, fall back to CPU
      if (warn_fallback) {
        warning("GPU operation failed, using CPU fallback: ", e$message)
      }
      a * b
    })
  } else {
    # Use CPU implementation
    if (!force_cpu && warn_fallback) {
      warning("GPU not available, using CPU implementation")
    }
    a * b
  }
}

#' GPU-accelerated Scalar Multiplication
#'
#' Performs scalar multiplication of a numeric vector using CUDA on the GPU.
#' This function provides a GPU-accelerated alternative for multiplying a vector
#' by a scalar value.
#'
#' @param x A numeric vector
#' @param scalar A numeric scalar value
#' @param force_cpu Logical. If TRUE, forces CPU implementation (for testing/fallback)
#' @param warn_fallback Logical. If TRUE, warns when falling back to CPU implementation
#' 
#' @return A numeric vector containing \code{x} multiplied by \code{scalar}
#' 
#' @details
#' This function transfers the input vector to GPU memory, performs the scalar
#' multiplication using a CUDA kernel with parallel threads, and transfers the 
#' result back to CPU. For small vectors, the CPU version may be faster due to 
#' memory transfer overhead.
#' 
#' If GPU is not available or GPU operations fail, the function automatically
#' falls back to CPU computation with an optional warning.
#' 
#' @examples
#' \dontrun{
#' # Scale a large vector on GPU
#' n <- 1e6
#' x <- runif(n)
#' scalar <- 3.14
#' result <- gpu_scale(x, scalar)
#' 
#' # Verify correctness against CPU
#' all.equal(result, x * scalar)
#' }
#' 
#' @export
gpu_scale <- function(x, scalar, force_cpu = FALSE, warn_fallback = TRUE) {
  # Input validation
  if (!is.numeric(x)) {
    stop("x must be a numeric vector")
  }
  
  if (!is.numeric(scalar) || length(scalar) != 1) {
    stop("scalar must be a single numeric value")
  }
  
  if (length(x) == 0) {
    return(numeric(0))
  }
  
  # Check if we should use GPU
  should_use_gpu <- !force_cpu && gpu_available()
  
  if (should_use_gpu) {
    # Try GPU implementation first
    tryCatch({
      # Create gpuVector object and perform scalar multiplication
      gpu_x <- as.gpuVector(x)
      gpu_result <- gpu_scale_vector(gpu_x, scalar)
      as.vector(gpu_result)
    }, error = function(e) {
      # GPU failed, fall back to CPU
      if (warn_fallback) {
        warning("GPU operation failed, using CPU fallback: ", e$message)
      }
      x * scalar
    })
  } else {
    # Use CPU implementation
    if (!force_cpu && warn_fallback) {
      warning("GPU not available, using CPU implementation")
    }
    x * scalar
  }
}

#' GPU-accelerated Dot Product
#'
#' Computes the dot product of two numeric vectors using CUDA on the GPU.
#' This function provides a GPU-accelerated alternative for computing the
#' inner product of two vectors.
#'
#' @param a A numeric vector
#' @param b A numeric vector of the same length as \code{a}
#' @param force_cpu Logical. If TRUE, forces CPU implementation (for testing/fallback)
#' @param warn_fallback Logical. If TRUE, warns when falling back to CPU implementation
#' 
#' @return A numeric scalar containing the dot product of \code{a} and \code{b}
#' 
#' @details
#' This function transfers the input vectors to GPU memory, performs the dot
#' product using a CUDA kernel with parallel reduction, and returns the scalar
#' result. The reduction operation uses shared memory for efficient computation.
#' 
#' If GPU is not available or GPU operations fail, the function automatically
#' falls back to CPU computation with an optional warning.
#' 
#' @examples
#' \dontrun{
#' # Compute dot product of large vectors on GPU
#' n <- 1e6
#' a <- runif(n)
#' b <- runif(n)
#' result <- gpu_dot(a, b)
#' 
#' # Verify correctness against CPU
#' all.equal(result, sum(a * b))
#' }
#' 
#' @export
gpu_dot <- function(a, b, force_cpu = FALSE, warn_fallback = TRUE) {
  # Input validation
  if (!is.numeric(a) || !is.numeric(b)) {
    stop("Both arguments must be numeric vectors")
  }
  
  if (length(a) != length(b)) {
    stop("Input vectors must have the same length")
  }
  
  if (length(a) == 0) {
    return(0.0)
  }
  
  # Check if we should use GPU
  should_use_gpu <- !force_cpu && gpu_available()
  
  if (should_use_gpu) {
    # Try GPU implementation first
    tryCatch({
      # Create gpuVector objects and perform dot product
      gpu_a <- as.gpuVector(a)
      gpu_b <- as.gpuVector(b)
      gpu_dot_vectors(gpu_a, gpu_b)
    }, error = function(e) {
      # GPU failed, fall back to CPU
      if (warn_fallback) {
        warning("GPU operation failed, using CPU fallback: ", e$message)
      }
      sum(a * b)
    })
  } else {
    # Use CPU implementation
    if (!force_cpu && warn_fallback) {
      warning("GPU not available, using CPU implementation")
    }
    sum(a * b)
  }
} 