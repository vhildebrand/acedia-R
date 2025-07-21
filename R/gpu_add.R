#' GPU-accelerated Vector Addition
#'
#' Performs element-wise addition of two numeric vectors using CUDA on the GPU.
#' This function provides a GPU-accelerated alternative to the standard R `+` operator
#' for large vectors where the computational overhead of GPU memory transfer is 
#' justified by the parallel processing benefits.
#'
#' @param a A numeric vector
#' @param b A numeric vector of the same length as \code{a}
#' @param force_cpu Logical. If TRUE, forces CPU implementation (for testing/fallback)
#' @param warn_fallback Logical. If TRUE, warns when falling back to CPU implementation
#' 
#' @return A numeric vector containing the element-wise sum of \code{a} and \code{b}
#' 
#' @details
#' This function transfers the input vectors to GPU memory, performs the addition
#' using a CUDA kernel with parallel threads, and transfers the result back to CPU.
#' For small vectors (< 10^4 elements), the CPU version may be faster due to 
#' memory transfer overhead.
#' 
#' If GPU is not available or GPU operations fail, the function automatically
#' falls back to CPU computation with an optional warning.
#' 
#' For chained GPU operations or advanced workflows, consider using \code{as.gpuVector()}
#' to create GPU-resident objects and use the \code{+} operator directly:
#' \code{gpu_a + gpu_b} where \code{gpu_a} and \code{gpu_b} are gpuVector objects.
#' 
#' @examples
#' \dontrun{
#' # Add two large vectors on GPU (with automatic fallback)
#' n <- 1e6
#' a <- runif(n)
#' b <- runif(n)
#' result <- gpu_add(a, b)
#' 
#' # Force CPU implementation for testing
#' result_cpu <- gpu_add(a, b, force_cpu = TRUE)
#' 
#' # For chained operations, use gpuVector objects:
#' gpu_a <- as.gpuVector(a)
#' gpu_b <- as.gpuVector(b) 
#' gpu_result <- gpu_a + gpu_b  # Stays on GPU
#' result2 <- as.vector(gpu_result)  # Transfer back when needed
#' 
#' # Verify correctness against CPU
#' all.equal(result, a + b)
#' all.equal(result2, a + b)
#' }
#' 
#' @export
gpu_add <- function(a, b, force_cpu = FALSE, warn_fallback = TRUE) {
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
      .Call("r_gpu_add", a, b)
    }, error = function(e) {
      # GPU failed, fall back to CPU
      if (warn_fallback) {
        warning("GPU operation failed, using CPU fallback: ", e$message)
      }
      a + b
    })
  } else {
    # Use CPU implementation
    if (!force_cpu && warn_fallback) {
      warning("GPU not available, using CPU implementation")
    }
    a + b
  }
} 