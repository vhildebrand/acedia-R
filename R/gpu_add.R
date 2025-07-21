#' GPU-accelerated Vector Addition
#'
#' Performs element-wise addition of two numeric vectors using CUDA on the GPU.
#' This function provides a GPU-accelerated alternative to the standard R `+` operator
#' for large vectors where the computational overhead of GPU memory transfer is 
#' justified by the parallel processing benefits.
#'
#' @param a A numeric vector
#' @param b A numeric vector of the same length as \code{a}
#' @param use_gpuvector Logical. If TRUE, uses the new gpuVector implementation 
#'   which is more efficient for chained operations. Default FALSE for backward compatibility.
#' 
#' @return A numeric vector containing the element-wise sum of \code{a} and \code{b}
#' 
#' @details
#' This function transfers the input vectors to GPU memory, performs the addition
#' using a CUDA kernel with parallel threads, and transfers the result back to CPU.
#' For small vectors (< 10^4 elements), the CPU version may be faster due to 
#' memory transfer overhead.
#' 
#' When \code{use_gpuvector=TRUE}, the function uses the new gpuVector abstraction
#' which provides better memory management and is more efficient for sequences of
#' GPU operations.
#' 
#' @examples
#' \dontrun{
#' # Add two large vectors on GPU (original method)
#' n <- 1e6
#' a <- runif(n)
#' b <- runif(n)
#' result <- gpu_add(a, b)
#' 
#' # Using new gpuVector implementation
#' result2 <- gpu_add(a, b, use_gpuvector = TRUE)
#' 
#' # Verify correctness against CPU
#' all.equal(result, a + b)
#' all.equal(result2, a + b)
#' }
#' 
#' @export
gpu_add <- function(a, b, use_gpuvector = FALSE) {
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
  
  if (use_gpuvector) {
    # Use the new gpuVector implementation
    gpu_a <- as.gpuVector(a)
    gpu_b <- as.gpuVector(b)
    gpu_result <- gpu_add_vectors(gpu_a, gpu_b)
    return(as.vector(gpu_result))
  } else {
    # Use the original C interface function for backward compatibility
    .Call("r_gpu_add", a, b)
  }
} 