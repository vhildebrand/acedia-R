#' Create a GPU Vector
#'
#' Creates a gpuVector object from an R numeric vector, transferring the data to GPU memory.
#'
#' @param x A numeric vector to transfer to GPU memory
#' 
#' @return A gpuVector object containing the data on GPU
#' 
#' @details
#' This function allocates GPU memory and copies the input vector to the GPU.
#' The resulting gpuVector object manages its own memory and will automatically
#' free GPU memory when it goes out of scope.
#' 
#' @examples
#' \dontrun{
#' # Create a gpuVector from R vector
#' x <- c(1, 2, 3, 4, 5)
#' gpu_x <- as.gpuVector(x)
#' 
#' # Convert back to R vector
#' y <- as.vector(gpu_x)
#' }
#' 
#' @export
as.gpuVector <- function(x) {
  if (!is.numeric(x)) {
    stop("Input must be a numeric vector")
  }
  
  # Use the Rcpp interface function
  as_gpuVector(x)
}

#' Convert GPU Vector to R Vector
#'
#' Converts a gpuVector object back to an R numeric vector by copying data from GPU to CPU.
#'
#' @param x A gpuVector object
#' @param mode The mode to convert to (ignored, always returns numeric)
#' 
#' @return A numeric vector containing the data from the GPU
#' 
#' @details
#' This function copies data from GPU memory back to CPU memory and returns
#' it as a standard R numeric vector.
#' 
#' @examples
#' \dontrun{
#' # Create gpuVector and convert back
#' x <- c(1, 2, 3, 4, 5)
#' gpu_x <- as.gpuVector(x)
#' y <- as.vector(gpu_x)
#' }
#' 
#' @export
as.vector.gpuVector <- function(x, mode = "any") {
  as_vector_gpuVector(x)
}

#' GPU-accelerated Vector Addition (gpuVector version)
#'
#' Performs element-wise addition of two gpuVector objects entirely on the GPU.
#'
#' @param a A gpuVector object
#' @param b A gpuVector object of the same length as \code{a}
#' 
#' @return A gpuVector object containing the element-wise sum
#' 
#' @details
#' This function performs vector addition entirely on the GPU without transferring
#' data back to CPU. This is more efficient than the original gpu_add function
#' when working with multiple GPU operations in sequence.
#' 
#' @examples
#' \dontrun{
#' # Create two gpuVectors and add them
#' n <- 1e6
#' a <- as.gpuVector(runif(n))
#' b <- as.gpuVector(runif(n))
#' result <- gpu_add_vectors(a, b)
#' 
#' # Convert result back to R if needed
#' r_result <- as.vector(result)
#' }
#' 
#' @export
gpu_add_vectors <- function(a, b) {
  # Input validation happens in C++ code
  gpu_add_rcpp(a, b)
}

#' Print method for gpuVector objects
#'
#' @param x A gpuVector object
#' @param ... Additional arguments (ignored)
#' @export
print.gpuVector <- function(x, ...) {
  print_gpuVector(x)
  invisible(x)
}

#' Size method for gpuVector objects
#'
#' @param x A gpuVector object
#' @return The number of elements in the gpuVector
#' @export
length.gpuVector <- function(x) {
  gpuVector_size(x)
}

#' Addition operator for gpuVector objects
#'
#' @param a A gpuVector object
#' @param b A gpuVector object
#' @return A gpuVector object containing the sum
#' @export
`+.gpuVector` <- function(a, b) {
  if (missing(b)) {
    # Unary plus - just return the vector
    return(a)
  }
  gpu_add_vectors(a, b)
} 