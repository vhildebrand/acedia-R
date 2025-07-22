#' Create a GPU Tensor
#'
#' Creates a gpuTensor object from R data, transferring it to GPU memory with specified shape.
#'
#' @param data A numeric vector or array containing the data
#' @param shape An integer vector specifying the tensor dimensions
#' @param dtype Data type: "double" or "float" (default: "double")
#' @param device Device location: "cuda" or "cpu" (default: "cuda")
#' @param requires_grad Whether to track gradients (default: FALSE)
#' 
#' @return A gpuTensor object
#' 
#' @details
#' This function creates a multi-dimensional GPU tensor with the specified shape.
#' The tensor supports views, broadcasting, async operations, and automatic differentiation
#' when requires_grad=TRUE.
#' 
#' @examples
#' \dontrun{
#' # Create a 2D tensor (matrix)
#' data <- 1:12
#' tensor <- gpu_tensor(data, shape = c(3, 4))
#' 
#' # Create a 3D tensor
#' tensor_3d <- gpu_tensor(runif(24), shape = c(2, 3, 4))
#' 
#' # Create with gradient tracking
#' tensor_grad <- gpu_tensor(1:6, shape = c(2, 3), requires_grad = TRUE)
#' }
#' 
#' @export
gpu_tensor <- function(data, shape, dtype = "double", device = "cuda", requires_grad = FALSE) {
  if (!is.numeric(data)) {
    stop("Data must be numeric")
  }
  
  if (!is.integer(shape)) {
    shape <- as.integer(shape)
  }
  
  if (any(shape <= 0)) {
    stop("All shape dimensions must be positive")
  }
  
  if (length(data) != prod(shape)) {
    stop("Data length (", length(data), ") doesn't match shape size (", prod(shape), ")")
  }
  
  if (device != "cuda") {
    stop("Only CUDA device is currently supported")
  }
  
  # Create tensor using unified interface
  tensor <- create_tensor_unified(as.numeric(data), shape, dtype)
  
  # Note: requires_grad functionality needs to be implemented for unified interface  
  if (requires_grad) {
    warning("requires_grad is not yet implemented for the unified interface")
  }
  
  class(tensor) <- c("gpuTensor", class(tensor))
  return(tensor)
}

#' Create Empty GPU Tensor
#'
#' Creates an uninitialized gpuTensor with the specified shape.
#'
#' @param shape An integer vector specifying the tensor dimensions
#' @param dtype Data type: "double" or "float" (default: "double")
#' @param device Device location: "cuda" or "cpu" (default: "cuda")
#' @param requires_grad Whether to track gradients (default: FALSE)
#' 
#' @return An empty gpuTensor object
#' 
#' @export
empty_tensor <- function(shape, dtype = "double", device = "cuda", requires_grad = FALSE) {
  if (!is.integer(shape)) {
    shape <- as.integer(shape)
  }
  
  if (any(shape <= 0)) {
    stop("All shape dimensions must be positive")
  }
  
  if (device != "cuda") {
    stop("Only CUDA device is currently supported")
  }
  
  # Create empty tensor using unified interface
  tensor <- create_empty_tensor_unified(shape, dtype)
  
  # Note: requires_grad functionality needs to be implemented for unified interface
  if (requires_grad) {
    warning("requires_grad is not yet implemented for the unified interface")
  }
  
  class(tensor) <- c("gpuTensor", class(tensor))
  return(tensor)
}

#' Create Tensor from Matrix/Array
#'
#' Creates a gpuTensor from an R matrix or array, preserving dimensions.
#'
#' @param x An R matrix or array
#' @param dtype Data type: "double" or "float" (default: "double")
#' @param requires_grad Whether to track gradients (default: FALSE)
#' 
#' @return A gpuTensor object with the same shape as the input
#' 
#' @export
as_tensor <- function(x, dtype = "double", requires_grad = FALSE) {
  if (!is.numeric(x)) {
    stop("Input must be numeric")
  }
  
  # Get dimensions (handle vectors, matrices, and arrays)
  shape <- dim(x)
  if (is.null(shape)) {
    shape <- length(x)  # Vector case
  }
  
  # Convert to vector for data transfer
  data <- as.vector(x)
  
  return(gpu_tensor(data, shape, dtype, "cuda", requires_grad))
}

#' Convert GPU Tensor to R Array
#'
#' Transfers data from GPU tensor back to R, preserving shape.
#'
#' @param tensor A gpuTensor object
#' 
#' @return An R array with the same shape as the tensor
#' 
#' @export
as.array.gpuTensor <- function(tensor) {
  return(tensor_to_r_unified(tensor))
}

#' @export
as.vector.gpuTensor <- function(x, mode = "any") {
  result <- as.array(x)
  return(as.vector(result))
}

#' Get Tensor Shape
#'
#' @param tensor A gpuTensor object
#' @return Integer vector containing the tensor dimensions
#' @export
shape <- function(tensor) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  return(tensor_shape(tensor))
}

#' Get Tensor Size
#'
#' @param tensor A gpuTensor object
#' @return Total number of elements in the tensor
#' @export
size <- function(tensor) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  return(tensor_size(tensor))
}

#' Print Tensor Information
#'
#' @param x A gpuTensor object
#' @param ... Additional arguments (ignored)
#' @export
print.gpuTensor <- function(x, ...) {
  cat("gpuTensor with shape", paste(shape(x), collapse=" x "), "\n")
  cat("Device: CUDA\n")
  cat("DType:", attr(x, "dtype", exact=TRUE) %||% "unknown", "\n")
  cat("Size:", size(x), "elements\n")
  
  # Show a preview of the data if it's not too large
  if (size(x) <= 100) {
    cat("Data:\n")
    data <- as.array(x)
    print(data)
  } else {
    cat("Use as.array() to view data\n")
  }
  
  invisible(x)
}

#' @export
`+.gpuTensor` <- function(a, b) {
  if (missing(b)) {
    # Unary plus
    return(a)
  }
  
  # Check types
  dtype_a <- attr(a, "dtype", exact=TRUE) %||% "double"
  
  if (inherits(b, "gpuTensor")) {
    dtype_b <- attr(b, "dtype", exact=TRUE) %||% "double"
    if (dtype_a != dtype_b) {
      stop("Cannot add tensors with different dtypes")
    }
    
    # Use unified interface
    result <- tensor_add_unified(a, b)
  } else if (is.numeric(b) && length(b) == 1) {
    # Scalar addition (implemented as a + b = a + (b * ones_like(a)))
    if (dtype_a == "double") {
      # For now, we'll use scalar multiplication followed by addition with original tensor
      # This is inefficient but works as a placeholder
      ones <- empty_tensor(shape(a), dtype_a)
      # TODO: Implement fill operation or ones_like function
      stop("Scalar addition not yet fully implemented")
    }
  } else {
    stop("Cannot add gpuTensor with object of type: ", class(b))
  }
  
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' @export
`*.gpuTensor` <- function(a, b) {
  if (missing(b)) {
    stop("Multiplication requires two operands")
  }
  
  dtype_a <- attr(a, "dtype", exact=TRUE) %||% "double"
  
  if (inherits(b, "gpuTensor")) {
    # Element-wise multiplication
    dtype_b <- attr(b, "dtype", exact=TRUE) %||% "double"
    if (dtype_a != dtype_b) {
      stop("Cannot multiply tensors with different dtypes")
    }
    
    # TODO: Implement tensor_mul_unified for element-wise multiplication
    stop("Element-wise multiplication not yet implemented in unified interface")
  } else if (is.numeric(b) && length(b) == 1) {
    # Scalar multiplication using unified interface
    result <- tensor_scalar_mul_unified(a, b)
  } else {
    stop("Cannot multiply gpuTensor with object of type: ", class(b))
  }
  
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Matrix Multiplication
#'
#' Performs matrix multiplication between two 2D tensors.
#'
#' @param a First tensor (2D)
#' @param b Second tensor (2D)
#' @return Result of matrix multiplication
#' @export
matmul <- function(a, b) {
  if (!inherits(a, "gpuTensor") || !inherits(b, "gpuTensor")) {
    stop("Both arguments must be gpuTensors")
  }
  
  dtype_a <- attr(a, "dtype", exact=TRUE) %||% "double"
  dtype_b <- attr(b, "dtype", exact=TRUE) %||% "double"
  
  if (dtype_a != dtype_b) {
    stop("Cannot multiply tensors with different dtypes")
  }
  
  if (dtype_a == "double") {
    result <- tensor_matmul_double(a, b)
  } else {
    stop("Matrix multiplication not implemented for dtype: ", dtype_a)
  }
  
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Tensor Sum
#'
#' Computes the sum of all elements in a tensor.
#'
#' @param tensor A gpuTensor object
#' @return Scalar sum of all elements
#' @export
sum.gpuTensor <- function(tensor, ...) {
  return(tensor_sum_unified(tensor))
}

#' Create Tensor View
#'
#' Creates a view of a tensor with a different shape (same data).
#'
#' @param tensor A gpuTensor object
#' @param new_shape Integer vector specifying the new shape
#' @return A new tensor that shares memory with the original
#' @export
view <- function(tensor, new_shape) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  dtype <- attr(tensor, "dtype", exact=TRUE) %||% "double"
  
  if (dtype == "double") {
    result <- tensor_view_double(tensor, as.integer(new_shape))
  } else {
    stop("View not implemented for dtype: ", dtype)
  }
  
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Check if Tensor is Contiguous
#'
#' @param tensor A gpuTensor object
#' @return TRUE if tensor has contiguous memory layout
#' @export
is_contiguous <- function(tensor) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  return(tensor_is_contiguous(tensor))
}

#' Synchronize Tensor Operations
#'
#' Waits for all asynchronous operations on the tensor to complete.
#'
#' @param tensor A gpuTensor object
#' @export
synchronize <- function(tensor) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  tensor_synchronize(tensor)
  invisible(tensor)
}

#' Enable Gradient Computation
#'
#' Sets whether gradients should be computed for this tensor.
#'
#' @param tensor A gpuTensor object
#' @param requires_grad Boolean indicating whether to track gradients
#' @return The tensor (for method chaining)
#' @export
requires_grad <- function(tensor, requires_grad = TRUE) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  dtype <- attr(tensor, "dtype", exact=TRUE) %||% "double"
  
  if (dtype == "double") {
    tensor_requires_grad_double(tensor, requires_grad)
  } else {
    stop("Autograd not implemented for dtype: ", dtype)
  }
  
  return(tensor)
}

# Utility function for null coalescing
`%||%` <- function(lhs, rhs) {
  if (!is.null(lhs)) lhs else rhs
} 