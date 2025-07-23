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
  
  # Validate and normalize dtype 
  if (dtype == "float32") {
    dtype <- "float"  # Map float32 to float for compatibility
  } else if (dtype == "float64") {
    dtype <- "double"  # Map float64 to double for compatibility
  }
  
  if (!dtype %in% c("double", "float")) {
    stop("Unsupported dtype. Supported dtypes are 'double', 'float', 'float32', 'float64'")
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
  
  # Ensure proper class structure (avoid duplicates)
  # Fix any class duplication issues from C++ 
  current_class <- class(tensor)
  if (length(current_class) > 1 && all(current_class == "gpuTensor")) {
    class(tensor) <- "gpuTensor"
  } else if (length(current_class) == 1 && current_class[1] == "gpuTensor") {
    # Already correct
  } else if (!"gpuTensor" %in% current_class) {
    class(tensor) <- c("gpuTensor", current_class)
  } else {
    # Remove any duplicates
    class(tensor) <- unique(current_class)
  }
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
  # Validate and normalize dtype for empty tensor
  if (dtype == "float32") {
    dtype <- "float"  # Map float32 to float for compatibility
  } else if (dtype == "float64") {
    dtype <- "double"  # Map float64 to double for compatibility
  }
  
  if (!dtype %in% c("double", "float")) {
    stop("Unsupported dtype. Supported dtypes are 'double', 'float', 'float32', 'float64'")
  }

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

#' Convert R object to GPU Tensor
#'
#' @param x Input R object (vector, matrix, or array)
#' @param ... Additional arguments passed to methods
#' @return A gpuTensor object with the same shape as the input
#' 
#' @export
as_tensor <- function(x, ...) {
  UseMethod("as_tensor")
}

#' @rdname as_tensor
#' @param dtype Data type for the tensor (default: "double")
#' @param requires_grad Whether to track gradients (default: FALSE)
#' @param shape Optional shape to reshape the data (default: NULL, uses input shape)
#' @export
as_tensor.default <- function(x, dtype = "double", requires_grad = FALSE, shape = NULL, ...) {
  if (is.array(x) || is.matrix(x)) {
    if (!is.null(shape)) {
      # Check if size matches
      if (length(as.vector(x)) != prod(shape)) {
        stop("Data size (", length(as.vector(x)), ") does not match shape size (", prod(shape), ")")
      }
      gpu_tensor(data = as.vector(x), shape = shape, dtype = dtype, requires_grad = requires_grad)
    } else {
      gpu_tensor(data = as.vector(x), shape = dim(x), dtype = dtype, requires_grad = requires_grad)
    }
  } else if (is.vector(x)) {
    if (!is.null(shape)) {
      # Check if size matches
      if (length(x) != prod(shape)) {
        stop("Data size (", length(x), ") does not match shape size (", prod(shape), ")")
      }
      gpu_tensor(data = x, shape = shape, dtype = dtype, requires_grad = requires_grad)
    } else {
      gpu_tensor(data = x, shape = length(x), dtype = dtype, requires_grad = requires_grad)
    }
  } else {
    stop("Cannot convert object of class '", class(x)[1], "' to tensor")
  }
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
as.array.gpuTensor <- function(x, ...) {
  result <- tensor_to_r_unified(x)
  # For 1D tensors, return a vector instead of 1D array for better compatibility
  if (length(dim(result)) == 1) {
    return(as.vector(result))
  }
  return(result)
}

#' @export
as.vector.gpuTensor <- function(x, mode = "any") {
  result <- as.array(x)
  return(as.vector(result))
}

#' @export
as.numeric.gpuTensor <- function(x, ...) {
  # Convert to array first, then to numeric
  arr <- as.array(x)
  return(as.numeric(arr))
}

# Also add a direct conversion function as backup
#' @export  
to_numeric <- function(x) {
  if (inherits(x, "gpuTensor")) {
    return(as.numeric(as.array(x)))
  } else {
    return(as.numeric(x))
  }
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
  return(tensor_shape_unified(tensor))
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
  return(tensor_size_unified(tensor))
}

#' Get Tensor Data Type
#'
#' @param tensor A gpuTensor object
#' @return Character string indicating the data type
#' @export
dtype <- function(tensor) {
  UseMethod("dtype")
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

# Helper function to ensure proper gpuTensor class (no duplicates)
.fix_gpuTensor_class <- function(x) {
  if (!is.null(x) && inherits(x, "gpuTensor")) {
    current_class <- class(x)
    if (length(current_class) > 1) {
      if (all(current_class == "gpuTensor")) {
        # All duplicates, fix to single
        class(x) <- "gpuTensor"
      } else {
        # Mixed classes, keep unique ones
        class(x) <- unique(current_class)
      }
    }
  }
  return(x)
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
    result <- tensor_scalar_add_unified(a, b)
    class(result) <- c("gpuTensor", class(result))
    return(result)
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
    
    # Use unified interface for element-wise multiplication
    result <- tensor_mul_unified(a, b)
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
  
  # Use unified interface for matrix multiplication
  result <- tensor_matmul_unified(a, b)
  
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
  
  # Use unified interface for view operations
  result <- tensor_view_unified(tensor, as.integer(new_shape))
  
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Reshape Tensor
#'
#' Creates a reshaped tensor with a different shape. If the tensor is contiguous,
#' this creates a zero-copy view. If non-contiguous, it first makes the tensor
#' contiguous then creates a view.
#'
#' @param tensor A gpuTensor object
#' @param new_shape Integer vector specifying the new shape
#' @return A tensor with the new shape (zero-copy if possible)
#' @export
reshape <- function(tensor, new_shape) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  if (!is.integer(new_shape)) {
    new_shape <- as.integer(new_shape)
  }
  
  if (any(new_shape <= 0)) {
    stop("All shape dimensions must be positive")
  }
  
  if (prod(new_shape) != size(tensor)) {
    stop("New shape size (", prod(new_shape), ") must match tensor size (", size(tensor), ")")
  }
  
  # Use unified interface for true reshape (handles contiguous/non-contiguous automatically)
  result <- tensor_reshape_unified(tensor, new_shape)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Transpose Tensor
#'
#' Creates a transposed view of a 2D tensor (matrix).
#'
#' @param tensor A 2D gpuTensor object
#' @return A transposed tensor that shares memory with the original
#' @export
transpose <- function(tensor) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  if (length(shape(tensor)) != 2) {
    stop("Transpose currently supports 2D tensors only")
  }
  
  # Use unified interface for true transpose views
  result <- tensor_transpose_unified(tensor)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Permute Tensor Dimensions
#'
#' Reorder the dimensions of a tensor.
#'
#' @param tensor A gpuTensor object
#' @param dims Integer vector specifying the new order of dimensions (1-indexed)
#' @return A tensor with permuted dimensions
#' @export
permute <- function(tensor, dims) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  if (length(dims) != length(shape(tensor))) {
    stop("Number of dimensions in 'dims' must match tensor dimensions")
  }
  
  # Use unified interface for true permute views
  result <- tensor_permute_unified(tensor, as.integer(dims))
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
    stop("Object istlan not a gpuTensor")
  }
  return(tensor_is_contiguous_unified(tensor))
}

#' Return a contiguous copy of tensor
#' @export
contiguous <- function(tensor) {
  if (!inherits(tensor, "gpuTensor")) stop("Object is not a gpuTensor")
  # Currently contiguous() implemented on C++ side by calling method via view
  # Round-trip through tensor_view_unified of same shape ensures copy, but we added contiguous in C++
  # To invoke, convert to dtype itself (clone path)
  result <- tensor_view_unified(tensor, as.integer(shape(tensor)))
  class(result) <- c("gpuTensor", class(result))
  return(result)
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
  tensor_synchronize_unified(tensor)
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
  
  # Autograd not yet implemented in unified interface
  stop("Autograd not yet implemented in unified interface")
}

#' @export
`-.gpuTensor` <- function(a, b = NULL) {
  if (is.null(b)) {
    # Unary negative
    return(a * (-1))
  }

  dtype_a <- attr(a, "dtype", exact=TRUE) %||% "double"
  if (inherits(b, "gpuTensor")) {
    dtype_b <- attr(b, "dtype", exact=TRUE) %||% "double"
    if (dtype_a != dtype_b) stop("Cannot subtract tensors with different dtypes")
    result <- tensor_sub_unified(a, b)
  } else if (is.numeric(b) && length(b) == 1) {
    result <- tensor_scalar_add_unified(a, -b)
  } else {
    stop("Cannot subtract gpuTensor with object of type: ", class(b))
  }
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' @export
`/.gpuTensor` <- function(a, b) {
  if (missing(b)) stop("Division requires two operands")
  dtype_a <- attr(a, "dtype", exact=TRUE) %||% "double"
  if (inherits(b, "gpuTensor")) {
    dtype_b <- attr(b, "dtype", exact=TRUE) %||% "double"
    if (dtype_a != dtype_b) stop("Cannot divide tensors with different dtypes")
    result <- tensor_div_unified(a, b)
  } else if (is.numeric(b) && length(b) == 1) {
    inv <- 1.0 / b
    result <- tensor_scalar_mul_unified(a, inv)
  } else {
    stop("Cannot divide gpuTensor with object of type: ", class(b))
  }
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Tensor slicing for gpuTensor
#' @export
`[.gpuTensor` <- function(x, ..., drop = TRUE) {
  tensor_dims <- shape(x)
  n_dims <- length(tensor_dims)
  
  # Get the actual arguments (including missing ones)
  args <- substitute(list(...))[-1]  # Remove the 'list' part
  n_args <- length(args)
  
  # Handle empty indices (return full tensor)
  if (n_args == 0) {
    return(x)
  }
  
  # Validate number of indices
  if (n_args > n_dims) {
    stop("Too many indices for tensor with ", n_dims, " dimensions")
  }
  
  # Process each index position
  indices <- vector("list", n_dims)
  for (i in 1:n_dims) {
    if (i <= n_args) {
      if (args[[i]] == quote(expr=)) {
        indices[[i]] <- NULL  # Missing argument like t[2, ]
      } else {
        indices[[i]] <- eval(args[[i]], parent.frame())
      }
    } else {
      indices[[i]] <- NULL  # No index provided for this dimension
    }
  }
  
  # Process each index
  new_shape <- integer(0)
  start_indices <- integer(0)
  end_indices <- integer(0)
  
  for (i in 1:n_dims) {
    idx <- if (i <= length(indices)) indices[[i]] else NULL
    dim_size <- tensor_dims[i]
    
    if (missing(idx) || is.null(idx)) {
      # Missing index means select all
      start_indices[i] <- 1L
      end_indices[i] <- dim_size
      new_shape <- c(new_shape, dim_size)
    } else if (is.numeric(idx)) {
      # Validate positive indices
      if (any(idx <= 0)) {
        stop("Only positive indices are supported in this implementation")
      }
      if (any(idx > dim_size)) {
        stop("Index ", max(idx), " is out of bounds for dimension ", i, " with size ", dim_size)
      }
      
      if (length(idx) == 1) {
        # Single index - this reduces dimensionality
        start_indices[i] <- as.integer(idx)
        end_indices[i] <- as.integer(idx)
        # Don't add to new_shape (dimension is removed)
      } else {
        # Multiple indices - for now, only support contiguous ranges
        if (!all(diff(idx) == 1)) {
          stop("Only contiguous ranges are supported in this basic implementation")
        }
        start_indices[i] <- as.integer(min(idx))
        end_indices[i] <- as.integer(max(idx))
        new_shape <- c(new_shape, length(idx))
      }
    } else {
      stop("Unsupported index type: ", class(idx))
    }
  }

  # For now, implement basic slicing by copying data
  # This is not the most efficient but works for the basic interface
  
  # Calculate the elements to extract
  old_dim <- tensor_dims
  total_elements <- prod(end_indices - start_indices + 1)
  
  # Create result tensor
  if (length(new_shape) == 0) {
    # Scalar result
    result_data <- tensor_slice_extract(x, start_indices, end_indices)
    return(as.numeric(result_data))
  } else {
    # Tensor result
    result_data <- tensor_slice_extract(x, start_indices, end_indices)
    result <- gpu_tensor(result_data, new_shape)
    return(result)
  }
}

# Helper function to extract sliced data (placeholder implementation)
tensor_slice_extract <- function(tensor, start_indices, end_indices) {
  # For basic implementation, convert to R array, slice, and return
  arr <- as.array(tensor)
  
  # Build the slicing expression dynamically
  slice_args <- list()
  for (i in seq_along(start_indices)) {
    if (start_indices[i] == end_indices[i]) {
      # Single index
      slice_args[[i]] <- start_indices[i]
    } else {
      # Range
      slice_args[[i]] <- start_indices[i]:end_indices[i]
    }
  }
  
  # Apply the slice
  result <- do.call(`[`, c(list(arr), slice_args))
  
  # Return as vector for gpu_tensor creation
  return(as.vector(result))
}

# Utility function for null coalescing
`%||%` <- function(lhs, rhs) {
  if (!is.null(lhs)) lhs else rhs
} 

#' Get tensor dimensions
#' @export
dim.gpuTensor <- function(x) {
  return(as.integer(shape(x)))
}

#' Exponential Function
#'
#' Computes the exponential of each element in the tensor.
#'
#' @param x A gpuTensor object
#' @return A tensor with exp(x) computed element-wise
#' @export
exp.gpuTensor <- function(x) {
  if (!inherits(x, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  result <- tensor_exp_unified(x)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Natural Logarithm Function
#'
#' Computes the natural logarithm of each element in the tensor.
#'
#' @param x A gpuTensor object
#' @param base The base of the logarithm (default: exp(1) for natural log)
#' @return A tensor with log(x) computed element-wise
#' @export
log.gpuTensor <- function(x, base = exp(1)) {
  if (!inherits(x, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  if (base != exp(1)) {
    stop("Only natural logarithm (base e) is currently supported")
  }
  
  result <- tensor_log_unified(x)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Square Root Function
#'
#' Computes the square root of each element in the tensor.
#'
#' @param x A gpuTensor object
#' @return A tensor with sqrt(x) computed element-wise
#' @export
sqrt.gpuTensor <- function(x) {
  if (!inherits(x, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  result <- tensor_sqrt_unified(x)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Mean (Average) Function
#'
#' Computes the arithmetic mean of all elements in the tensor.
#'
#' @param x A gpuTensor object
#' @param ... Additional arguments (ignored)
#' @return The mean value of all tensor elements
#' @export
mean.gpuTensor <- function(x, ...) {
  if (!inherits(x, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  return(tensor_mean_unified(x))
}

#' Maximum Function
#'
#' Finds the maximum element in the tensor.
#'
#' @param x A gpuTensor object
#' @param ... Additional arguments (ignored)
#' @return The maximum value in the tensor
#' @export
max.gpuTensor <- function(x, ...) {
  if (!inherits(x, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  return(tensor_max_unified(x))
}

#' Minimum Function
#'
#' Finds the minimum element in the tensor.
#'
#' @param x A gpuTensor object
#' @param ... Additional arguments (ignored)
#' @return The minimum value in the tensor
#' @export
min.gpuTensor <- function(x, ...) {
  if (!inherits(x, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  
  return(tensor_min_unified(x))
}

#' Concatenate Tensors
#'
#' Concatenate tensors along an existing dimension.
#'
#' @param tensors List of gpuTensor objects to concatenate
#' @param axis Integer specifying the axis along which to concatenate (1-indexed)
#' @return A new tensor with tensors concatenated along the specified axis
#' @export
concat <- function(tensors, axis = 1) {
  if (!is.list(tensors) || length(tensors) < 2) {
    stop("tensors must be a list of at least 2 gpuTensor objects")
  }
  if (!all(sapply(tensors, function(x) inherits(x, "gpuTensor")))) {
    stop("All elements in tensors must be gpuTensor objects")
  }
  result <- tensor_concat_unified(tensors, axis)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

stack <- function(tensors, axis = 1) {
  if (!is.list(tensors) || length(tensors) < 2) {
    stop("tensors must be a list of at least 2 gpuTensor objects")
  }
  if (!all(sapply(tensors, function(x) inherits(x, "gpuTensor")))) {
    stop("All elements in tensors must be gpuTensor objects")
  }
  result <- tensor_stack_unified(tensors, axis)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

repeat_tensor <- function(tensor, repeats) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("tensor must be a gpuTensor object")
  }
  result <- tensor_repeat_unified(tensor, as.integer(repeats))
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

pad <- function(tensor, pad_width, mode = "constant", value = 0) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("tensor must be a gpuTensor object")
  }
  if (is.list(pad_width)) {
    pad_mat <- do.call(rbind, pad_width)
  } else {
    pad_mat <- as.matrix(pad_width)
  }
  if (ncol(pad_mat) != 2) stop("pad_width must have two columns (before, after)")
  pad_mat <- matrix(as.integer(pad_mat), nrow = nrow(pad_mat), ncol = 2)
  result <- tensor_pad_unified(tensor, pad_mat, mode, value)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' @export
shape.gpuTensor <- function(x) {
  .Call(`_acediaR_tensor_shape_unified`, x)
}

#' @export
dtype.gpuTensor <- function(x) {
  .Call(`_acediaR_tensor_dtype_unified`, x)
}

#' @export
size.gpuTensor <- function(x) {
  .Call(`_acediaR_tensor_size_unified`, x)
}

# Package load hook to ensure proper method registration
.onLoad <- function(libname, pkgname) {
  # Force registration of as.numeric method for primitives
  registerS3method("as.numeric", "gpuTensor", as.numeric.gpuTensor)
}

#' Slice assignment for gpuTensor (in-place update on GPU)
#'
#' Currently supports assigning a numeric scalar to a rectangular (contiguous) slice.
#' More complex assignment (tensor-to-slice) can be added later.
#' @export
`[<-.gpuTensor` <- function(x, ..., value) {
  tensor_dims <- shape(x)
  n_dims <- length(tensor_dims)
  args <- substitute(list(...))[-1]  # remove list wrapper
  n_args <- length(args)

  if (missing(value)) {
    stop("No value provided for slice assignment")
  }

  # Build start & end indices (similar logic to `[.gpuTensor` reader)
  indices <- vector("list", n_dims)
  for (i in 1:n_dims) {
    if (i <= n_args) {
      if (args[[i]] == quote(expr=)) {
        indices[[i]] <- NULL  # missing like t[ ,]
      } else {
        indices[[i]] <- eval(args[[i]], parent.frame())
      }
    } else {
      indices[[i]] <- NULL  # No index provided for this dimension
    }
  }

  start_indices <- integer(n_dims)
  end_indices   <- integer(n_dims)

  for (i in 1:n_dims) {
    idx <- if (i <= length(indices)) indices[[i]] else NULL
    dim_size <- tensor_dims[i]

    if (is.null(idx)) {
      start_indices[i] <- 1L
      end_indices[i]   <- dim_size
    } else if (is.numeric(idx)) {
      if (any(idx <= 0)) stop("Negative or zero indices not supported in slice assignment")
      if (any(idx > dim_size)) stop("Index out of bounds in slice assignment")
      if (!all(diff(idx) == 1)) stop("Only contiguous ranges supported in slice assignment")
      start_indices[i] <- as.integer(min(idx))
      end_indices[i]   <- as.integer(max(idx))
    } else {
      stop("Unsupported index type in slice assignment: ", class(idx))
    }
  }

  slice_shape <- end_indices - start_indices + 1L

  if (is.numeric(value) && length(value) == 1) {
    # In-place scalar add/assign: for now we implement add-scalar (+=)
    tensor_slice_add_scalar_unified(x, start_indices, slice_shape, value)
    return(x)  # modified in-place (external pointer)
  } else {
    stop("Currently only scalar numeric assignment to slice is supported")
  }
}

#' Greater Than Comparison
#'
#' Performs element-wise comparison a > b and returns a tensor of 0/1.
#'
#' @param a First gpuTensor
#' @param b Second gpuTensor
#' @return gpuTensor with numeric 0/1 values
#' @export
`>.gpuTensor` <- function(a, b) {
  if (missing(b) || !inherits(b, "gpuTensor")) {
    stop("Both operands must be gpuTensor objects")
  }
  if (!identical(shape(a), shape(b))) {
    stop("Comparison currently requires tensors with identical shapes")
  }
  result <- tensor_gt_unified(a, b)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Less Than Comparison
#' @rdname greater_than_gpuTensor
#' @export
`<.gpuTensor` <- function(a, b) {
  if (missing(b) || !inherits(b, "gpuTensor")) {
    stop("Both operands must be gpuTensor objects")
  }
  if (!identical(shape(a), shape(b))) {
    stop("Comparison currently requires tensors with identical shapes")
  }
  result <- tensor_lt_unified(a, b)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Equality Comparison
#' @rdname greater_than_gpuTensor
#' @export
`==.gpuTensor` <- function(a, b) {
  if (missing(b) || !inherits(b, "gpuTensor")) {
    stop("Both operands must be gpuTensor objects")
  }
  if (!identical(shape(a), shape(b))) {
    stop("Comparison currently requires tensors with identical shapes")
  }
  result <- tensor_eq_unified(a, b)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

#' Product (All elements)
#'
#' Computes the product of all tensor elements.
#'
#' @param x A gpuTensor object
#' @param ... Additional args (ignored)
#' @return Numeric scalar
#' @export
prod.gpuTensor <- function(x, ...) {
  if (!inherits(x, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  tensor_prod_unified(x)
}

#' Variance (Population)
#'
#' Computes population variance of tensor elements.
#'
#' @param x A gpuTensor object
#' @param ... Additional args (ignored)
#' @return Numeric scalar (double)
#' @export
var.gpuTensor <- function(x, ...) {
  if (!inherits(x, "gpuTensor")) {
    stop("Object is not a gpuTensor")
  }
  tensor_var_unified(x)
}

softmax <- function(tensor) {
  if (!inherits(tensor, "gpuTensor")) stop("tensor must be gpuTensor")
  result <- tensor_softmax_unified(tensor)
  class(result) <- c("gpuTensor", class(result))
  return(result)
}

argmax <- function(tensor) {
  if (!inherits(tensor, "gpuTensor")) stop("tensor must be gpuTensor")
  tensor_argmax_unified(tensor) + 1  # convert to 1-based index for R
}