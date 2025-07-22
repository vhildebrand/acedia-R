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
  
  # Validate dtype (currently only double/float supported by high-level R interface)
  if (!dtype %in% c("double", "float")) {
    stop("Unsupported dtype. Supported dtypes are 'double' and 'float'")
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
  # Validate dtype for empty tensor as well
  if (!dtype %in% c("double", "float")) {
    stop("Unsupported dtype. Supported dtypes are 'double' and 'float'")
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
    stop("Object is not a gpuTensor")
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
  
  # For now, implement a basic version via R array manipulation
  # TODO: Implement with CUDA kernels for efficiency
  warning("concat currently uses R array manipulation. CUDA kernel implementation needed for optimal performance.")
  
  # Convert all tensors to arrays
  arrays <- lapply(tensors, as.array)
  
  # Use base R's abind or implement our own concatenation
  result_array <- do.call(abind::abind, c(arrays, along = axis))
  
  # Convert back to gpu_tensor
  result <- gpu_tensor(as.vector(result_array), dim(result_array))
  return(result)
}

#' Stack Tensors
#'
#' Stack tensors along a new dimension.
#'
#' @param tensors List of gpuTensor objects to stack
#' @param axis Integer specifying the axis along which to stack (1-indexed)
#' @return A new tensor with tensors stacked along the specified axis
#' @export
stack <- function(tensors, axis = 1) {
  if (!is.list(tensors) || length(tensors) < 2) {
    stop("tensors must be a list of at least 2 gpuTensor objects")
  }
  
  if (!all(sapply(tensors, function(x) inherits(x, "gpuTensor")))) {
    stop("All elements in tensors must be gpuTensor objects")
  }
  
  # Check that all tensors have the same shape
  first_shape <- shape(tensors[[1]])
  if (!all(sapply(tensors, function(x) identical(shape(x), first_shape)))) {
    stop("All tensors must have the same shape for stacking")
  }
  
  # For now, implement via R array manipulation
  warning("stack currently uses R array manipulation. CUDA kernel implementation needed for optimal performance.")
  
  arrays <- lapply(tensors, as.array)
  result_array <- do.call(abind::abind, c(arrays, along = axis))
  
  result <- gpu_tensor(as.vector(result_array), dim(result_array))
  return(result)
}

#' Repeat Tensor
#'
#' Repeat tensor along specified dimensions.
#'
#' @param tensor A gpuTensor object
#' @param repeats Integer vector specifying repeat counts for each dimension
#' @return A new tensor with the input tensor repeated
#' @export
repeat_tensor <- function(tensor, repeats) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("tensor must be a gpuTensor object")
  }
  
  if (length(repeats) != length(shape(tensor))) {
    stop("Length of repeats must match number of tensor dimensions")
  }
  
  if (any(repeats <= 0)) {
    stop("All repeat counts must be positive")
  }
  
  # For now, implement via R array manipulation
  warning("repeat_tensor currently uses R array manipulation. CUDA kernel implementation needed for optimal performance.")
  
  array_data <- as.array(tensor)
  
  # Use base R's rep function with array indexing
  for (dim in seq_along(repeats)) {
    if (repeats[dim] > 1) {
      indices <- rep(seq_len(dim(array_data)[dim]), repeats[dim])
      array_data <- apply(array_data, setdiff(seq_len(length(dim(array_data))), dim), 
                         function(x) x[indices])
    }
  }
  
  result <- gpu_tensor(as.vector(array_data), dim(array_data))
  return(result)
}

#' Pad Tensor
#'
#' Pad tensor with specified values.
#'
#' @param tensor A gpuTensor object
#' @param pad_width List or matrix specifying padding for each dimension
#' @param mode Padding mode: "constant", "reflect", or "replicate"
#' @param value Constant value for constant padding (default: 0)
#' @return A new tensor with padding applied
#' @export
pad <- function(tensor, pad_width, mode = "constant", value = 0) {
  if (!inherits(tensor, "gpuTensor")) {
    stop("tensor must be a gpuTensor object")
  }
  
  if (!mode %in% c("constant", "reflect", "replicate")) {
    stop("mode must be one of: 'constant', 'reflect', 'replicate'")
  }
  
  # For now, implement basic constant padding via R array manipulation
  warning("pad currently uses R array manipulation. CUDA kernel implementation needed for optimal performance.")
  
  array_data <- as.array(tensor)
  
  # Convert pad_width to standard format
  if (is.list(pad_width)) {
    if (length(pad_width) != length(shape(tensor))) {
      stop("pad_width list length must match number of dimensions")
    }
    pad_matrix <- do.call(rbind, pad_width)
  } else if (is.matrix(pad_width)) {
    if (nrow(pad_width) != length(shape(tensor)) || ncol(pad_width) != 2) {
      stop("pad_width matrix must be ndims x 2")
    }
    pad_matrix <- pad_width
  } else {
    stop("pad_width must be a list or matrix")
  }
  
  # Apply padding using base R functionality
  # This is a simplified implementation - full CUDA implementation would be much faster
  if (mode == "constant") {
    # Use basic array padding with constant values
    padded_dims <- dim(array_data) + pad_matrix[, 1] + pad_matrix[, 2]
    result_array <- array(value, dim = padded_dims)
    
    # Copy original data to center of padded array
    indices <- lapply(seq_len(length(dim(array_data))), function(i) {
      (pad_matrix[i, 1] + 1):(pad_matrix[i, 1] + dim(array_data)[i])
    })
    
    do.call(`[<-`, c(list(result_array), indices, list(array_data)))
    result <- gpu_tensor(as.vector(result_array), dim(result_array))
    return(result)
  }
  
  # For reflect and replicate modes, would need more complex implementation
  stop("Only constant padding mode is currently implemented")
} 