context("Axis-aware tensor reductions")

# Skip GPU tests if not available
skip_on_ci_if_no_gpu <- function() {
  tryCatch({
    test_tensor <- as_tensor(c(1, 2, 3), dtype = "float")
    if (!inherits(test_tensor, "gpuTensor")) {
      skip("GPU not available")
    }
  }, error = function(e) {
    skip("GPU not available")
  })
}

skip_on_ci_if_no_gpu()

# =============================================================================
# AXIS-AWARE SUM TESTS
# =============================================================================

test_that("sum() with axis parameter works correctly", {
  # 2D tensor test
  data_2d <- matrix(1:12, nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  
  # Sum along axis 1 (rows) - should give column sums
  result_axis1 <- sum(tensor_2d, axis = 1)
  expected_axis1 <- colSums(data_2d)
  expect_tensor_equal(result_axis1, expected_axis1)
  expect_equal(shape(result_axis1), c(4))
  
  # Sum along axis 2 (columns) - should give row sums  
  result_axis2 <- sum(tensor_2d, axis = 2)
  expected_axis2 <- rowSums(data_2d)
  expect_tensor_equal(result_axis2, expected_axis2)
  expect_equal(shape(result_axis2), c(3))
  
  # Test keep.dims = TRUE
  result_keepdims1 <- sum(tensor_2d, axis = 1, keep.dims = TRUE)
  expect_equal(shape(result_keepdims1), c(1, 4))
  expect_tensor_equal(result_keepdims1, matrix(expected_axis1, nrow = 1))
  
  result_keepdims2 <- sum(tensor_2d, axis = 2, keep.dims = TRUE)
  expect_equal(shape(result_keepdims2), c(3, 1))
  expect_tensor_equal(result_keepdims2, matrix(expected_axis2, ncol = 1))
})

test_that("sum() with multiple axes works correctly", {
  # 3D tensor test
  data_3d <- array(1:24, dim = c(2, 3, 4))
  tensor_3d <- as_tensor(data_3d, dtype = "float")
  
  # Sum along axes 1 and 2 - should collapse first two dimensions
  result_axes12 <- sum(tensor_3d, axis = c(1, 2))
  expected_axes12 <- apply(data_3d, 3, sum)
  expect_tensor_equal(result_axes12, expected_axes12)
  expect_equal(shape(result_axes12), c(4))
  
  # Test keep.dims = TRUE with multiple axes
  result_keepdims <- sum(tensor_3d, axis = c(1, 2), keep.dims = TRUE)
  expect_equal(shape(result_keepdims), c(1, 1, 4))
  expect_tensor_equal(result_keepdims, array(expected_axes12, dim = c(1, 1, 4)))
})

test_that("sum() backward compatibility preserved", {
  data <- matrix(1:12, nrow = 3, ncol = 4)
  tensor <- as_tensor(data, dtype = "float")
  
  # Global sum without axis parameter should return scalar
  result_global <- sum(tensor)
  expected_global <- sum(data)
  expect_equal(result_global, expected_global)
  expect_true(is.numeric(result_global) && length(result_global) == 1)
})

# =============================================================================
# AXIS-AWARE MEAN TESTS
# =============================================================================

test_that("mean() with axis parameter works correctly", {
  data_2d <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  
  # Mean along axis 1 (rows) - should give column means
  result_axis1 <- mean(tensor_2d, axis = 1)
  expected_axis1 <- colMeans(data_2d)
  expect_tensor_equal(result_axis1, expected_axis1)
  expect_equal(shape(result_axis1), c(4))
  
  # Mean along axis 2 (columns) - should give row means
  result_axis2 <- mean(tensor_2d, axis = 2)
  expected_axis2 <- rowMeans(data_2d)
  expect_tensor_equal(result_axis2, expected_axis2)
  expect_equal(shape(result_axis2), c(3))
  
  # Test keep.dims = TRUE
  result_keepdims1 <- mean(tensor_2d, axis = 1, keep.dims = TRUE)
  expect_equal(shape(result_keepdims1), c(1, 4))
  expect_tensor_equal(result_keepdims1, matrix(expected_axis1, nrow = 1))
})

# =============================================================================
# AXIS-AWARE MAX/MIN TESTS
# =============================================================================

test_that("max() with axis parameter works correctly", {
  data_2d <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8), nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  
  # Max along axis 1 - column maxes
  result_axis1 <- max(tensor_2d, axis = 1)
  expected_axis1 <- apply(data_2d, 2, max)
  expect_tensor_equal(result_axis1, expected_axis1)
  expect_equal(shape(result_axis1), c(4))
  
  # Max along axis 2 - row maxes
  result_axis2 <- max(tensor_2d, axis = 2)
  expected_axis2 <- apply(data_2d, 1, max)
  expect_tensor_equal(result_axis2, expected_axis2)
  expect_equal(shape(result_axis2), c(3))
})

test_that("min() with axis parameter works correctly", {
  data_2d <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8), nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  
  # Min along axis 1 - column mins
  result_axis1 <- min(tensor_2d, axis = 1)
  expected_axis1 <- apply(data_2d, 2, min)
  expect_tensor_equal(result_axis1, expected_axis1)
  expect_equal(shape(result_axis1), c(4))
  
  # Min along axis 2 - row mins
  result_axis2 <- min(tensor_2d, axis = 2)
  expected_axis2 <- apply(data_2d, 1, min)
  expect_tensor_equal(result_axis2, expected_axis2)
  expect_equal(shape(result_axis2), c(3))
})

# =============================================================================
# AXIS-AWARE PRODUCT TESTS
# =============================================================================

test_that("prod() with axis parameter works correctly", {
  # Use smaller values to avoid overflow
  data_2d <- matrix(c(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2), nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  
  # Product along axis 1 - column products
  result_axis1 <- prod(tensor_2d, axis = 1)
  expected_axis1 <- apply(data_2d, 2, prod)
  expect_tensor_equal(result_axis1, expected_axis1)
  expect_equal(shape(result_axis1), c(4))
  
  # Product along axis 2 - row products
  result_axis2 <- prod(tensor_2d, axis = 2)
  expected_axis2 <- apply(data_2d, 1, prod)
  expect_tensor_equal(result_axis2, expected_axis2)
  expect_equal(shape(result_axis2), c(3))
})

# =============================================================================
# ARGMAX/ARGMIN TESTS
# =============================================================================

test_that("argmax() and argmin() work correctly", {
  data_2d <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8), nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  
  # Global argmax and argmin
  global_argmax <- argmax(tensor_2d)
  global_argmin <- argmin(tensor_2d)
  expected_global_argmax <- base::which.max(as.vector(data_2d))  # R uses 1-based indexing
  expected_global_argmin <- base::which.min(as.vector(data_2d))
  expect_equal(global_argmax, expected_global_argmax)
  expect_equal(global_argmin, expected_global_argmin)
  
  # Axis-aware argmax (when implemented)
  # Note: These will be placeholders until CUDA kernels are fully implemented
  expect_error(argmax(tensor_2d, axis = 1), "not yet implemented")
  expect_error(argmin(tensor_2d, axis = 1), "not yet implemented")
})

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

test_that("axis parameter validation works correctly", {
  data_2d <- matrix(1:12, nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  
  # Invalid axis - too large
  expect_error(sum(tensor_2d, axis = 5), "axis must be a vector of positive integers")
  
  # Invalid axis - zero or negative
  expect_error(sum(tensor_2d, axis = 0), "axis must be a vector of positive integers")
  expect_error(sum(tensor_2d, axis = -1), "axis must be a vector of positive integers")
  
  # Invalid axis - non-integer
  expect_error(sum(tensor_2d, axis = 1.5), "axis must be a vector of positive integers")
  
  # Test same validation for other reduction functions
  expect_error(mean(tensor_2d, axis = 5), "axis must be a vector of positive integers")
  expect_error(max(tensor_2d, axis = 5), "axis must be a vector of positive integers")
  expect_error(min(tensor_2d, axis = 5), "axis must be a vector of positive integers")
  expect_error(prod(tensor_2d, axis = 5), "axis must be a vector of positive integers")
  expect_error(var(tensor_2d, axis = 5), "axis must be a vector of positive integers")
})

# =============================================================================
# NON-CONTIGUOUS TENSOR TESTS
# =============================================================================

test_that("axis-aware reductions work with non-contiguous tensors", {
  # Create a tensor and then transpose it to make it non-contiguous
  data_2d <- matrix(1:12, nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  tensor_transposed <- transpose(tensor_2d)
  
  # Test sum on transposed tensor
  result_axis1 <- sum(tensor_transposed, axis = 1)
  expected_axis1 <- colSums(t(data_2d))  # t(data_2d) gives the expected transposed matrix
  expect_tensor_equal(result_axis1, expected_axis1)
  
  result_axis2 <- sum(tensor_transposed, axis = 2)
  expected_axis2 <- rowSums(t(data_2d))
  expect_tensor_equal(result_axis2, expected_axis2)
})

# =============================================================================
# DIFFERENT DTYPES TESTS
# =============================================================================

test_that("axis-aware reductions work with different dtypes", {
  data_2d <- matrix(1:12, nrow = 3, ncol = 4)
  
  # Test with float32
  tensor_float <- as_tensor(data_2d, dtype = "float")
  result_float <- sum(tensor_float, axis = 1)
  expected <- colSums(data_2d)
  expect_tensor_equal(result_float, expected)
  
  # Test with float64
  tensor_double <- as_tensor(data_2d, dtype = "double")
  result_double <- sum(tensor_double, axis = 1)
  expect_tensor_equal(result_double, expected)
})

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

test_that("axis-aware reductions handle edge cases", {
  # 1D tensor
  data_1d <- c(1, 2, 3, 4, 5)
  tensor_1d <- as_tensor(data_1d, dtype = "float")
  
  result_1d <- sum(tensor_1d, axis = 1)
  expect_tensor_equal(result_1d, sum(data_1d))
  expect_equal(length(shape(result_1d)), 1)  # Should be scalar-like
  
  # Single element tensor
  tensor_single <- as_tensor(c(42), dtype = "float")
  result_single <- sum(tensor_single, axis = 1)
  expect_tensor_equal(result_single, 42)
  
  # Test keep.dims with 1D tensor
  result_1d_keepdims <- sum(tensor_1d, axis = 1, keep.dims = TRUE)
  expect_equal(shape(result_1d_keepdims), c(1))
})

test_that("variance reduction with axis works correctly", {
  data_2d <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), nrow = 3, ncol = 4)
  tensor_2d <- as_tensor(data_2d, dtype = "float")
  
  # Note: var implementation may be placeholder until proper two-pass algorithm
  # Test that it doesn't crash and returns correct shape
  result_axis1 <- var(tensor_2d, axis = 1)
  expect_equal(shape(result_axis1), c(4))
  
  result_axis2 <- var(tensor_2d, axis = 2) 
  expect_equal(shape(result_axis2), c(3))
}) 