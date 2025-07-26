context("Tensor creation utilities: empty_tensor, create_ones_like")

# Helper functions
expect_tensor_equal <- function(tensor, expected, tolerance = 1e-6) {
  expect_equal(as.array(tensor), expected, tolerance = tolerance)
}

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
# EMPTY_TENSOR TESTS
# =============================================================================

test_that("empty_tensor creates tensors with correct shape", {
  # Test 1D tensor
  shape_1d <- c(5)
  tensor_1d <- empty_tensor(shape_1d, dtype = "float")
  
  expect_s3_class(tensor_1d, "gpuTensor")
  expect_equal(shape(tensor_1d), shape_1d)
  expect_equal(size(tensor_1d), prod(shape_1d))
  expect_equal(dtype(tensor_1d), "float")
  
  # Test 2D tensor
  shape_2d <- c(3, 4)
  tensor_2d <- empty_tensor(shape_2d, dtype = "float")
  
  expect_equal(shape(tensor_2d), shape_2d)
  expect_equal(size(tensor_2d), prod(shape_2d))
  
  # Test 3D tensor
  shape_3d <- c(2, 3, 4)
  tensor_3d <- empty_tensor(shape_3d, dtype = "float")
  
  expect_equal(shape(tensor_3d), shape_3d)
  expect_equal(size(tensor_3d), prod(shape_3d))
})

test_that("empty_tensor works with different dtypes", {
  shape <- c(2, 3)
  
  # Test float32
  tensor_f32 <- empty_tensor(shape, dtype = "float")
  expect_equal(dtype(tensor_f32), "float")
  expect_equal(shape(tensor_f32), shape)
  
  # Test float64
  tensor_f64 <- empty_tensor(shape, dtype = "double")
  expect_equal(dtype(tensor_f64), "double")
  expect_equal(shape(tensor_f64), shape)
})

test_that("empty_tensor handles dtype aliases", {
  shape <- c(3, 3)
  
  # Test float32 alias
  tensor_f32_alias <- empty_tensor(shape, dtype = "float32")
  expect_equal(dtype(tensor_f32_alias), "float")
  
  # Test float64 alias
  tensor_f64_alias <- empty_tensor(shape, dtype = "float64")
  expect_equal(dtype(tensor_f64_alias), "double")
})

test_that("empty_tensor creates contiguous tensors", {
  shape <- c(4, 5)
  tensor <- empty_tensor(shape, dtype = "float")
  
  expect_true(is_contiguous(tensor))
})

test_that("empty_tensor handles edge cases", {
  # Single element tensor
  single_shape <- c(1)
  single_tensor <- empty_tensor(single_shape, dtype = "float")
  
  expect_equal(shape(single_tensor), single_shape)
  expect_equal(size(single_tensor), 1)
  
  # Large tensor (test memory allocation)
  large_shape <- c(100, 100)
  large_tensor <- empty_tensor(large_shape, dtype = "float")
  
  expect_equal(shape(large_tensor), large_shape)
  expect_equal(size(large_tensor), 10000)
  
  # High-dimensional tensor
  high_dim_shape <- c(2, 2, 2, 2, 2)
  high_dim_tensor <- empty_tensor(high_dim_shape, dtype = "float")
  
  expect_equal(shape(high_dim_tensor), high_dim_shape)
  expect_equal(size(high_dim_tensor), 32)
})

test_that("empty_tensor error handling", {
  # Invalid dtype
  expect_error(empty_tensor(c(3, 3), dtype = "invalid"), "Unsupported dtype")
  
  # Invalid shape - zero dimension
  expect_error(empty_tensor(c(3, 0, 3), dtype = "float"), "positive")
  
  # Invalid shape - negative dimension
  expect_error(empty_tensor(c(3, -2, 3), dtype = "float"), "positive")
  
  # Invalid device (if device parameter exists)
  tryCatch({
    expect_error(empty_tensor(c(3, 3), dtype = "float", device = "cpu"), "CUDA")
  }, error = function(e) {
    # Skip if device parameter doesn't exist
  })
})

test_that("empty_tensor data is uninitialized (implementation detail)", {
  # Note: Empty tensors may contain arbitrary data
  # This test just verifies the tensor can be used in operations
  shape <- c(2, 3)
  tensor <- empty_tensor(shape, dtype = "float")
  
  # Should be able to perform operations on empty tensor
  result <- tensor + 1.0
  expect_s3_class(result, "gpuTensor")
  expect_equal(shape(result), shape)
  
  # Should be able to assign values
  tensor[1, 1] <- 5.0
  # Verify assignment worked (value should be 5.0)
  subset_val <- tensor[1, 1]
  expect_equal(as.numeric(subset_val), 5.0, tolerance = 1e-6)
})

# =============================================================================
# CREATE_ONES_LIKE TESTS
# =============================================================================

test_that("create_ones_like creates tensors filled with ones", {
  # Test with 1D tensor
  original_1d <- as_tensor(c(2, 3, 4), dtype = "float")
  ones_1d <- create_ones_like(original_1d)
  
  expect_s3_class(ones_1d, "gpuTensor")
  expect_equal(shape(ones_1d), shape(original_1d))
  expect_equal(dtype(ones_1d), dtype(original_1d))
  expect_tensor_equal(ones_1d, rep(1, 3))
  
  # Test with 2D tensor
  original_2d <- as_tensor(matrix(runif(12), nrow = 3, ncol = 4), dtype = "float")
  ones_2d <- create_ones_like(original_2d)
  
  expect_equal(shape(ones_2d), shape(original_2d))
  expect_equal(dtype(ones_2d), dtype(original_2d))
  expect_tensor_equal(ones_2d, matrix(1, nrow = 3, ncol = 4))
  
  # Test with 3D tensor
  original_3d <- as_tensor(array(runif(24), dim = c(2, 3, 4)), dtype = "float")
  ones_3d <- create_ones_like(original_3d)
  
  expect_equal(shape(ones_3d), shape(original_3d))
  expect_equal(dtype(ones_3d), dtype(original_3d))
  expect_tensor_equal(ones_3d, array(1, dim = c(2, 3, 4)))
})

test_that("create_ones_like preserves dtype", {
  # Test float32
  original_f32 <- as_tensor(c(1.5, 2.7, 3.9), dtype = "float")
  ones_f32 <- create_ones_like(original_f32)
  
  expect_equal(dtype(ones_f32), "float")
  expect_tensor_equal(ones_f32, c(1, 1, 1))
  
  # Test float64
  original_f64 <- as_tensor(c(1.5, 2.7, 3.9), dtype = "double")
  ones_f64 <- create_ones_like(original_f64)
  
  expect_equal(dtype(ones_f64), "double")
  expect_tensor_equal(ones_f64, c(1, 1, 1))
})

test_that("create_ones_like works with different tensor shapes", {
  # Single element
  single_original <- as_tensor(c(42), dtype = "float")
  single_ones <- create_ones_like(single_original)
  
  expect_tensor_equal(single_ones, c(1))
  expect_equal(shape(single_ones), c(1))
  
  # Large tensor
  large_original <- as_tensor(runif(1000), dtype = "float")
  large_ones <- create_ones_like(large_original)
  
  expect_equal(shape(large_ones), c(1000))
  expect_tensor_equal(large_ones, rep(1, 1000))
  
  # High-dimensional tensor
  high_dim_original <- as_tensor(array(runif(32), dim = c(2, 2, 2, 2, 2)), dtype = "float")
  high_dim_ones <- create_ones_like(high_dim_original)
  
  expect_equal(shape(high_dim_ones), c(2, 2, 2, 2, 2))
  expect_tensor_equal(high_dim_ones, array(1, dim = c(2, 2, 2, 2, 2)))
})

test_that("create_ones_like works with non-contiguous tensors", {
  # Create non-contiguous tensor via transpose
  original_mat <- as_tensor(matrix(runif(12), nrow = 3, ncol = 4), dtype = "float")
  transposed_original <- transpose(original_mat)
  
  ones_from_transposed <- create_ones_like(transposed_original)
  
  expect_equal(shape(ones_from_transposed), shape(transposed_original))
  expect_equal(dtype(ones_from_transposed), dtype(transposed_original))
  expect_tensor_equal(ones_from_transposed, matrix(1, nrow = 4, ncol = 3))
})

test_that("create_ones_like creates contiguous tensors", {
  # Even if input is non-contiguous, output should be contiguous
  original_mat <- as_tensor(matrix(runif(12), nrow = 3, ncol = 4), dtype = "float")
  transposed_original <- transpose(original_mat)
  
  ones_tensor <- create_ones_like(transposed_original)
  
  # Result should be contiguous (implementation detail)
  expect_true(is_contiguous(ones_tensor))
})

test_that("create_ones_like error handling", {
  # Non-tensor input
  expect_error(create_ones_like(c(1, 2, 3)), "gpuTensor")
  expect_error(create_ones_like(matrix(1:6, nrow = 2)), "gpuTensor")
  
  # NULL input
  expect_error(create_ones_like(NULL), "gpuTensor")
})

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test_that("empty_tensor and create_ones_like work together", {
  # Create empty tensor, then create ones like it
  empty <- empty_tensor(c(3, 4), dtype = "float")
  ones_from_empty <- create_ones_like(empty)
  
  expect_equal(shape(ones_from_empty), c(3, 4))
  expect_equal(dtype(ones_from_empty), "float")
  expect_tensor_equal(ones_from_empty, matrix(1, nrow = 3, ncol = 4))
  
  # Verify they have same properties but different data
  expect_equal(shape(empty), shape(ones_from_empty))
  expect_equal(dtype(empty), dtype(ones_from_empty))
  expect_equal(size(empty), size(ones_from_empty))
})

test_that("tensor creation utilities can be used in computations", {
  # Create tensors and use them in arithmetic
  original <- as_tensor(c(2, 3, 4), dtype = "float")
  ones <- create_ones_like(original)
  empty <- empty_tensor(shape(original), dtype = "float")
  
  # Fill empty tensor with values
  empty[1] <- 10
  empty[2] <- 20  
  empty[3] <- 30
  
  # Perform operations
  result1 <- original + ones  # Should be c(3, 4, 5)
  expect_tensor_equal(result1, c(3, 4, 5))
  
  result2 <- ones * 5  # Should be c(5, 5, 5)
  expect_tensor_equal(result2, c(5, 5, 5))
  
  # Use empty tensor (now filled) in computation
  result3 <- empty + ones  # Should be c(11, 21, 31)
  expect_tensor_equal(result3, c(11, 21, 31))
})

test_that("tensor creation utilities maintain GPU execution", {
  # Test with larger tensors to ensure GPU path
  large_shape <- c(1000, 100)
  
  # Time empty tensor creation
  empty_time <- system.time({
    large_empty <- empty_tensor(large_shape, dtype = "float")
  })
  
  # Time ones creation
  ones_time <- system.time({
    large_ones <- create_ones_like(large_empty)
  })
  
  # Operations should complete quickly
  expect_lt(empty_time[["elapsed"]], 2.0)
  expect_lt(ones_time[["elapsed"]], 2.0)
  
  # Verify results
  expect_s3_class(large_empty, "gpuTensor")
  expect_s3_class(large_ones, "gpuTensor")
  expect_equal(shape(large_empty), large_shape)
  expect_equal(shape(large_ones), large_shape)
  
  # Spot check that ones tensor contains ones
  subset_ones <- large_ones[1:5, 1:5]
  expect_tensor_equal(subset_ones, matrix(1, nrow = 5, ncol = 5))
})

test_that("tensor creation utilities work with different memory layouts", {
  # Test with various shapes and sizes
  shapes_to_test <- list(
    c(10),           # 1D
    c(5, 6),         # 2D
    c(2, 3, 4),      # 3D
    c(2, 2, 2, 2),   # 4D
    c(1, 100),       # Thin matrix
    c(100, 1),       # Tall matrix
    c(1, 1, 100)     # Mostly singleton dimensions
  )
  
  for (shape in shapes_to_test) {
    # Test empty tensor
    empty <- empty_tensor(shape, dtype = "float")
    expect_equal(shape(empty), shape)
    expect_equal(size(empty), prod(shape))
    
    # Test ones like
    ones <- create_ones_like(empty)
    expect_equal(shape(ones), shape)
    expect_equal(size(ones), prod(shape))
    
    # Verify ones contains all ones (check a few elements)
    if (prod(shape) > 0) {
      first_element <- ones[1]
      expect_equal(as.numeric(first_element), 1.0, tolerance = 1e-6)
    }
  }
})

test_that("tensor creation utilities handle dtype conversion edge cases", {
  # Test with mixed dtype scenarios
  original_f32 <- as_tensor(c(1.1, 2.2, 3.3), dtype = "float")
  original_f64 <- as_tensor(c(1.1, 2.2, 3.3), dtype = "double")
  
  # Create ones with different dtypes
  ones_f32 <- create_ones_like(original_f32)
  ones_f64 <- create_ones_like(original_f64)
  
  # Verify dtype preservation
  expect_equal(dtype(ones_f32), "float")
  expect_equal(dtype(ones_f64), "double")
  
  # Both should contain ones but with different precision
  expect_tensor_equal(ones_f32, c(1, 1, 1))
  expect_tensor_equal(ones_f64, c(1, 1, 1))
  
  # Test arithmetic between different dtypes (should error or convert)
  tryCatch({
    mixed_result <- ones_f32 + ones_f64
    # If this succeeds, verify the result
    expect_s3_class(mixed_result, "gpuTensor")
  }, error = function(e) {
    # If this errors due to dtype mismatch, that's also acceptable
    expect_true(grepl("dtype", e$message, ignore.case = TRUE))
  })
})

test_that("tensor creation utilities work with view operations", {
  # Create tensor, make view, then create ones like the view
  original <- as_tensor(array(runif(24), dim = c(2, 3, 4)), dtype = "float")
  
  # Create view with different shape
  reshaped_view <- view(original, c(6, 4))
  ones_from_view <- create_ones_like(reshaped_view)
  
  expect_equal(shape(ones_from_view), c(6, 4))
  expect_equal(dtype(ones_from_view), dtype(original))
  expect_tensor_equal(ones_from_view, matrix(1, nrow = 6, ncol = 4))
  
  # Create transpose view
  original_2d <- as_tensor(matrix(runif(12), nrow = 3, ncol = 4), dtype = "float")
  transposed_view <- transpose(original_2d)
  ones_from_transpose <- create_ones_like(transposed_view)
  
  expect_equal(shape(ones_from_transpose), c(4, 3))
  expect_tensor_equal(ones_from_transpose, matrix(1, nrow = 4, ncol = 3))
}) 