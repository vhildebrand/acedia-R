context("Range and extrema helpers: range, which.max, which.min")

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
# RANGE TESTS
# =============================================================================

test_that("range works correctly for 1D tensors", {
  # Test basic range
  data_1d <- c(3, 1, 4, 1, 5, 9, 2, 6)
  tensor_1d <- as_tensor(data_1d, dtype = "float")
  
  result <- range(tensor_1d)
  expected <- range(data_1d)
  
  expect_true(is.numeric(result))
  expect_equal(length(result), 2)
  expect_equal(result, expected, tolerance = 1e-6)
})

test_that("range works correctly for different dtypes", {
  data <- c(1.5, 2.7, 0.3, 4.9, 1.1)
  
  # Test float32
  tensor_f32 <- as_tensor(data, dtype = "float")
  result_f32 <- range(tensor_f32)
  expected <- range(data)
  
  expect_equal(result_f32, expected, tolerance = 1e-6)
  
  # Test float64
  tensor_f64 <- as_tensor(data, dtype = "double")
  result_f64 <- range(tensor_f64)
  
  expect_equal(result_f64, expected, tolerance = 1e-15)
})

test_that("range works correctly for 2D tensors", {
  # Test with matrix
  mat_data <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8), nrow = 3, ncol = 4)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  result <- range(mat_tensor)
  expected <- range(mat_data)
  
  expect_equal(result, expected, tolerance = 1e-6)
})

test_that("range works correctly for 3D tensors", {
  # Test with 3D array
  arr_data <- array(runif(24, min = -5, max = 10), dim = c(2, 3, 4))
  arr_tensor <- as_tensor(arr_data, dtype = "float")
  
  result <- range(arr_tensor)
  expected <- range(arr_data)
  
  expect_equal(result, expected, tolerance = 1e-6)
})

test_that("range handles edge cases", {
  # Single element
  single_tensor <- as_tensor(c(42), dtype = "float")
  result_single <- range(single_tensor)
  expect_equal(result_single, c(42, 42))
  
  # All same values
  same_data <- rep(3.14, 5)
  same_tensor <- as_tensor(same_data, dtype = "float")
  result_same <- range(same_tensor)
  expect_equal(result_same, c(3.14, 3.14), tolerance = 1e-6)
  
  # Negative values
  neg_data <- c(-5, -1, -10, -3, -7)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  result_neg <- range(neg_tensor)
  expect_equal(result_neg, range(neg_data), tolerance = 1e-6)
  
  # Mixed positive/negative
  mixed_data <- c(-2, 5, -8, 3, 0, -1, 7)
  mixed_tensor <- as_tensor(mixed_data, dtype = "float")
  result_mixed <- range(mixed_tensor)
  expect_equal(result_mixed, range(mixed_data), tolerance = 1e-6)
})

test_that("range works on non-contiguous tensors", {
  # Create non-contiguous tensor via transpose
  mat_data <- matrix(c(1, 8, 3, 2, 9, 4, 5, 1, 6), nrow = 3, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  transposed <- transpose(mat_tensor)
  
  result <- range(transposed)
  expected <- range(t(mat_data))
  
  expect_equal(result, expected, tolerance = 1e-6)
})

# =============================================================================
# WHICH.MAX TESTS
# =============================================================================

test_that("which.max works correctly for 1D tensors", {
  # Test basic which.max
  data_1d <- c(3, 1, 4, 1, 5, 9, 2, 6)
  tensor_1d <- as_tensor(data_1d, dtype = "float")
  
  result <- which.max(tensor_1d)
  expected <- which.max(data_1d)
  
  expect_true(is.numeric(result))
  expect_equal(length(result), 1)
  expect_equal(result, expected)
})

test_that("which.max works correctly for different dtypes", {
  data <- c(1.5, 4.9, 2.7, 0.3, 1.1)
  
  # Test float32
  tensor_f32 <- as_tensor(data, dtype = "float")
  result_f32 <- which.max(tensor_f32)
  expected <- which.max(data)
  
  expect_equal(result_f32, expected)
  
  # Test float64
  tensor_f64 <- as_tensor(data, dtype = "double")
  result_f64 <- which.max(tensor_f64)
  
  expect_equal(result_f64, expected)
})

test_that("which.max works correctly for 2D tensors", {
  # Test with matrix (should find max in flattened order)
  mat_data <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8), nrow = 3, ncol = 4)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  result <- which.max(mat_tensor)
  expected <- which.max(as.vector(mat_data))
  
  expect_equal(result, expected)
})

test_that("which.max handles edge cases", {
  # Single element
  single_tensor <- as_tensor(c(42), dtype = "float")
  result_single <- which.max(single_tensor)
  expect_equal(result_single, 1)
  
  # Multiple maximum values (should return first)
  multi_max_data <- c(5, 2, 5, 1, 5, 3)
  multi_max_tensor <- as_tensor(multi_max_data, dtype = "float")
  result_multi <- which.max(multi_max_tensor)
  expected_multi <- which.max(multi_max_data)
  expect_equal(result_multi, expected_multi)
  
  # Negative values
  neg_data <- c(-5, -1, -10, -3, -7)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  result_neg <- which.max(neg_tensor)
  expect_equal(result_neg, which.max(neg_data))
  
  # All same values
  same_data <- rep(3.14, 5)
  same_tensor <- as_tensor(same_data, dtype = "float")
  result_same <- which.max(same_tensor)
  expect_equal(result_same, 1)  # Should return first occurrence
})

test_that("which.max works on non-contiguous tensors", {
  # Create non-contiguous tensor via transpose
  mat_data <- matrix(c(1, 8, 3, 2, 9, 4, 5, 1, 6), nrow = 3, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  transposed <- transpose(mat_tensor)
  
  result <- which.max(transposed)
  expected <- which.max(as.vector(t(mat_data)))
  
  expect_equal(result, expected)
})

# =============================================================================
# WHICH.MIN TESTS
# =============================================================================

test_that("which.min works correctly for 1D tensors", {
  # Test basic which.min
  data_1d <- c(3, 1, 4, 1, 5, 9, 2, 6)
  tensor_1d <- as_tensor(data_1d, dtype = "float")
  
  result <- which.min(tensor_1d)
  expected <- which.min(data_1d)
  
  expect_true(is.numeric(result))
  expect_equal(length(result), 1)
  expect_equal(result, expected)
})

test_that("which.min works correctly for different dtypes", {
  data <- c(1.5, 0.3, 2.7, 4.9, 1.1)
  
  # Test float32
  tensor_f32 <- as_tensor(data, dtype = "float")
  result_f32 <- which.min(tensor_f32)
  expected <- which.min(data)
  
  expect_equal(result_f32, expected)
  
  # Test float64
  tensor_f64 <- as_tensor(data, dtype = "double")
  result_f64 <- which.min(tensor_f64)
  
  expect_equal(result_f64, expected)
})

test_that("which.min works correctly for 2D tensors", {
  # Test with matrix (should find min in flattened order)
  mat_data <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8), nrow = 3, ncol = 4)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  result <- which.min(mat_tensor)
  expected <- which.min(as.vector(mat_data))
  
  expect_equal(result, expected)
})

test_that("which.min handles edge cases", {
  # Single element
  single_tensor <- as_tensor(c(42), dtype = "float")
  result_single <- which.min(single_tensor)
  expect_equal(result_single, 1)
  
  # Multiple minimum values (should return first)
  multi_min_data <- c(5, 2, 5, 2, 3, 2)
  multi_min_tensor <- as_tensor(multi_min_data, dtype = "float")
  result_multi <- which.min(multi_min_tensor)
  expected_multi <- which.min(multi_min_data)
  expect_equal(result_multi, expected_multi)
  
  # Negative values
  neg_data <- c(-5, -1, -10, -3, -7)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  result_neg <- which.min(neg_tensor)
  expect_equal(result_neg, which.min(neg_data))
  
  # All same values
  same_data <- rep(3.14, 5)
  same_tensor <- as_tensor(same_data, dtype = "float")
  result_same <- which.min(same_tensor)
  expect_equal(result_same, 1)  # Should return first occurrence
})

test_that("which.min works on non-contiguous tensors", {
  # Create non-contiguous tensor via transpose
  mat_data <- matrix(c(1, 8, 3, 2, 9, 4, 5, 1, 6), nrow = 3, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  transposed <- transpose(mat_tensor)
  
  result <- which.min(transposed)
  expected <- which.min(as.vector(t(mat_data)))
  
  expect_equal(result, expected)
})

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test_that("range and which.max/which.min are consistent", {
  # Test that range bounds match which.max/which.min results
  data <- c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8)
  tensor <- as_tensor(data, dtype = "float")
  
  # Get range and indices
  range_result <- range(tensor)
  max_idx <- which.max(tensor)
  min_idx <- which.min(tensor)
  
  # Verify consistency
  expect_equal(range_result[1], data[min_idx], tolerance = 1e-6)  # min value
  expect_equal(range_result[2], data[max_idx], tolerance = 1e-6)  # max value
})

test_that("extrema helpers work with large tensors", {
  # Test with larger tensors to ensure GPU execution
  n <- 10000
  set.seed(123)
  data_large <- runif(n, min = -100, max = 100)
  tensor_large <- as_tensor(data_large, dtype = "float")
  
  # Test range
  range_large <- range(tensor_large)
  expected_range <- range(data_large)
  expect_equal(range_large, expected_range, tolerance = 1e-6)
  
  # Test which.max
  max_idx_large <- which.max(tensor_large)
  expected_max_idx <- which.max(data_large)
  expect_equal(max_idx_large, expected_max_idx)
  
  # Test which.min
  min_idx_large <- which.min(tensor_large)
  expected_min_idx <- which.min(data_large)
  expect_equal(min_idx_large, expected_min_idx)
  
  # Verify values at indices
  expect_equal(data_large[max_idx_large], max(data_large), tolerance = 1e-6)
  expect_equal(data_large[min_idx_large], min(data_large), tolerance = 1e-6)
})

test_that("extrema helpers work with different tensor shapes", {
  # Test 3D tensor
  arr_data <- array(runif(60, min = -10, max = 10), dim = c(3, 4, 5))
  arr_tensor <- as_tensor(arr_data, dtype = "float")
  
  # Range should work across all dimensions
  range_3d <- range(arr_tensor)
  expected_range_3d <- range(arr_data)
  expect_equal(range_3d, expected_range_3d, tolerance = 1e-6)
  
  # which.max/which.min should work on flattened tensor
  max_idx_3d <- which.max(arr_tensor)
  min_idx_3d <- which.min(arr_tensor)
  
  expected_max_idx_3d <- which.max(as.vector(arr_data))
  expected_min_idx_3d <- which.min(as.vector(arr_data))
  
  expect_equal(max_idx_3d, expected_max_idx_3d)
  expect_equal(min_idx_3d, expected_min_idx_3d)
})

test_that("extrema helpers handle special values", {
  # Test with zeros
  zero_data <- c(0, 1, 0, 2, 0, 3)
  zero_tensor <- as_tensor(zero_data, dtype = "float")
  
  range_zero <- range(zero_tensor)
  expect_equal(range_zero, c(0, 3), tolerance = 1e-6)
  
  max_idx_zero <- which.max(zero_tensor)
  min_idx_zero <- which.min(zero_tensor)
  expect_equal(max_idx_zero, which.max(zero_data))
  expect_equal(min_idx_zero, which.min(zero_data))
  
  # Test with very small differences
  small_diff_data <- c(1.0000001, 1.0000002, 1.0000000, 1.0000003)
  small_diff_tensor <- as_tensor(small_diff_data, dtype = "double")  # Use double for precision
  
  range_small <- range(small_diff_tensor)
  expected_small <- range(small_diff_data)
  expect_equal(range_small, expected_small, tolerance = 1e-10)
  
  max_idx_small <- which.max(small_diff_tensor)
  min_idx_small <- which.min(small_diff_tensor)
  expect_equal(max_idx_small, which.max(small_diff_data))
  expect_equal(min_idx_small, which.min(small_diff_data))
})

test_that("extrema helpers maintain performance with GPU execution", {
  # Performance test with moderately large tensor
  n <- 50000
  data_perf <- runif(n, min = -1000, max = 1000)
  tensor_perf <- as_tensor(data_perf, dtype = "float")
  
  # Time the operations
  range_time <- system.time({
    range_result <- range(tensor_perf)
  })
  
  max_time <- system.time({
    max_idx <- which.max(tensor_perf)
  })
  
  min_time <- system.time({
    min_idx <- which.min(tensor_perf)
  })
  
  # Operations should complete quickly
  expect_lt(range_time[["elapsed"]], 1.0)
  expect_lt(max_time[["elapsed"]], 1.0)
  expect_lt(min_time[["elapsed"]], 1.0)
  
  # Verify correctness
  expected_range <- range(data_perf)
  expected_max_idx <- which.max(data_perf)
  expected_min_idx <- which.min(data_perf)
  
  expect_equal(range_result, expected_range, tolerance = 1e-6)
  expect_equal(max_idx, expected_max_idx)
  expect_equal(min_idx, expected_min_idx)
}) 