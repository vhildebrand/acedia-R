context("Cumulative operations: cumsum, cumprod, diff")

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
# CUMULATIVE SUM TESTS
# =============================================================================

test_that("cumsum works correctly for 1D tensors", {
  # Test basic cumsum
  data_1d <- c(1, 2, 3, 4, 5)
  tensor_1d <- as_tensor(data_1d, dtype = "float")
  
  result <- cumsum(tensor_1d)
  expected <- cumsum(data_1d)
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), length(data_1d))
  expect_equal(dtype(result), "float")
})

test_that("cumsum works correctly for different dtypes", {
  data <- c(1.5, 2.7, 3.1, 4.9)
  
  # Test float32
  tensor_f32 <- as_tensor(data, dtype = "float")
  result_f32 <- cumsum(tensor_f32)
  expected <- cumsum(data)
  
  expect_tensor_equal(result_f32, expected)
  expect_equal(dtype(result_f32), "float")
  
  # Test float64
  tensor_f64 <- as_tensor(data, dtype = "double")
  result_f64 <- cumsum(tensor_f64)
  
  expect_tensor_equal(result_f64, expected)
  expect_equal(dtype(result_f64), "double")
})

test_that("cumsum works correctly for 2D tensors", {
  # Test with matrix (should work element-wise in memory order)
  mat_data <- matrix(1:12, nrow = 3, ncol = 4)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  result <- cumsum(mat_tensor)
  expected_vec <- cumsum(as.vector(mat_data))
  expected_mat <- array(expected_vec, dim = dim(mat_data))
  
  expect_tensor_equal(result, expected_mat)
  expect_equal(shape(result), c(3, 4))
})

test_that("cumsum handles edge cases", {
  # Single element
  single_tensor <- as_tensor(c(42), dtype = "float")
  result_single <- cumsum(single_tensor)
  expect_tensor_equal(result_single, c(42))
  
  # Empty tensor (if supported)
  if (exists("empty_tensor")) {
    tryCatch({
      empty <- empty_tensor(c(0), dtype = "float")
      result_empty <- cumsum(empty)
      expect_equal(size(result_empty), 0)
    }, error = function(e) {
      skip("Empty tensor cumsum not supported")
    })
  }
  
  # Negative values
  neg_data <- c(-1, 2, -3, 4, -5)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  result_neg <- cumsum(neg_tensor)
  expect_tensor_equal(result_neg, cumsum(neg_data))
})

# =============================================================================
# CUMULATIVE PRODUCT TESTS
# =============================================================================

test_that("cumprod works correctly for 1D tensors", {
  # Test basic cumprod with small values to avoid overflow
  data_1d <- c(1, 2, 1.5, 1.2, 1.1)
  tensor_1d <- as_tensor(data_1d, dtype = "float")
  
  result <- cumprod(tensor_1d)
  expected <- cumprod(data_1d)
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), length(data_1d))
  expect_equal(dtype(result), "float")
})

test_that("cumprod works correctly for different dtypes", {
  data <- c(1.1, 1.2, 1.05, 0.9)
  
  # Test float32
  tensor_f32 <- as_tensor(data, dtype = "float")
  result_f32 <- cumprod(tensor_f32)
  expected <- cumprod(data)
  
  expect_tensor_equal(result_f32, expected)
  expect_equal(dtype(result_f32), "float")
  
  # Test float64
  tensor_f64 <- as_tensor(data, dtype = "double")
  result_f64 <- cumprod(tensor_f64)
  
  expect_tensor_equal(result_f64, expected)
  expect_equal(dtype(result_f64), "double")
})

test_that("cumprod works correctly for 2D tensors", {
  # Test with matrix (should work element-wise in memory order)
  mat_data <- matrix(c(1, 1.1, 1.2, 0.9, 1.05, 1.15), nrow = 2, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  result <- cumprod(mat_tensor)
  expected_vec <- cumprod(as.vector(mat_data))
  expected_mat <- array(expected_vec, dim = dim(mat_data))
  
  expect_tensor_equal(result, expected_mat)
  expect_equal(shape(result), c(2, 3))
})

test_that("cumprod handles edge cases", {
  # Single element
  single_tensor <- as_tensor(c(3.14), dtype = "float")
  result_single <- cumprod(single_tensor)
  expect_tensor_equal(result_single, c(3.14))
  
  # With zeros
  zero_data <- c(1, 2, 0, 4, 5)
  zero_tensor <- as_tensor(zero_data, dtype = "float")
  result_zero <- cumprod(zero_tensor)
  expect_tensor_equal(result_zero, cumprod(zero_data))
  
  # With negative values
  neg_data <- c(-1, 2, -1.5, 1.2)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  result_neg <- cumprod(neg_tensor)
  expect_tensor_equal(result_neg, cumprod(neg_data))
})

# =============================================================================
# DIFF TESTS
# =============================================================================

test_that("diff works correctly for 1D tensors", {
  # Test basic diff
  data_1d <- c(1, 4, 6, 8, 12)
  tensor_1d <- as_tensor(data_1d, dtype = "float")
  
  result <- diff(tensor_1d)
  expected <- diff(data_1d)
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), length(expected))
  expect_equal(dtype(result), "float")
})

test_that("diff works correctly with different lag values", {
  data <- c(1, 3, 6, 10, 15, 21)
  tensor <- as_tensor(data, dtype = "float")
  
  # Test lag = 1 (default)
  result_lag1 <- diff(tensor, lag = 1)
  expected_lag1 <- diff(data, lag = 1)
  expect_tensor_equal(result_lag1, expected_lag1)
  
  # Test lag = 2
  result_lag2 <- diff(tensor, lag = 2)
  expected_lag2 <- diff(data, lag = 2)
  expect_tensor_equal(result_lag2, expected_lag2)
  
  # Test lag = 3
  result_lag3 <- diff(tensor, lag = 3)
  expected_lag3 <- diff(data, lag = 3)
  expect_tensor_equal(result_lag3, expected_lag3)
})

test_that("diff works correctly with different differences values", {
  data <- c(1, 2, 4, 7, 11, 16, 22)
  tensor <- as_tensor(data, dtype = "float")
  
  # Test differences = 1 (default)
  result_diff1 <- diff(tensor, differences = 1)
  expected_diff1 <- diff(data, differences = 1)
  expect_tensor_equal(result_diff1, expected_diff1)
  
  # Test differences = 2
  result_diff2 <- diff(tensor, differences = 2)
  expected_diff2 <- diff(data, differences = 2)
  expect_tensor_equal(result_diff2, expected_diff2)
  
  # Test differences = 3
  result_diff3 <- diff(tensor, differences = 3)
  expected_diff3 <- diff(data, differences = 3)
  expect_tensor_equal(result_diff3, expected_diff3)
})

test_that("diff works correctly for different dtypes", {
  data <- c(1.5, 3.2, 5.1, 7.8, 10.9)
  
  # Test float32
  tensor_f32 <- as_tensor(data, dtype = "float")
  result_f32 <- diff(tensor_f32)
  expected <- diff(data)
  
  expect_tensor_equal(result_f32, expected)
  expect_equal(dtype(result_f32), "float")
  
  # Test float64
  tensor_f64 <- as_tensor(data, dtype = "double")
  result_f64 <- diff(tensor_f64)
  
  expect_tensor_equal(result_f64, expected)
  expect_equal(dtype(result_f64), "double")
})

test_that("diff works correctly for 2D tensors", {
  # Test with matrix (should work element-wise in memory order)
  mat_data <- matrix(c(1, 2, 4, 7, 11, 16), nrow = 2, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  result <- diff(mat_tensor)
  expected_vec <- diff(as.vector(mat_data))
  expected_mat <- array(expected_vec, dim = c(dim(mat_data)[1], dim(mat_data)[2]))
  
  expect_tensor_equal(result, expected_mat)
})

test_that("diff handles edge cases", {
  # Two elements (minimum for diff)
  two_tensor <- as_tensor(c(5, 8), dtype = "float")
  result_two <- diff(two_tensor)
  expect_tensor_equal(result_two, c(3))
  
  # Single element (should result in empty)
  single_tensor <- as_tensor(c(42), dtype = "float")
  result_single <- diff(single_tensor)
  expected_single <- diff(c(42))
  expect_equal(length(as.vector(result_single)), length(expected_single))
  
  # Negative differences
  neg_data <- c(10, 7, 5, 8, 3)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  result_neg <- diff(neg_tensor)
  expect_tensor_equal(result_neg, diff(neg_data))
})

test_that("diff error handling", {
  tensor <- as_tensor(c(1, 2, 3, 4), dtype = "float")
  
  # Invalid lag
  expect_error(diff(tensor, lag = 0), "lag must be positive")
  expect_error(diff(tensor, lag = -1), "lag must be positive")
  
  # Invalid differences
  expect_error(diff(tensor, differences = 0), "differences must be positive")
  expect_error(diff(tensor, differences = -1), "differences must be positive")
})

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test_that("cumulative operations work on non-contiguous tensors", {
  # Create non-contiguous tensor via transpose
  mat_data <- matrix(1:12, nrow = 3, ncol = 4)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  transposed <- transpose(mat_tensor)
  
  # Test cumsum on transposed tensor
  result_cumsum <- cumsum(transposed)
  expected_cumsum_vec <- cumsum(as.vector(t(mat_data)))
  expected_cumsum <- array(expected_cumsum_vec, dim = c(4, 3))
  expect_tensor_equal(result_cumsum, expected_cumsum)
  
  # Test cumprod on transposed tensor
  # Use smaller values to avoid overflow
  small_data <- matrix(c(1, 1.1, 1.05, 0.95, 1.2, 0.9), nrow = 2, ncol = 3)
  small_tensor <- as_tensor(small_data, dtype = "float")
  small_transposed <- transpose(small_tensor)
  
  result_cumprod <- cumprod(small_transposed)
  expected_cumprod_vec <- cumprod(as.vector(t(small_data)))
  expected_cumprod <- array(expected_cumprod_vec, dim = c(3, 2))
  expect_tensor_equal(result_cumprod, expected_cumprod)
})

test_that("cumulative operations can be chained", {
  data <- c(1, 2, 1, 3, 1, 2)
  tensor <- as_tensor(data, dtype = "float")
  
  # Chain: cumsum -> diff
  cumsum_result <- cumsum(tensor)
  diff_result <- diff(cumsum_result)
  
  # diff(cumsum(x)) should approximately equal the original data (except first element)
  expected_chain <- diff(cumsum(data))
  expect_tensor_equal(diff_result, expected_chain)
  
  # Chain: cumprod -> log -> diff (for positive values)
  pos_data <- c(1.1, 1.2, 1.05, 1.15, 1.08)
  pos_tensor <- as_tensor(pos_data, dtype = "float")
  
  cumprod_result <- cumprod(pos_tensor)
  log_result <- log(cumprod_result)
  final_diff <- diff(log_result)
  
  expected_final <- diff(log(cumprod(pos_data)))
  expect_tensor_equal(final_diff, expected_final)
})

test_that("cumulative operations maintain GPU execution", {
  # Test with larger tensors
  n <- 1000
  data_large <- stats::runif(n, 0.5, 2.0)  # Positive values for cumprod
  tensor_large <- as_tensor(data_large, dtype = "float")
  
  # All operations should return GPU tensors
  cumsum_large <- cumsum(tensor_large)
  cumprod_large <- cumprod(tensor_large)
  diff_large <- diff(tensor_large)
  
  expect_s3_class(cumsum_large, "gpuTensor")
  expect_s3_class(cumprod_large, "gpuTensor")
  expect_s3_class(diff_large, "gpuTensor")
  
  # Verify shapes
  expect_equal(shape(cumsum_large), n)
  expect_equal(shape(cumprod_large), n)
  expect_equal(shape(diff_large), n - 1)
  
  # Spot check correctness
  expect_equal(as.vector(cumsum_large)[1:5], cumsum(data_large[1:5]), tolerance = 1e-6)
  expect_equal(as.vector(diff_large)[1:5], diff(data_large)[1:5], tolerance = 1e-6)
}) 