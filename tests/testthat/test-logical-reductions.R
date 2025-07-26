context("Logical reductions: any, all")

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
# ANY TESTS
# =============================================================================

test_that("any works correctly for 1D tensors with logical values", {
  # Test basic any with logical-like values (0/1)
  data_true <- c(0, 0, 1, 0)  # Should be TRUE
  tensor_true <- as_tensor(data_true, dtype = "float")
  
  result_true <- any(tensor_true)
  expected_true <- any(as.logical(data_true))
  
  expect_true(is.logical(result_true))
  expect_equal(result_true, expected_true)
  
  # Test with all zeros (should be FALSE)
  data_false <- c(0, 0, 0, 0)
  tensor_false <- as_tensor(data_false, dtype = "float")
  
  result_false <- any(tensor_false)
  expected_false <- any(as.logical(data_false))
  
  expect_equal(result_false, expected_false)
  
  # Test with all ones (should be TRUE)
  data_all_true <- c(1, 1, 1, 1)
  tensor_all_true <- as_tensor(data_all_true, dtype = "float")
  
  result_all_true <- any(tensor_all_true)
  expected_all_true <- any(as.logical(data_all_true))
  
  expect_equal(result_all_true, expected_all_true)
})

test_that("any works correctly for different dtypes", {
  # Test with float32
  data_f32 <- c(0, 0, 0.5, 0)  # Non-zero values are truthy
  tensor_f32 <- as_tensor(data_f32, dtype = "float")
  
  result_f32 <- any(tensor_f32)
  expected_f32 <- any(data_f32 != 0)
  
  expect_equal(result_f32, expected_f32)
  
  # Test with float64
  data_f64 <- c(0, 0, 0, 0)
  tensor_f64 <- as_tensor(data_f64, dtype = "double")
  
  result_f64 <- any(tensor_f64)
  expected_f64 <- any(data_f64 != 0)
  
  expect_equal(result_f64, expected_f64)
})

test_that("any works correctly for 2D tensors", {
  # Test with matrix containing some non-zero values
  mat_data <- matrix(c(0, 0, 1, 0, 0, 0), nrow = 2, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  result <- any(mat_tensor)
  expected <- any(mat_data != 0)
  
  expect_equal(result, expected)
  
  # Test with all-zero matrix
  zero_mat <- matrix(0, nrow = 3, ncol = 3)
  zero_tensor <- as_tensor(zero_mat, dtype = "float")
  
  result_zero <- any(zero_tensor)
  expect_false(result_zero)
})

test_that("any works correctly for 3D tensors", {
  # Test with 3D array
  arr_data <- array(c(0, 0, 0, 1, 0, 0, 0, 0), dim = c(2, 2, 2))
  arr_tensor <- as_tensor(arr_data, dtype = "float")
  
  result <- any(arr_tensor)
  expected <- any(arr_data != 0)
  
  expect_equal(result, expected)
})

test_that("any handles edge cases", {
  # Single element - true
  single_true <- as_tensor(c(1), dtype = "float")
  result_single_true <- any(single_true)
  expect_true(result_single_true)
  
  # Single element - false
  single_false <- as_tensor(c(0), dtype = "float")
  result_single_false <- any(single_false)
  expect_false(result_single_false)
  
  # Negative values (should be truthy)
  neg_data <- c(0, -1, 0, 0)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  result_neg <- any(neg_tensor)
  expect_true(result_neg)
  
  # Small non-zero values
  small_data <- c(0, 0, 1e-10, 0)
  small_tensor <- as_tensor(small_data, dtype = "float")
  result_small <- any(small_tensor)
  expect_true(result_small)
})

test_that("any works on non-contiguous tensors", {
  # Create non-contiguous tensor via transpose
  mat_data <- matrix(c(0, 1, 0, 0, 0, 0), nrow = 2, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  transposed <- transpose(mat_tensor)
  
  result <- any(transposed)
  expected <- any(t(mat_data) != 0)
  
  expect_equal(result, expected)
})

# =============================================================================
# ALL TESTS
# =============================================================================

test_that("all works correctly for 1D tensors with logical values", {
  # Test basic all with all non-zero values (should be TRUE)
  data_true <- c(1, 2, 3, 4)  # All non-zero
  tensor_true <- as_tensor(data_true, dtype = "float")
  
  result_true <- all(tensor_true)
  expected_true <- all(data_true != 0)
  
  expect_true(is.logical(result_true))
  expect_equal(result_true, expected_true)
  
  # Test with some zeros (should be FALSE)
  data_false <- c(1, 0, 3, 4)
  tensor_false <- as_tensor(data_false, dtype = "float")
  
  result_false <- all(tensor_false)
  expected_false <- all(data_false != 0)
  
  expect_equal(result_false, expected_false)
  
  # Test with all zeros (should be FALSE)
  data_all_false <- c(0, 0, 0, 0)
  tensor_all_false <- as_tensor(data_all_false, dtype = "float")
  
  result_all_false <- all(tensor_all_false)
  expected_all_false <- all(data_all_false != 0)
  
  expect_equal(result_all_false, expected_all_false)
})

test_that("all works correctly for different dtypes", {
  # Test with float32 - all non-zero
  data_f32 <- c(1.5, 2.7, 0.1, 3.9)
  tensor_f32 <- as_tensor(data_f32, dtype = "float")
  
  result_f32 <- all(tensor_f32)
  expected_f32 <- all(data_f32 != 0)
  
  expect_equal(result_f32, expected_f32)
  
  # Test with float64 - contains zero
  data_f64 <- c(1.5, 0, 2.7, 3.9)
  tensor_f64 <- as_tensor(data_f64, dtype = "double")
  
  result_f64 <- all(tensor_f64)
  expected_f64 <- all(data_f64 != 0)
  
  expect_equal(result_f64, expected_f64)
})

test_that("all works correctly for 2D tensors", {
  # Test with matrix containing all non-zero values
  mat_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  result <- all(mat_tensor)
  expected <- all(mat_data != 0)
  
  expect_equal(result, expected)
  
  # Test with matrix containing some zeros
  mixed_mat <- matrix(c(1, 0, 3, 4, 5, 6), nrow = 2, ncol = 3)
  mixed_tensor <- as_tensor(mixed_mat, dtype = "float")
  
  result_mixed <- all(mixed_tensor)
  expected_mixed <- all(mixed_mat != 0)
  
  expect_equal(result_mixed, expected_mixed)
})

test_that("all works correctly for 3D tensors", {
  # Test with 3D array - all non-zero
  arr_data <- array(c(1, 2, 3, 4, 5, 6, 7, 8), dim = c(2, 2, 2))
  arr_tensor <- as_tensor(arr_data, dtype = "float")
  
  result <- all(arr_tensor)
  expected <- all(arr_data != 0)
  
  expect_equal(result, expected)
  
  # Test with 3D array containing zero
  arr_zero <- array(c(1, 2, 0, 4, 5, 6, 7, 8), dim = c(2, 2, 2))
  arr_zero_tensor <- as_tensor(arr_zero, dtype = "float")
  
  result_zero <- all(arr_zero_tensor)
  expected_zero <- all(arr_zero != 0)
  
  expect_equal(result_zero, expected_zero)
})

test_that("all handles edge cases", {
  # Single element - true
  single_true <- as_tensor(c(5), dtype = "float")
  result_single_true <- all(single_true)
  expect_true(result_single_true)
  
  # Single element - false
  single_false <- as_tensor(c(0), dtype = "float")
  result_single_false <- all(single_false)
  expect_false(result_single_false)
  
  # Negative values (should be truthy)
  neg_data <- c(-1, -2, -3, -4)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  result_neg <- all(neg_tensor)
  expect_true(result_neg)
  
  # Mixed positive/negative with zero
  mixed_data <- c(-1, 2, 0, -4)
  mixed_tensor <- as_tensor(mixed_data, dtype = "float")
  result_mixed <- all(mixed_tensor)
  expect_false(result_mixed)
  
  # Very small non-zero values
  small_data <- c(1e-10, 2e-10, 3e-10, 4e-10)
  small_tensor <- as_tensor(small_data, dtype = "float")
  result_small <- all(small_tensor)
  expect_true(result_small)
})

test_that("all works on non-contiguous tensors", {
  # Create non-contiguous tensor via transpose
  mat_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  transposed <- transpose(mat_tensor)
  
  result <- all(transposed)
  expected <- all(t(mat_data) != 0)
  
  expect_equal(result, expected)
  
  # Test with transpose containing zero
  mat_zero <- matrix(c(1, 0, 3, 4, 5, 6), nrow = 2, ncol = 3)
  mat_zero_tensor <- as_tensor(mat_zero, dtype = "float")
  transposed_zero <- transpose(mat_zero_tensor)
  
  result_zero <- all(transposed_zero)
  expected_zero <- all(t(mat_zero) != 0)
  
  expect_equal(result_zero, expected_zero)
})

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test_that("any and all are logically consistent", {
  # Test De Morgan's laws and logical consistency
  
  # Case 1: All zeros - any should be FALSE, all should be FALSE
  all_zero <- c(0, 0, 0, 0)
  tensor_zero <- as_tensor(all_zero, dtype = "float")
  
  expect_false(any(tensor_zero))
  expect_false(all(tensor_zero))
  
  # Case 2: All non-zero - any should be TRUE, all should be TRUE
  all_nonzero <- c(1, 2, 3, 4)
  tensor_nonzero <- as_tensor(all_nonzero, dtype = "float")
  
  expect_true(any(tensor_nonzero))
  expect_true(all(tensor_nonzero))
  
  # Case 3: Mixed - any should be TRUE, all should be FALSE
  mixed <- c(1, 0, 3, 4)
  tensor_mixed <- as_tensor(mixed, dtype = "float")
  
  expect_true(any(tensor_mixed))
  expect_false(all(tensor_mixed))
})

test_that("any and all work with comparison results", {
  # Test with results from comparison operations
  data <- c(1, 3, 5, 2, 4)
  tensor <- as_tensor(data, dtype = "float")
  
  # Create comparison tensor (should result in 0/1 values)
  greater_than_3 <- tensor > 3
  
  # any should be TRUE (some values > 3)
  result_any <- any(greater_than_3)
  expected_any <- any(data > 3)
  expect_equal(result_any, expected_any)
  
  # all should be FALSE (not all values > 3)
  result_all <- all(greater_than_3)
  expected_all <- all(data > 3)
  expect_equal(result_all, expected_all)
  
  # Test with all elements satisfying condition
  all_greater <- tensor > 0  # All positive
  result_all_pos <- all(all_greater)
  expected_all_pos <- all(data > 0)
  expect_equal(result_all_pos, expected_all_pos)
})

test_that("logical reductions work with large tensors", {
  # Test with larger tensors to ensure performance
  n <- 10000
  
  # Create tensor with mostly ones and a few zeros
  set.seed(123)
  large_data <- sample(c(0, 1), n, replace = TRUE, prob = c(0.1, 0.9))
  large_tensor <- as_tensor(large_data, dtype = "float")
  
  # Time the operations
  any_time <- system.time({
    any_result <- any(large_tensor)
  })
  
  all_time <- system.time({
    all_result <- all(large_tensor)
  })
  
  # Operations should complete quickly
  expect_lt(any_time[["elapsed"]], 1.0)
  expect_lt(all_time[["elapsed"]], 1.0)
  
  # Verify correctness
  expected_any <- any(large_data != 0)
  expected_all <- all(large_data != 0)
  
  expect_equal(any_result, expected_any)
  expect_equal(all_result, expected_all)
})

test_that("logical reductions work with different tensor shapes", {
  # Test 3D tensor with known pattern
  arr_data <- array(1, dim = c(3, 4, 5))  # All ones
  arr_data[2, 2, 2] <- 0  # Set one element to zero
  
  arr_tensor <- as_tensor(arr_data, dtype = "float")
  
  # any should be TRUE (most elements are non-zero)
  any_result <- any(arr_tensor)
  expect_true(any_result)
  
  # all should be FALSE (one element is zero)
  all_result <- all(arr_tensor)
  expect_false(all_result)
  
  # Verify against R computation
  expected_any <- any(arr_data != 0)
  expected_all <- all(arr_data != 0)
  
  expect_equal(any_result, expected_any)
  expect_equal(all_result, expected_all)
})

test_that("logical reductions handle numerical precision", {
  # Test with very small values that might be affected by floating point precision
  small_data <- c(1e-15, 2e-15, 0, 3e-15)
  small_tensor <- as_tensor(small_data, dtype = "double")  # Use double for precision
  
  any_small <- any(small_tensor)
  all_small <- all(small_tensor)
  
  # Even very small non-zero values should be considered TRUE
  expect_true(any_small)
  expect_false(all_small)  # Because one element is exactly 0
  
  # Test with all very small values
  all_small_data <- c(1e-15, 2e-15, 3e-15, 4e-15)
  all_small_tensor <- as_tensor(all_small_data, dtype = "double")
  
  any_all_small <- any(all_small_tensor)
  all_all_small <- all(all_small_tensor)
  
  expect_true(any_all_small)
  expect_true(all_all_small)
})

test_that("logical reductions work with chained operations", {
  # Test logical reductions on results of other operations
  data <- c(1, 2, 3, 4, 5)
  tensor <- as_tensor(data, dtype = "float")
  
  # Chain: square -> compare -> logical reduction
  squared <- tensor * tensor
  greater_than_10 <- squared > 10
  
  any_gt10 <- any(greater_than_10)
  all_gt10 <- all(greater_than_10)
  
  # Verify against R computation
  squared_r <- data^2
  expected_any <- any(squared_r > 10)
  expected_all <- all(squared_r > 10)
  
  expect_equal(any_gt10, expected_any)
  expect_equal(all_gt10, expected_all)
}) 