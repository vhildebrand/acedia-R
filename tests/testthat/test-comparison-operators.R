context("Comparison operators: ==, !=, <, <=, >, >=")

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
# EQUALITY TESTS (==)
# =============================================================================

test_that("equality operator works correctly", {
  # Test basic equality
  a_data <- c(1, 2, 3, 2, 1)
  b_data <- c(1, 3, 3, 2, 2)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor == b_tensor
  expected <- ifelse(a_data == b_data, 1, 0)  # GPU returns 1/0 instead of TRUE/FALSE
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
})

test_that("equality operator works with scalars", {
  data <- c(1, 2, 3, 2, 1)
  tensor <- as_tensor(data, dtype = "float")
  scalar <- 2
  
  result <- tensor == scalar
  expected <- ifelse(data == scalar, 1, 0)
  
  expect_tensor_equal(result, expected)
})

test_that("equality operator works with different dtypes", {
  data <- c(1.5, 2.7, 3.1, 2.7)
  
  # Test float32
  tensor_f32 <- as_tensor(data, dtype = "float")
  result_f32 <- tensor_f32 == 2.7
  expected <- ifelse(abs(data - 2.7) < 1e-6, 1, 0)  # Account for floating point precision
  
  expect_tensor_equal(result_f32, expected)
  
  # Test float64
  tensor_f64 <- as_tensor(data, dtype = "double")
  result_f64 <- tensor_f64 == 2.7
  
  expect_tensor_equal(result_f64, expected)
})

# =============================================================================
# INEQUALITY TESTS (!=)
# =============================================================================

test_that("inequality operator works correctly", {
  # Test basic inequality
  a_data <- c(1, 2, 3, 2, 1)
  b_data <- c(1, 3, 3, 2, 2)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor != b_tensor
  expected <- ifelse(a_data != b_data, 1, 0)  # GPU returns 1/0 instead of TRUE/FALSE
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
})

test_that("inequality operator works with scalars", {
  data <- c(1, 2, 3, 2, 1)
  tensor <- as_tensor(data, dtype = "float")
  scalar <- 2
  
  result <- tensor != scalar
  expected <- ifelse(data != scalar, 1, 0)
  
  expect_tensor_equal(result, expected)
})

test_that("inequality operator works with different dtypes", {
  data <- c(1.5, 2.7, 3.1, 2.7)
  
  # Test float32
  tensor_f32 <- as_tensor(data, dtype = "float")
  result_f32 <- tensor_f32 != 2.7
  expected <- ifelse(abs(data - 2.7) >= 1e-6, 1, 0)  # Account for floating point precision
  
  expect_tensor_equal(result_f32, expected)
  
  # Test float64
  tensor_f64 <- as_tensor(data, dtype = "double")
  result_f64 <- tensor_f64 != 2.7
  
  expect_tensor_equal(result_f64, expected)
})

# =============================================================================
# LESS THAN TESTS (<)
# =============================================================================

test_that("less than operator works correctly", {
  # Test basic less than
  a_data <- c(1, 3, 2, 4, 1)
  b_data <- c(2, 3, 1, 5, 2)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor < b_tensor
  expected <- ifelse(a_data < b_data, 1, 0)
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
})

test_that("less than operator works with scalars", {
  data <- c(1, 3, 2, 4, 1)
  tensor <- as_tensor(data, dtype = "float")
  scalar <- 2.5
  
  result <- tensor < scalar
  expected <- ifelse(data < scalar, 1, 0)
  
  expect_tensor_equal(result, expected)
})

# =============================================================================
# LESS THAN OR EQUAL TESTS (<=)
# =============================================================================

test_that("less than or equal operator works correctly", {
  # Test basic less than or equal
  a_data <- c(1, 3, 2, 4, 1)
  b_data <- c(2, 3, 1, 5, 2)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor <= b_tensor
  expected <- ifelse(a_data <= b_data, 1, 0)
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
})

test_that("less than or equal operator works with scalars", {
  data <- c(1, 3, 2, 4, 1)
  tensor <- as_tensor(data, dtype = "float")
  scalar <- 3
  
  result <- tensor <= scalar
  expected <- ifelse(data <= scalar, 1, 0)
  
  expect_tensor_equal(result, expected)
})

test_that("less than or equal operator handles edge cases", {
  # Test with equal values
  equal_data <- c(2, 2, 2, 2)
  equal_tensor <- as_tensor(equal_data, dtype = "float")
  
  result_equal <- equal_tensor <= 2
  expect_tensor_equal(result_equal, c(1, 1, 1, 1))
  
  # Test with mixed values
  mixed_data <- c(1, 2, 3, 2, 1)
  mixed_tensor <- as_tensor(mixed_data, dtype = "float")
  
  result_mixed <- mixed_tensor <= 2
  expected_mixed <- ifelse(mixed_data <= 2, 1, 0)
  expect_tensor_equal(result_mixed, expected_mixed)
})

# =============================================================================
# GREATER THAN TESTS (>)
# =============================================================================

test_that("greater than operator works correctly", {
  # Test basic greater than
  a_data <- c(3, 1, 4, 2, 5)
  b_data <- c(2, 3, 3, 2, 4)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor > b_tensor
  expected <- ifelse(a_data > b_data, 1, 0)
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
})

test_that("greater than operator works with scalars", {
  data <- c(1, 3, 2, 4, 1)
  tensor <- as_tensor(data, dtype = "float")
  scalar <- 2.5
  
  result <- tensor > scalar
  expected <- ifelse(data > scalar, 1, 0)
  
  expect_tensor_equal(result, expected)
})

# =============================================================================
# GREATER THAN OR EQUAL TESTS (>=)
# =============================================================================

test_that("greater than or equal operator works correctly", {
  # Test basic greater than or equal
  a_data <- c(3, 1, 4, 2, 5)
  b_data <- c(2, 3, 3, 2, 4)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor >= b_tensor
  expected <- ifelse(a_data >= b_data, 1, 0)
  
  expect_s3_class(result, "gpuTensor")
  expect_tensor_equal(result, expected)
})

test_that("greater than or equal operator works with scalars", {
  data <- c(1, 3, 2, 4, 1)
  tensor <- as_tensor(data, dtype = "float")
  scalar <- 3
  
  result <- tensor >= scalar
  expected <- ifelse(data >= scalar, 1, 0)
  
  expect_tensor_equal(result, expected)
})

test_that("greater than or equal operator handles edge cases", {
  # Test with equal values
  equal_data <- c(3, 3, 3, 3)
  equal_tensor <- as_tensor(equal_data, dtype = "float")
  
  result_equal <- equal_tensor >= 3
  expect_tensor_equal(result_equal, c(1, 1, 1, 1))
  
  # Test with mixed values
  mixed_data <- c(1, 3, 2, 4, 1)
  mixed_tensor <- as_tensor(mixed_data, dtype = "float")
  
  result_mixed <- mixed_tensor >= 3
  expected_mixed <- ifelse(mixed_data >= 3, 1, 0)
  expect_tensor_equal(result_mixed, expected_mixed)
})

# =============================================================================
# BROADCASTING TESTS
# =============================================================================

test_that("comparison operators work with broadcasting", {
  # Test matrix-vector broadcasting
  mat_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  vec_data <- c(2, 3, 4)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  # Test greater than with broadcasting
  result_gt <- mat_tensor > vec_tensor
  expected_gt <- sweep(mat_data, 2, vec_data, ">")
  expected_gt_numeric <- ifelse(expected_gt, 1, 0)
  
  expect_tensor_equal(result_gt, expected_gt_numeric)
  
  # Test equality with broadcasting
  result_eq <- mat_tensor == vec_tensor
  expected_eq <- sweep(mat_data, 2, vec_data, "==")
  expected_eq_numeric <- ifelse(expected_eq, 1, 0)
  
  expect_tensor_equal(result_eq, expected_eq_numeric)
})

test_that("comparison operators work with scalar broadcasting", {
  # Test 2D tensor with scalar
  mat_data <- matrix(c(1, 4, 2, 5, 3, 6), nrow = 2, ncol = 3)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  scalar <- 3
  
  # Test all comparison operators with scalar
  result_eq <- mat_tensor == scalar
  expected_eq <- ifelse(mat_data == scalar, 1, 0)
  expect_tensor_equal(result_eq, expected_eq)
  
  result_ne <- mat_tensor != scalar
  expected_ne <- ifelse(mat_data != scalar, 1, 0)
  expect_tensor_equal(result_ne, expected_ne)
  
  result_lt <- mat_tensor < scalar
  expected_lt <- ifelse(mat_data < scalar, 1, 0)
  expect_tensor_equal(result_lt, expected_lt)
  
  result_le <- mat_tensor <= scalar
  expected_le <- ifelse(mat_data <= scalar, 1, 0)
  expect_tensor_equal(result_le, expected_le)
  
  result_gt <- mat_tensor > scalar
  expected_gt <- ifelse(mat_data > scalar, 1, 0)
  expect_tensor_equal(result_gt, expected_gt)
  
  result_ge <- mat_tensor >= scalar
  expected_ge <- ifelse(mat_data >= scalar, 1, 0)
  expect_tensor_equal(result_ge, expected_ge)
})

# =============================================================================
# DIFFERENT DTYPES TESTS
# =============================================================================

test_that("comparison operators work with different dtypes", {
  data_a <- c(1.1, 2.2, 3.3, 4.4)
  data_b <- c(1.0, 2.5, 3.3, 4.0)
  
  # Test float32
  a_f32 <- as_tensor(data_a, dtype = "float")
  b_f32 <- as_tensor(data_b, dtype = "float")
  
  result_gt_f32 <- a_f32 > b_f32
  expected_gt <- ifelse(data_a > data_b, 1, 0)
  expect_tensor_equal(result_gt_f32, expected_gt)
  
  result_eq_f32 <- a_f32 == b_f32
  expected_eq <- ifelse(abs(data_a - data_b) < 1e-6, 1, 0)  # Account for float precision
  expect_tensor_equal(result_eq_f32, expected_eq)
  
  # Test float64
  a_f64 <- as_tensor(data_a, dtype = "double")
  b_f64 <- as_tensor(data_b, dtype = "double")
  
  result_gt_f64 <- a_f64 > b_f64
  expect_tensor_equal(result_gt_f64, expected_gt)
  
  result_eq_f64 <- a_f64 == b_f64
  expected_eq_precise <- ifelse(abs(data_a - data_b) < 1e-15, 1, 0)  # Higher precision for double
  expect_tensor_equal(result_eq_f64, expected_eq_precise)
})

# =============================================================================
# NON-CONTIGUOUS TENSOR TESTS
# =============================================================================

test_that("comparison operators work on non-contiguous tensors", {
  # Create non-contiguous tensors via transpose
  mat_a <- matrix(c(1, 4, 2, 5, 3, 6), nrow = 2, ncol = 3)
  mat_b <- matrix(c(2, 3, 4, 5, 6, 7), nrow = 2, ncol = 3)
  
  tensor_a <- as_tensor(mat_a, dtype = "float")
  tensor_b <- as_tensor(mat_b, dtype = "float")
  
  # Create transpose views
  transposed_a <- transpose(tensor_a)
  transposed_b <- transpose(tensor_b)
  
  # Test comparison on transposed tensors
  result_gt <- transposed_a > transposed_b
  expected_gt <- ifelse(t(mat_a) > t(mat_b), 1, 0)
  expect_tensor_equal(result_gt, expected_gt)
  
  result_le <- transposed_a <= transposed_b
  expected_le <- ifelse(t(mat_a) <= t(mat_b), 1, 0)
  expect_tensor_equal(result_le, expected_le)
})

# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

test_that("comparison operators handle edge cases", {
  # Single element tensors
  single_a <- as_tensor(c(5), dtype = "float")
  single_b <- as_tensor(c(3), dtype = "float")
  
  result_single <- single_a > single_b
  expect_tensor_equal(result_single, c(1))
  
  # Zero values
  zero_data <- c(0, 1, -1, 0)
  zero_tensor <- as_tensor(zero_data, dtype = "float")
  
  result_zero_eq <- zero_tensor == 0
  expected_zero_eq <- ifelse(zero_data == 0, 1, 0)
  expect_tensor_equal(result_zero_eq, expected_zero_eq)
  
  result_zero_gt <- zero_tensor > 0
  expected_zero_gt <- ifelse(zero_data > 0, 1, 0)
  expect_tensor_equal(result_zero_gt, expected_zero_gt)
  
  # Negative values
  neg_data <- c(-3, -1, 0, 1, 3)
  neg_tensor <- as_tensor(neg_data, dtype = "float")
  
  result_neg_lt <- neg_tensor < 0
  expected_neg_lt <- ifelse(neg_data < 0, 1, 0)
  expect_tensor_equal(result_neg_lt, expected_neg_lt)
})

test_that("comparison operators handle floating point precision", {
  # Test with values that might have precision issues
  data_precise <- c(0.1 + 0.2, 0.3, 1.0/3.0, 0.3333333)
  tensor_precise <- as_tensor(data_precise, dtype = "float")
  
  # Test equality with expected precision issues
  result_eq_03 <- tensor_precise == 0.3
  # First element (0.1 + 0.2) might not exactly equal 0.3 due to floating point
  
  # Test with tolerance-based comparison (using >= and <=)
  tolerance <- 1e-6
  approx_equal <- (tensor_precise >= (0.3 - tolerance)) * (tensor_precise <= (0.3 + tolerance))
  
  # Should handle precision appropriately
  expect_s3_class(result_eq_03, "gpuTensor")
  expect_s3_class(approx_equal, "gpuTensor")
})

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test_that("comparison operators can be chained and combined", {
  data <- c(1, 2, 3, 4, 5)
  tensor <- as_tensor(data, dtype = "float")
  
  # Test chained comparisons: 2 <= x <= 4
  ge_2 <- tensor >= 2
  le_4 <- tensor <= 4
  in_range <- ge_2 * le_4  # Element-wise multiplication for AND operation
  
  expected_in_range <- ifelse((data >= 2) & (data <= 4), 1, 0)
  expect_tensor_equal(in_range, expected_in_range)
  
  # Test combined with logical operations
  gt_2 <- tensor > 2
  lt_4 <- tensor < 4
  strict_range <- gt_2 * lt_4
  
  expected_strict <- ifelse((data > 2) & (data < 4), 1, 0)
  expect_tensor_equal(strict_range, expected_strict)
})

test_that("comparison results can be used for indexing and masking", {
  data <- c(1, 5, 2, 8, 3, 6, 4, 7)
  tensor <- as_tensor(data, dtype = "float")
  
  # Create mask for values > 4
  mask_gt4 <- tensor > 4
  expected_mask <- ifelse(data > 4, 1, 0)
  expect_tensor_equal(mask_gt4, expected_mask)
  
  # Verify mask properties
  expect_equal(shape(mask_gt4), shape(tensor))
  expect_equal(dtype(mask_gt4), dtype(tensor))  # Should maintain same dtype
  
  # Test that mask values are 0 or 1
  mask_values <- as.vector(mask_gt4)
  expect_true(all(mask_values %in% c(0, 1)))
})

test_that("comparison operators maintain GPU execution", {
  # Test with larger tensors to ensure GPU path
  n <- 10000
  data_large <- stats::runif(n, -10, 10)
  tensor_large <- as_tensor(data_large, dtype = "float")
  
  # Test various comparison operations
  result_gt <- tensor_large > 0
  result_eq <- tensor_large == 0  # Unlikely to be true for random data
  result_ne <- tensor_large != 0
  result_le <- tensor_large <= 5
  result_ge <- tensor_large >= -5
  
  # All should be GPU tensors
  expect_s3_class(result_gt, "gpuTensor")
  expect_s3_class(result_eq, "gpuTensor")
  expect_s3_class(result_ne, "gpuTensor")
  expect_s3_class(result_le, "gpuTensor")
  expect_s3_class(result_ge, "gpuTensor")
  
  # Verify correctness for a subset
  subset_idx <- 1:100
  expect_equal(as.vector(result_gt)[subset_idx], 
               ifelse(data_large[subset_idx] > 0, 1, 0), 
               tolerance = 1e-6)
  expect_equal(as.vector(result_le)[subset_idx], 
               ifelse(data_large[subset_idx] <= 5, 1, 0), 
               tolerance = 1e-6)
}) 