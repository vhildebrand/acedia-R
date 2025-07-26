context("Comprehensive broadcasting tests for all operators")

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
# ARITHMETIC OPERATORS BROADCASTING
# =============================================================================

test_that("addition broadcasting works correctly", {
  # Matrix + vector (broadcast along columns)
  mat_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  vec_data <- c(10, 20, 30)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  result <- mat_tensor + vec_tensor
  expected <- sweep(mat_data, 2, vec_data, "+")
  
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), c(2, 3))
  
  # Vector + matrix (should be commutative)
  result_comm <- vec_tensor + mat_tensor
  expect_tensor_equal(result_comm, expected)
  
  # 3D tensor + matrix broadcasting
  arr_data <- array(1:24, dim = c(2, 3, 4))
  mat_2d_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  
  arr_tensor <- as_tensor(arr_data, dtype = "float")
  mat_2d_tensor <- as_tensor(mat_2d_data, dtype = "float")
  
  result_3d <- arr_tensor + mat_2d_tensor
  expected_3d <- sweep(arr_data, c(1, 2), mat_2d_data, "+")
  
  expect_tensor_equal(result_3d, expected_3d)
  expect_equal(shape(result_3d), c(2, 3, 4))
})

test_that("subtraction broadcasting works correctly", {
  # Matrix - vector
  mat_data <- matrix(c(10, 20, 30, 40, 50, 60), nrow = 2, ncol = 3)
  vec_data <- c(1, 2, 3)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  result <- mat_tensor - vec_tensor
  expected <- sweep(mat_data, 2, vec_data, "-")
  
  expect_tensor_equal(result, expected)
  
  # Vector - matrix (not commutative)
  result_rev <- vec_tensor - mat_tensor
  expected_rev <- sweep(-mat_data, 2, vec_data, "+")
  
  expect_tensor_equal(result_rev, expected_rev)
})

test_that("multiplication broadcasting works correctly", {
  # Matrix * vector
  mat_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  vec_data <- c(2, 3, 4)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  result <- mat_tensor * vec_tensor
  expected <- sweep(mat_data, 2, vec_data, "*")
  
  expect_tensor_equal(result, expected)
  
  # Test with different shapes
  # Row vector * column vector should produce matrix
  row_vec_data <- c(1, 2, 3)
  col_vec_data <- c(10, 20)
  
  row_tensor <- as_tensor(row_vec_data, dtype = "float")
  col_tensor <- as_tensor(col_vec_data, dtype = "float")
  
  # Reshape to explicit row/column vectors
  row_matrix <- view(row_tensor, c(1, 3))
  col_matrix <- view(col_tensor, c(2, 1))
  
  result_outer <- row_matrix * col_matrix
  expected_outer <- outer(col_vec_data, row_vec_data)
  
  expect_tensor_equal(result_outer, expected_outer)
  expect_equal(shape(result_outer), c(2, 3))
})

test_that("division broadcasting works correctly", {
  # Matrix / vector
  mat_data <- matrix(c(10, 20, 30, 40, 50, 60), nrow = 2, ncol = 3)
  vec_data <- c(2, 4, 5)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  result <- mat_tensor / vec_tensor
  expected <- sweep(mat_data, 2, vec_data, "/")
  
  expect_tensor_equal(result, expected)
  
  # Vector / matrix
  result_rev <- vec_tensor / mat_tensor
  expected_rev <- sweep(mat_data, 2, vec_data, function(x, y) y / x)
  
  expect_tensor_equal(result_rev, expected_rev)
})

# =============================================================================
# COMPARISON OPERATORS BROADCASTING
# =============================================================================

test_that("comparison operators broadcasting works correctly", {
  # Matrix > vector
  mat_data <- matrix(c(1, 5, 2, 6, 3, 7), nrow = 2, ncol = 3)
  vec_data <- c(3, 4, 5)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  # Greater than
  result_gt <- mat_tensor > vec_tensor
  expected_gt <- sweep(mat_data, 2, vec_data, ">")
  expected_gt_numeric <- ifelse(expected_gt, 1, 0)
  expect_tensor_equal(result_gt, expected_gt_numeric)
  
  # Less than
  result_lt <- mat_tensor < vec_tensor
  expected_lt <- sweep(mat_data, 2, vec_data, "<")
  expected_lt_numeric <- ifelse(expected_lt, 1, 0)
  expect_tensor_equal(result_lt, expected_lt_numeric)
  
  # Equal
  result_eq <- mat_tensor == vec_tensor
  expected_eq <- sweep(mat_data, 2, vec_data, "==")
  expected_eq_numeric <- ifelse(expected_eq, 1, 0)
  expect_tensor_equal(result_eq, expected_eq_numeric)
  
  # Not equal
  result_ne <- mat_tensor != vec_tensor
  expected_ne <- sweep(mat_data, 2, vec_data, "!=")
  expected_ne_numeric <- ifelse(expected_ne, 1, 0)
  expect_tensor_equal(result_ne, expected_ne_numeric)
  
  # Greater than or equal
  result_ge <- mat_tensor >= vec_tensor
  expected_ge <- sweep(mat_data, 2, vec_data, ">=")
  expected_ge_numeric <- ifelse(expected_ge, 1, 0)
  expect_tensor_equal(result_ge, expected_ge_numeric)
  
  # Less than or equal
  result_le <- mat_tensor <= vec_tensor
  expected_le <- sweep(mat_data, 2, vec_data, "<=")
  expected_le_numeric <- ifelse(expected_le, 1, 0)
  expect_tensor_equal(result_le, expected_le_numeric)
})

# =============================================================================
# ELEMENT-WISE BINARY FUNCTIONS BROADCASTING
# =============================================================================

test_that("pmax/pmin broadcasting works correctly", {
  # Matrix pmax vector
  mat_data <- matrix(c(1, 5, 2, 6, 3, 7), nrow = 2, ncol = 3)
  vec_data <- c(3, 4, 5)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  # Element-wise maximum
  result_pmax <- pmax(mat_tensor, vec_tensor)
  expected_pmax <- sweep(mat_data, 2, vec_data, pmax)
  expect_tensor_equal(result_pmax, expected_pmax)
  
  # Element-wise minimum
  result_pmin <- pmin(mat_tensor, vec_tensor)
  expected_pmin <- sweep(mat_data, 2, vec_data, pmin)
  expect_tensor_equal(result_pmin, expected_pmin)
})

test_that("tensor_pow broadcasting works correctly", {
  # Matrix ^ vector (element-wise power)
  base_data <- matrix(c(2, 3, 2, 3, 2, 3), nrow = 2, ncol = 3)
  exp_data <- c(1, 2, 3)
  
  base_tensor <- as_tensor(base_data, dtype = "float")
  exp_tensor <- as_tensor(exp_data, dtype = "float")
  
  result_pow <- tensor_pow(base_tensor, exp_tensor)
  expected_pow <- sweep(base_data, 2, exp_data, "^")
  expect_tensor_equal(result_pow, expected_pow)
})

# =============================================================================
# SCALAR BROADCASTING TESTS
# =============================================================================

test_that("scalar broadcasting works for all operators", {
  # Test with 3D tensor
  arr_data <- array(c(1, 2, 3, 4, 5, 6, 7, 8), dim = c(2, 2, 2))
  arr_tensor <- as_tensor(arr_data, dtype = "float")
  scalar <- 3.0
  
  # Arithmetic operations
  result_add <- arr_tensor + scalar
  expected_add <- arr_data + scalar
  expect_tensor_equal(result_add, expected_add)
  
  result_sub <- arr_tensor - scalar
  expected_sub <- arr_data - scalar
  expect_tensor_equal(result_sub, expected_sub)
  
  result_mul <- arr_tensor * scalar
  expected_mul <- arr_data * scalar
  expect_tensor_equal(result_mul, expected_mul)
  
  result_div <- arr_tensor / scalar
  expected_div <- arr_data / scalar
  expect_tensor_equal(result_div, expected_div)
  
  # Comparison operations
  result_gt <- arr_tensor > scalar
  expected_gt <- ifelse(arr_data > scalar, 1, 0)
  expect_tensor_equal(result_gt, expected_gt)
  
  result_lt <- arr_tensor < scalar
  expected_lt <- ifelse(arr_data < scalar, 1, 0)
  expect_tensor_equal(result_lt, expected_lt)
  
  result_eq <- arr_tensor == scalar
  expected_eq <- ifelse(arr_data == scalar, 1, 0)
  expect_tensor_equal(result_eq, expected_eq)
  
  # Power operation
  result_pow <- arr_tensor ^ 2
  expected_pow <- arr_data ^ 2
  expect_tensor_equal(result_pow, expected_pow)
})

# =============================================================================
# COMPLEX BROADCASTING SCENARIOS
# =============================================================================

test_that("multi-dimensional broadcasting works correctly", {
  # Test (2, 1, 3) + (1, 4, 1) -> (2, 4, 3)
  tensor_a_data <- array(c(1, 2, 3, 4, 5, 6), dim = c(2, 1, 3))
  tensor_b_data <- array(c(10, 20, 30, 40), dim = c(1, 4, 1))
  
  tensor_a <- as_tensor(tensor_a_data, dtype = "float")
  tensor_b <- as_tensor(tensor_b_data, dtype = "float")
  
  result <- tensor_a + tensor_b
  
  # Manual computation of expected result
  expected <- array(0, dim = c(2, 4, 3))
  for (i in 1:2) {
    for (j in 1:4) {
      for (k in 1:3) {
        expected[i, j, k] <- tensor_a_data[i, 1, k] + tensor_b_data[1, j, 1]
      }
    }
  }
  
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), c(2, 4, 3))
})

test_that("broadcasting with singleton dimensions works correctly", {
  # Test (3, 1, 2) * (1, 4, 1) -> (3, 4, 2)
  a_data <- array(c(1, 2, 3, 4, 5, 6), dim = c(3, 1, 2))
  b_data <- array(c(2, 3, 4, 5), dim = c(1, 4, 1))
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor * b_tensor
  
  # Verify shape
  expect_equal(shape(result), c(3, 4, 2))
  
  # Spot check a few values
  result_array <- as.array(result)
  expect_equal(result_array[1, 1, 1], a_data[1, 1, 1] * b_data[1, 1, 1], tolerance = 1e-6)
  expect_equal(result_array[2, 3, 2], a_data[2, 1, 2] * b_data[1, 3, 1], tolerance = 1e-6)
})

test_that("broadcasting with leading singleton dimensions", {
  # Test (1, 3) + (2, 3) -> (2, 3)
  a_data <- matrix(c(1, 2, 3), nrow = 1, ncol = 3)
  b_data <- matrix(c(10, 20, 30, 40, 50, 60), nrow = 2, ncol = 3)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor + b_tensor
  expected <- sweep(b_data, 2, a_data[1, ], "+")
  
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), c(2, 3))
})

test_that("broadcasting with trailing singleton dimensions", {
  # Test (3, 1) + (3, 4) -> (3, 4)
  a_data <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
  b_data <- matrix(c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120), nrow = 3, ncol = 4)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  result <- a_tensor + b_tensor
  expected <- sweep(b_data, 1, a_data[, 1], "+")
  
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), c(3, 4))
})

# =============================================================================
# BROADCASTING WITH DIFFERENT DTYPES
# =============================================================================

test_that("broadcasting maintains dtype consistency", {
  # Test that broadcasting preserves dtypes or handles dtype mismatches appropriately
  mat_f32 <- as_tensor(matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2), dtype = "float")
  vec_f32 <- as_tensor(c(10, 20), dtype = "float")
  
  result_f32 <- mat_f32 + vec_f32
  expect_equal(dtype(result_f32), "float")
  
  mat_f64 <- as_tensor(matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2), dtype = "double")
  vec_f64 <- as_tensor(c(10, 20), dtype = "double")
  
  result_f64 <- mat_f64 + vec_f64
  expect_equal(dtype(result_f64), "double")
  
  # Mixed dtypes should error or convert (implementation dependent)
  tryCatch({
    mixed_result <- mat_f32 + vec_f64
    # If this succeeds, check the result dtype
    expect_s3_class(mixed_result, "gpuTensor")
  }, error = function(e) {
    # If this errors due to dtype mismatch, that's acceptable
    expect_true(grepl("dtype", e$message, ignore.case = TRUE))
  })
})

# =============================================================================
# BROADCASTING WITH NON-CONTIGUOUS TENSORS
# =============================================================================

test_that("broadcasting works with non-contiguous tensors", {
  # Create non-contiguous tensors via transpose and test broadcasting
  mat_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  vec_data <- c(10, 20)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  # Create transpose view (non-contiguous)
  transposed_mat <- transpose(mat_tensor)  # Now 3x2
  
  # Broadcasting with non-contiguous tensor
  result <- transposed_mat + vec_tensor  # (3,2) + (2,) -> (3,2)
  expected <- sweep(t(mat_data), 2, vec_data, "+")
  
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), c(3, 2))
})

# =============================================================================
# ERROR HANDLING FOR BROADCASTING
# =============================================================================

test_that("broadcasting error handling works correctly", {
  # Incompatible shapes that cannot be broadcast
  tensor_a <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  tensor_b <- as_tensor(c(1, 2), dtype = "float")  # Length 2, can't broadcast to (2,3)
  
  # This should work: (2,3) + (2,) broadcasts along rows
  result_valid <- tensor_a + tensor_b
  expect_equal(shape(result_valid), c(2, 3))
  
  # This should fail: incompatible shapes
  tensor_c <- as_tensor(matrix(1:8, nrow = 2, ncol = 4), dtype = "float")  # (2,4)
  expect_error(tensor_a + tensor_c, "shape|dimension|broadcast")
  
  # Test with completely incompatible shapes
  tensor_3d <- as_tensor(array(1:12, dim = c(2, 2, 3)), dtype = "float")
  tensor_2d_bad <- as_tensor(matrix(1:8, nrow = 4, ncol = 2), dtype = "float")
  
  expect_error(tensor_3d + tensor_2d_bad, "shape|dimension|broadcast")
})

# =============================================================================
# PERFORMANCE TESTS FOR BROADCASTING
# =============================================================================

test_that("broadcasting maintains GPU execution with large tensors", {
  # Test broadcasting with larger tensors to ensure GPU execution
  large_mat <- as_tensor(matrix(stats::runif(10000), nrow = 100, ncol = 100), dtype = "float")
  vec_broadcast <- as_tensor(stats::runif(100), dtype = "float")
  
  # Time broadcasting operation
  broadcast_time <- system.time({
    result <- large_mat + vec_broadcast
  })
  
  # Should complete quickly on GPU
  expect_lt(broadcast_time[["elapsed"]], 2.0)
  
  # Verify result properties
  expect_s3_class(result, "gpuTensor")
  expect_equal(shape(result), c(100, 100))
  
  # Spot check correctness
  large_mat_subset <- as.array(large_mat)[1:5, 1:5]
  vec_subset <- as.vector(vec_broadcast)[1:5]
  result_subset <- as.array(result)[1:5, 1:5]
  # Use R's native broadcasting behavior (row-wise), not sweep column-wise
  expected_subset <- large_mat_subset + vec_subset
  
  expect_equal(result_subset, expected_subset, tolerance = 1e-6)
})

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test_that("broadcasting works in complex expression chains", {
  # Test chained operations with broadcasting
  a <- as_tensor(matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2), dtype = "float")
  b <- as_tensor(c(10, 20), dtype = "float")
  c <- as_tensor(c(100, 200), dtype = "float")
  
  # Chain: (a + b) * c
  result <- (a + b) * c
  
  # Compute expected step by step using R's native broadcasting (row-wise)
  step1 <- as.array(a) + as.vector(b)  # a + b
  step2 <- step1 * as.vector(c)        # result * c
  
  expect_tensor_equal(result, step2)
  expect_equal(shape(result), c(2, 2))
})

test_that("broadcasting works with reductions", {
  # Test broadcasting followed by reductions
  mat_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  vec_data <- c(10, 20, 30)
  
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  # Broadcast then reduce
  broadcast_result <- mat_tensor + vec_tensor
  sum_result <- sum(broadcast_result)
  
  # Compute expected
  expected_broadcast <- sweep(mat_data, 2, vec_data, "+")
  expected_sum <- sum(expected_broadcast)
  
  expect_equal(as.numeric(sum_result), expected_sum, tolerance = 1e-6)
})

test_that("broadcasting works with views and reshapes", {
  # Test broadcasting with tensor views
  original_data <- array(1:12, dim = c(3, 4))
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Create view
  reshaped_view <- view(original_tensor, c(2, 6))
  vec_for_broadcast <- as_tensor(c(1, 2, 3, 4, 5, 6), dtype = "float")
  
  # Broadcasting with view
  result <- reshaped_view + vec_for_broadcast
  
  # Compute expected
  reshaped_data <- array(as.vector(original_data), dim = c(2, 6))
  expected <- sweep(reshaped_data, 2, as.vector(vec_for_broadcast), "+")
  
  expect_tensor_equal(result, expected)
  expect_equal(shape(result), c(2, 6))
}) 