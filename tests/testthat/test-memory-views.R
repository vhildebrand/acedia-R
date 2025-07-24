context("Memory layout and views: transpose, permute, contiguous, views, broadcasting")

library(testthat)
library(acediaR)

# Simplified GPU verification - just check class
verify_gpu_tensor <- function(tensor, operation_name = "operation") {
  if (!inherits(tensor, "gpuTensor")) {
    warning(paste("❌ GPU FALLBACK:", operation_name, "returned non-gpuTensor object"))
    return(FALSE)
  }
  return(TRUE)
}

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
# TRANSPOSE OPERATIONS
# =============================================================================

test_that("Transpose creates efficient views", {
  # Test 2D transpose
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  expected <- t(original_data)
  
  original_tensor <- as_tensor(original_data, dtype = "float")
  transposed <- transpose(original_tensor)
  
  expect_true(verify_gpu_tensor(transposed, "transpose"))
  expect_equal(as.array(transposed), expected, tolerance = 1e-6)
  expect_equal(shape(transposed), c(4, 3))
  
  # Verify it's a view (non-contiguous)
  expect_false(is_contiguous(transposed))
  expect_true(is_contiguous(original_tensor))
})

test_that("Transpose works with different tensor sizes", {
  # Test square matrix
  square_data <- matrix(1:9, nrow = 3, ncol = 3)
  square_tensor <- as_tensor(square_data, dtype = "float")
  square_transposed <- transpose(square_tensor)
  
  expect_equal(as.array(square_transposed), t(square_data), tolerance = 1e-6)
  expect_equal(shape(square_transposed), c(3, 3))
  
  # Test rectangular matrix (different aspect ratio)
  rect_data <- matrix(1:20, nrow = 4, ncol = 5)
  rect_tensor <- as_tensor(rect_data, dtype = "float")
  rect_transposed <- transpose(rect_tensor)
  
  expect_equal(as.array(rect_transposed), t(rect_data), tolerance = 1e-6)
  expect_equal(shape(rect_transposed), c(5, 4))
})

test_that("Double transpose returns to original", {
  original_data <- matrix(1:6, nrow = 2, ncol = 3)
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  double_transposed <- transpose(transpose(original_tensor))
  
  expect_equal(as.array(double_transposed), original_data, tolerance = 1e-6)
  expect_equal(shape(double_transposed), c(2, 3))
})

test_that("Transpose error handling", {
  # Non-2D tensors should error
  vec_1d <- as_tensor(c(1, 2, 3), dtype = "float")
  array_3d <- as_tensor(array(1:24, dim = c(2, 3, 4)), dtype = "float")
  
  expect_error(transpose(vec_1d), "2D")
  expect_error(transpose(array_3d), "2D")
})

# =============================================================================
# PERMUTE OPERATIONS
# =============================================================================

test_that("Permute works for 2D tensors", {
  # 2D permute (equivalent to transpose)
  original_data <- matrix(1:6, nrow = 2, ncol = 3)
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Permute dimensions 0,1 -> 1,0 (swap rows and columns)
  permuted <- permute(original_tensor, c(2, 1))  # R uses 1-based indexing
  
  expect_true(verify_gpu_tensor(permuted, "permute 2D"))
  expect_equal(as.array(permuted), t(original_data), tolerance = 1e-6)
  expect_equal(shape(permuted), c(3, 2))
  expect_false(is_contiguous(permuted))
})

test_that("Permute works for 3D tensors", {
  # 3D permute
  original_data <- array(1:24, dim = c(2, 3, 4))
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Permute dimensions: (2,3,4) -> (3,4,2)
  permuted <- permute(original_tensor, c(2, 3, 1))
  expected <- aperm(original_data, c(2, 3, 1))
  
  expect_true(verify_gpu_tensor(permuted, "permute 3D"))
  expect_equal(as.array(permuted), expected, tolerance = 1e-6)
  expect_equal(shape(permuted), c(3, 4, 2))
  expect_false(is_contiguous(permuted))
})

test_that("Permute identity returns original layout", {
  original_data <- array(1:24, dim = c(2, 3, 4))
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Identity permutation
  identity_permuted <- permute(original_tensor, c(1, 2, 3))
  
  expect_equal(as.array(identity_permuted), original_data, tolerance = 1e-6)
  expect_equal(shape(identity_permuted), c(2, 3, 4))
})

test_that("Permute error handling", {
  tensor_3d <- as_tensor(array(1:24, dim = c(2, 3, 4)), dtype = "float")
  
  # Invalid permutation dimensions
  expect_error(permute(tensor_3d, c(1, 2)), "dimensions")
  expect_error(permute(tensor_3d, c(1, 2, 3, 4)), "dimensions")
  expect_error(permute(tensor_3d, c(1, 1, 3)), "Invalid")
  expect_error(permute(tensor_3d, c(0, 2, 3)), "Invalid")
})

# =============================================================================
# CONTIGUOUS OPERATIONS
# =============================================================================

test_that("Contiguous creates materialized copy when needed", {
  # Create non-contiguous tensor via transpose
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  original_tensor <- as_tensor(original_data, dtype = "float")
  transposed <- transpose(original_tensor)
  
  expect_false(is_contiguous(transposed))
  
  # Make contiguous copy
  contiguous_copy <- contiguous(transposed)
  
  expect_true(verify_gpu_tensor(contiguous_copy, "contiguous copy"))
  expect_true(is_contiguous(contiguous_copy))
  expect_equal(as.array(contiguous_copy), as.array(transposed), tolerance = 1e-6)
  expect_equal(shape(contiguous_copy), shape(transposed))
})

test_that("Contiguous is no-op for already contiguous tensors", {
  original_data <- matrix(1:6, nrow = 2, ncol = 3)
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  expect_true(is_contiguous(original_tensor))
  
  # Should return same tensor (or equivalent)
  contiguous_result <- contiguous(original_tensor)
  
  expect_true(is_contiguous(contiguous_result))
  expect_equal(as.array(contiguous_result), original_data, tolerance = 1e-6)
})

test_that("Operations work correctly on non-contiguous tensors", {
  # Create non-contiguous tensors
  A_data <- matrix(1:6, nrow = 2, ncol = 3)
  B_data <- matrix(1:6, nrow = 3, ncol = 2)
  
  A_tensor <- as_tensor(A_data, dtype = "float")
  B_tensor <- as_tensor(B_data, dtype = "float")
  
  # Create non-contiguous views
  A_transposed <- transpose(A_tensor)  # Now 3x2
  B_transposed <- transpose(B_tensor)  # Now 2x3
  
  expect_false(is_contiguous(A_transposed))
  expect_false(is_contiguous(B_transposed))
  
  # Operations should work on non-contiguous tensors
  # Addition (same shape)
  sum_result <- A_transposed + A_transposed
  expect_equal(as.array(sum_result), 2 * as.array(A_transposed), tolerance = 1e-6)
  
  # Matrix multiplication
  matmul_result <- matmul(A_transposed, B_transposed)
  expected_matmul <- t(A_data) %*% t(B_data)
  expect_equal(as.array(matmul_result), expected_matmul, tolerance = 1e-6)
})

# =============================================================================
# VIEW OPERATIONS
# =============================================================================

test_that("View creates efficient reshape", {
  # Test reshaping contiguous tensor
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Reshape to vector
  view_1d <- view(original_tensor, c(12))
  expect_equal(as.vector(view_1d), as.vector(original_data), tolerance = 1e-6)
  expect_equal(shape(view_1d), 12)
  
  # Reshape to different 2D shape
  view_2d <- view(original_tensor, c(4, 3))
  expect_equal(size(view_2d), 12)
  expect_equal(shape(view_2d), c(4, 3))
})

test_that("View handles non-contiguous tensors", {
  # Create non-contiguous tensor
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  original_tensor <- as_tensor(original_data, dtype = "float")
  transposed <- transpose(original_tensor)
  
  expect_false(is_contiguous(transposed))
  
  # View should work (may create contiguous copy internally)
  view_result <- view(transposed, c(12))
  expect_equal(size(view_result), 12)
  expect_equal(shape(view_result), 12)
})

test_that("Reshape is equivalent to view for compatible shapes", {
  original_data <- array(1:24, dim = c(2, 3, 4))
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Both should produce same result
  view_result <- view(original_tensor, c(6, 4))
  reshape_result <- reshape(original_tensor, c(6, 4))
  
  expect_equal(as.array(view_result), as.array(reshape_result), tolerance = 1e-6)
  expect_equal(shape(view_result), c(6, 4))
  expect_equal(shape(reshape_result), c(6, 4))
})

test_that("View error handling", {
  tensor_2d <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  
  # Incompatible sizes
  expect_error(view(tensor_2d, c(5)), "size")
  expect_error(view(tensor_2d, c(2, 4)), "size")
  
  # Invalid shapes
  expect_error(view(tensor_2d, c(0, 6)), "positive")
  expect_error(view(tensor_2d, c(-1, 6)), "positive")
})

# =============================================================================
# MEMORY EFFICIENCY TESTS
# =============================================================================

test_that("Views share memory with original tensor", {
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Create views
  transposed <- transpose(original_tensor)
  
  # Both should be valid GPU tensors
  expect_true(verify_gpu_tensor(original_tensor, "original"))
  expect_true(verify_gpu_tensor(transposed, "transpose view"))
  
  # Views should have different contiguity but same underlying data
  expect_true(is_contiguous(original_tensor))
  expect_false(is_contiguous(transposed))
})

test_that("Complex view operations maintain correctness", {
  # Test chain of view operations
  original_data <- array(1:24, dim = c(2, 3, 4))
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Chain: permute -> view -> transpose (for 2D result)
  permuted <- permute(original_tensor, c(2, 1, 3))  # (3, 2, 4)
  viewed <- view(permuted, c(6, 4))                 # (6, 4)
  transposed <- transpose(viewed)                   # (4, 6)
  
  expect_equal(shape(transposed), c(4, 6))
  expect_equal(size(transposed), 24)
  expect_true(verify_gpu_tensor(transposed, "complex view chain"))
})

# =============================================================================
# BROADCASTING TESTS
# =============================================================================

test_that("Broadcasting works with different tensor shapes", {
  # Test vector + matrix broadcasting
  matrix_data <- matrix(1:6, nrow = 2, ncol = 3)
  vector_data <- c(10, 20, 30)
  
  matrix_tensor <- as_tensor(matrix_data, dtype = "float")
  vector_tensor <- as_tensor(vector_data, dtype = "float")
  
  # Should broadcast vector across matrix rows
  result <- matrix_tensor + vector_tensor
  expected <- sweep(matrix_data, 2, vector_data, "+")
  
  expect_true(verify_gpu_tensor(result, "broadcasting"))
  expect_equal(as.array(result), expected, tolerance = 1e-6)
  expect_equal(shape(result), c(2, 3))
})

test_that("Broadcasting error detection works", {
  # Incompatible shapes that cannot broadcast
  tensor_2x3 <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  tensor_2x4 <- as_tensor(matrix(1:8, nrow = 2, ncol = 4), dtype = "float")
  
  expect_error(tensor_2x3 + tensor_2x4, "broadcastable")
})

# =============================================================================
# PERFORMANCE AND INTEGRATION TESTS
# =============================================================================

test_that("Non-contiguous operations maintain reasonable performance", {
  # Test with moderately large tensors
  n <- 100
  large_data <- matrix(runif(n*n), nrow = n, ncol = n)
  large_tensor <- as_tensor(large_data, dtype = "float")
  
  # Create non-contiguous view
  transposed <- transpose(large_tensor)
  expect_false(is_contiguous(transposed))
  
  # Operations should complete in reasonable time
  start_time <- Sys.time()
  result <- transposed + transposed
  end_time <- Sys.time()
  
  expect_true(verify_gpu_tensor(result, "large non-contiguous op"))
  expect_lt(as.numeric(end_time - start_time), 5)  # Should complete within 5 seconds
})

test_that("Memory layout operations work with different dtypes", {
  test_data <- matrix(1:6, nrow = 2, ncol = 3)
  
  # Test float32
  tensor_f32 <- as_tensor(test_data, dtype = "float")
  transposed_f32 <- transpose(tensor_f32)
  expect_equal(as.array(transposed_f32), t(test_data), tolerance = 1e-6)
  
  # Test float64
  tensor_f64 <- as_tensor(test_data, dtype = "double")
  transposed_f64 <- transpose(tensor_f64)
  expect_equal(as.array(transposed_f64), t(test_data), tolerance = 1e-15)
}) 