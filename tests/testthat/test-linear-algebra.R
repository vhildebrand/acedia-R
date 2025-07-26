context("Linear algebra operations: matmul, outer product, matvec, vecmat")

## loaded in setup.R

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
# MATRIX MULTIPLICATION
# =============================================================================

test_that("Matrix multiplication works correctly", {
  # Test 2x3 * 3x2 = 2x2
  A_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  B_data <- matrix(c(7, 8, 9, 10, 11, 12), nrow = 3, ncol = 2)
  expected <- A_data %*% B_data
  
  # Test float32
  A_f32 <- as_tensor(A_data, dtype = "float")
  B_f32 <- as_tensor(B_data, dtype = "float")
  result_f32 <- matmul(A_f32, B_f32)
  
  expect_true(verify_gpu_tensor(result_f32, "matmul float32"))
  expect_equal(as.array(result_f32), expected, tolerance = 1e-6)
  expect_equal(shape(result_f32), c(2, 2))
  
  # Test float64
  A_f64 <- as_tensor(A_data, dtype = "double")
  B_f64 <- as_tensor(B_data, dtype = "double")
  result_f64 <- matmul(A_f64, B_f64)
  
  expect_true(verify_gpu_tensor(result_f64, "matmul float64"))
  expect_equal(as.array(result_f64), expected, tolerance = 1e-15)
})

test_that("Matrix multiplication handles different sizes", {
  # Test square matrices
  A_square <- matrix(1:4, nrow = 2, ncol = 2)
  B_square <- matrix(5:8, nrow = 2, ncol = 2)
  expected_square <- A_square %*% B_square
  
  A_tensor <- as_tensor(A_square, dtype = "float")
  B_tensor <- as_tensor(B_square, dtype = "float")
  result <- matmul(A_tensor, B_tensor)
  
  expect_equal(as.array(result), expected_square, tolerance = 1e-6)
  expect_equal(shape(result), c(2, 2))
  
  # Test rectangular matrices
  A_rect <- matrix(1:6, nrow = 2, ncol = 3)
  B_rect <- matrix(1:12, nrow = 3, ncol = 4)
  expected_rect <- A_rect %*% B_rect
  
  A_rect_tensor <- as_tensor(A_rect, dtype = "float")
  B_rect_tensor <- as_tensor(B_rect, dtype = "float")
  result_rect <- matmul(A_rect_tensor, B_rect_tensor)
  
  expect_equal(as.array(result_rect), expected_rect, tolerance = 1e-6)
  expect_equal(shape(result_rect), c(2, 4))
})

test_that("Matrix multiplication error handling", {
  # Incompatible dimensions
  A_bad <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  B_bad <- as_tensor(matrix(1:8, nrow = 2, ncol = 4), dtype = "float")  # Wrong inner dim
  
  expect_error(matmul(A_bad, B_bad), "Incompatible|dimensions")
  
  # Non-2D tensors
  vec_1d <- as_tensor(c(1, 2, 3), dtype = "float")
  mat_2d <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  
  expect_error(matmul(vec_1d, mat_2d), "2D|matrix")
  expect_error(matmul(mat_2d, vec_1d), "2D|matrix")
})

# =============================================================================
# OUTER PRODUCT
# =============================================================================

test_that("Outer product works correctly", {
  # Test basic outer product
  a_data <- c(1, 2, 3)
  b_data <- c(4, 5)
  expected <- outer(a_data, b_data)
  
  # Test float32
  a_f32 <- as_tensor(a_data, dtype = "float")
  b_f32 <- as_tensor(b_data, dtype = "float")
  result_f32 <- outer_product(a_f32, b_f32)
  
  expect_true(verify_gpu_tensor(result_f32, "outer product float32"))
  expect_equal(as.array(result_f32), expected, tolerance = 1e-6)
  expect_equal(shape(result_f32), c(3, 2))
  
  # Test float64
  a_f64 <- as_tensor(a_data, dtype = "double")
  b_f64 <- as_tensor(b_data, dtype = "double")
  result_f64 <- outer_product(a_f64, b_f64)
  
  expect_true(verify_gpu_tensor(result_f64, "outer product float64"))
  expect_equal(as.array(result_f64), expected, tolerance = 1e-15)
})

test_that("Outer product handles different vector sizes", {
  # Test with different sized vectors
  a_long <- c(1, 2, 3, 4, 5)
  b_short <- c(2, 3)
  expected <- outer(a_long, b_short)
  
  a_tensor <- as_tensor(a_long, dtype = "float")
  b_tensor <- as_tensor(b_short, dtype = "float")
  result <- outer_product(a_tensor, b_tensor)
  
  expect_equal(as.array(result), expected, tolerance = 1e-6)
  expect_equal(shape(result), c(5, 2))
  
  # Test symmetric case
  vec_sym <- c(1, 2, 3)
  expected_sym <- outer(vec_sym, vec_sym)
  
  vec_tensor <- as_tensor(vec_sym, dtype = "float")
  result_sym <- outer_product(vec_tensor, vec_tensor)
  
  expect_equal(as.array(result_sym), expected_sym, tolerance = 1e-6)
  expect_equal(shape(result_sym), c(3, 3))
})

test_that("Outer product error handling", {
  # Non-1D tensors
  mat_2d <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  vec_1d <- as_tensor(c(1, 2, 3), dtype = "float")
  
  expect_error(outer_product(mat_2d, vec_1d), "1D|vector")
  expect_error(outer_product(vec_1d, mat_2d), "1D|vector")
})

# =============================================================================
# MATRIX-VECTOR MULTIPLICATION
# =============================================================================

test_that("Matrix-vector multiplication works correctly", {
  # Test A * v where A is 3x2 and v is length 2
  A_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2)
  v_data <- c(7, 8)
  expected <- A_data %*% v_data
  
  # Test float32
  A_f32 <- as_tensor(A_data, dtype = "float")
  v_f32 <- as_tensor(v_data, dtype = "float")
  result_f32 <- matvec(A_f32, v_f32)
  
  expect_true(verify_gpu_tensor(result_f32, "matvec float32"))
  expect_equal(as.vector(result_f32), as.vector(expected), tolerance = 1e-6)
  expect_equal(shape(result_f32), c(3, 1))  # Result should be column vector
  
  # Test float64
  A_f64 <- as_tensor(A_data, dtype = "double")
  v_f64 <- as_tensor(v_data, dtype = "double")
  result_f64 <- matvec(A_f64, v_f64)
  
  expect_true(verify_gpu_tensor(result_f64, "matvec float64"))
  expect_equal(as.vector(result_f64), as.vector(expected), tolerance = 1e-15)
})

test_that("Matrix-vector multiplication handles different sizes", {
  # Test with square matrix
  A_square <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
  v_square <- c(5, 6)
  expected_square <- A_square %*% v_square
  
  A_tensor <- as_tensor(A_square, dtype = "float")
  v_tensor <- as_tensor(v_square, dtype = "float")
  result <- matvec(A_tensor, v_tensor)
  
  expect_equal(as.vector(result), as.vector(expected_square), tolerance = 1e-6)
  expect_equal(shape(result), c(2, 1))
  
  # Test with larger matrix
  A_large <- matrix(1:12, nrow = 4, ncol = 3)
  v_large <- c(1, 2, 3)
  expected_large <- A_large %*% v_large
  
  A_large_tensor <- as_tensor(A_large, dtype = "float")
  v_large_tensor <- as_tensor(v_large, dtype = "float")
  result_large <- matvec(A_large_tensor, v_large_tensor)
  
  expect_equal(as.vector(result_large), as.vector(expected_large), tolerance = 1e-6)
  expect_equal(shape(result_large), c(4, 1))
})

test_that("Matrix-vector multiplication error handling", {
  # Incompatible dimensions
  A_bad <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  v_bad <- as_tensor(c(1, 2), dtype = "float")  # Wrong length
  
  expect_error(matvec(A_bad, v_bad), "Incompatible|dimensions")
  
  # Wrong tensor dimensions
  mat_2d <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  mat_2d_wrong <- as_tensor(matrix(1:4, nrow = 2, ncol = 2), dtype = "float")
  
  expect_error(matvec(mat_2d_wrong, mat_2d), "1D|vector")
})

# =============================================================================
# VECTOR-MATRIX MULTIPLICATION
# =============================================================================

test_that("Vector-matrix multiplication works correctly", {
  # Test v * A where v is length 3 and A is 3x2
  v_data <- c(1, 2, 3)
  A_data <- matrix(c(4, 5, 6, 7, 8, 9), nrow = 3, ncol = 2)
  expected <- t(v_data) %*% A_data
  
  # Test float32
  v_f32 <- as_tensor(v_data, dtype = "float")
  A_f32 <- as_tensor(A_data, dtype = "float")
  result_f32 <- vecmat(v_f32, A_f32)
  
  expect_true(verify_gpu_tensor(result_f32, "vecmat float32"))
  expect_equal(as.vector(result_f32), as.vector(expected), tolerance = 1e-6)
  expect_equal(shape(result_f32), c(1, 2))  # Result should be row vector
  
  # Test float64
  v_f64 <- as_tensor(v_data, dtype = "double")
  A_f64 <- as_tensor(A_data, dtype = "double")
  result_f64 <- vecmat(v_f64, A_f64)
  
  expect_true(verify_gpu_tensor(result_f64, "vecmat float64"))
  expect_equal(as.vector(result_f64), as.vector(expected), tolerance = 1e-15)
})

test_that("Vector-matrix multiplication handles different sizes", {
  # Test with square matrix
  v_square <- c(1, 2)
  A_square <- matrix(c(3, 4, 5, 6), nrow = 2, ncol = 2)
  expected_square <- t(v_square) %*% A_square
  
  v_tensor <- as_tensor(v_square, dtype = "float")
  A_tensor <- as_tensor(A_square, dtype = "float")
  result <- vecmat(v_tensor, A_tensor)
  
  expect_equal(as.vector(result), as.vector(expected_square), tolerance = 1e-6)
  expect_equal(shape(result), c(1, 2))
  
  # Test with rectangular matrix
  v_rect <- c(1, 2, 3, 4)
  A_rect <- matrix(1:12, nrow = 4, ncol = 3)
  expected_rect <- t(v_rect) %*% A_rect
  
  v_rect_tensor <- as_tensor(v_rect, dtype = "float")
  A_rect_tensor <- as_tensor(A_rect, dtype = "float")
  result_rect <- vecmat(v_rect_tensor, A_rect_tensor)
  
  expect_equal(as.vector(result_rect), as.vector(expected_rect), tolerance = 1e-6)
  expect_equal(shape(result_rect), c(1, 3))
})

test_that("Vector-matrix multiplication error handling", {
  # Incompatible dimensions
  v_bad <- as_tensor(c(1, 2), dtype = "float")
  A_bad <- as_tensor(matrix(1:6, nrow = 3, ncol = 2), dtype = "float")  # Wrong rows
  
  expect_error(vecmat(v_bad, A_bad), "Incompatible|dimensions")
  
  # Wrong tensor dimensions
  vec_1d <- as_tensor(c(1, 2, 3), dtype = "float")
  vec_1d_wrong <- as_tensor(c(1, 2, 3, 4), dtype = "float")
  
  expect_error(vecmat(vec_1d_wrong, vec_1d), "2D|matrix")
})

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test_that("Linear algebra operations can be chained", {
  # Test chaining: outer product -> matrix multiply
  a_vec <- c(1, 2)
  b_vec <- c(3, 4, 5)
  C_mat <- matrix(1:6, nrow = 3, ncol = 2)
  
  a_tensor <- as_tensor(a_vec, dtype = "float")
  b_tensor <- as_tensor(b_vec, dtype = "float")
  C_tensor <- as_tensor(C_mat, dtype = "float")
  
  # Create outer product: 2x3 matrix
  outer_result <- outer_product(a_tensor, b_tensor)
  expect_equal(shape(outer_result), c(2, 3))
  
  # Multiply by C: (2x3) * (3x2) = (2x2)
  final_result <- matmul(outer_result, C_tensor)
  expect_equal(shape(final_result), c(2, 2))
  
  # Verify against CPU computation
  expected <- outer(a_vec, b_vec) %*% C_mat
  expect_equal(as.array(final_result), expected, tolerance = 1e-6)
})

test_that("Linear algebra operations maintain GPU execution", {
  # Test with larger tensors to ensure GPU path
  n <- 100
  A_large <- matrix(stats::runif(n*n), nrow = n, ncol = n)
  B_large <- matrix(stats::runif(n*n), nrow = n, ncol = n)
  v_large <- stats::runif(n)
  
  A_tensor <- as_tensor(A_large, dtype = "float")
  B_tensor <- as_tensor(B_large, dtype = "float")
  v_tensor <- as_tensor(v_large, dtype = "float")
  
  # Multiple operations
  matmul_result <- matmul(A_tensor, B_tensor)
  matvec_result <- matvec(A_tensor, v_tensor)
  vecmat_result <- vecmat(v_tensor, A_tensor)
  
  expect_true(verify_gpu_tensor(matmul_result, "large matmul"))
  expect_true(verify_gpu_tensor(matvec_result, "large matvec"))
  expect_true(verify_gpu_tensor(vecmat_result, "large vecmat"))
  
  # Verify shapes
  expect_equal(shape(matmul_result), c(n, n))
  expect_equal(shape(matvec_result), c(n, 1))
  expect_equal(shape(vecmat_result), c(1, n))
})

# =============================================================================
# PERFORMANCE TESTS (from loose test files)
# =============================================================================

test_that("Large outer product performance test", {
  # Performance test for outer product (adapted from test_new_ops.R)
  n1 <- 500  # Reduced size for CI
  n2 <- 500
  
  a_large <- stats::runif(n1)
  b_large <- stats::runif(n2)
  
  a_tensor_large <- as_tensor(a_large, dtype = "float")
  b_tensor_large <- as_tensor(b_large, dtype = "float")
  
  # Time GPU version
  gpu_time <- system.time({
    result_gpu_large <- outer_product(a_tensor_large, b_tensor_large)
    synchronize(result_gpu_large)
  })
  
  # Verify correctness for a subset
  subset_indices <- 1:min(100, n1)
  subset_j <- 1:min(100, n2)
  
  expected_subset <- outer(a_large[subset_indices], b_large[subset_j])
  actual_subset <- as.array(result_gpu_large)[subset_indices, subset_j]
  
  expect_equal(actual_subset, expected_subset, tolerance = 1e-6)
  expect_equal(shape(result_gpu_large), c(n1, n2))
  expect_lt(gpu_time[["elapsed"]], 2.0)  # Should complete reasonably fast
})

test_that("Matrix-vector operations with transpose views work efficiently", {
  # Test from test_noncontiguous.R - operations on transpose views
  # Create larger matrix and transpose it
  large_data <- matrix(stats::runif(400), nrow = 20, ncol = 20)
  large_matrix <- as_tensor(large_data, dtype = "float")
  
  # Create transpose view
  transposed_matrix <- transpose(large_matrix)  # 20x20 -> 20x20 (but transposed)
  expect_equal(shape(transposed_matrix), c(20, 20))
  
  # Vector for multiplication
  v_data <- stats::runif(20)
  v_tensor <- as_tensor(v_data, dtype = "float")
  
  # Matrix-vector multiplication with transpose view
  result_matvec <- matvec(transposed_matrix, v_tensor)
  expected_matvec <- t(large_data) %*% v_data
  
  expect_equal(as.vector(result_matvec), as.vector(expected_matvec), tolerance = 1e-6)
  expect_equal(shape(result_matvec), c(20, 1))
  
  # Vector-matrix multiplication with transpose view
  result_vecmat <- vecmat(v_tensor, transposed_matrix)
  expected_vecmat <- t(v_data) %*% t(large_data)
  
  expect_equal(as.vector(result_vecmat), as.vector(expected_vecmat), tolerance = 1e-6)
  expect_equal(shape(result_vecmat), c(1, 20))
}) 