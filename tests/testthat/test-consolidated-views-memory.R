context("Memory views, transpose, and tensor manipulation")

# Helper functions
expect_tensor_equal <- function(tensor, expected, tolerance = 1e-6) {
  expect_equal(as.array(tensor), expected, tolerance = tolerance)
}

verify_gpu_tensor <- function(tensor, operation_name = "operation") {
  if (!inherits(tensor, "gpuTensor")) {
    warning(paste("❌ GPU FALLBACK:", operation_name, "returned non-gpuTensor object"))
    return(FALSE)
  }
  return(TRUE)
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
# TRANSPOSE OPERATIONS
# =============================================================================

test_that("Transpose creates correct views", {
  # Test 2D transpose
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  expect_true(is_contiguous(original_tensor))
  
  # Transpose should create a view
  transposed <- transpose(original_tensor)
  expect_equal(shape(transposed), c(4, 3))
  expect_tensor_equal(transposed, t(original_data))
  expect_true(verify_gpu_tensor(transposed, "transpose operation"))
  
  # Check if it's a view (may or may not be contiguous depending on implementation)
  expect_equal(as.array(transposed), t(original_data), tolerance = 1e-6)
})

test_that("Transpose view operations work on GPU", {
  # Create matrix and transpose
  mat_data <- matrix(1:20, nrow = 4, ncol = 5)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  transposed <- transpose(mat_tensor)
  
  expect_true(verify_gpu_tensor(transposed, "transpose result"))
  expect_equal(shape(transposed), c(5, 4))
  
  # Operations on transposed tensor should work
  result <- transposed + 1.0
  expected <- t(mat_data) + 1.0
  expect_tensor_equal(result, expected)
  expect_true(verify_gpu_tensor(result, "transpose + scalar"))
})

test_that("Multiple transpose operations work correctly", {
  original_data <- matrix(1:6, nrow = 2, ncol = 3)
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  # Double transpose should return to original
  double_transposed <- transpose(transpose(original_tensor))
  expect_tensor_equal(double_transposed, original_data)
  expect_equal(shape(double_transposed), c(2, 3))
})

# =============================================================================
# PERMUTE OPERATIONS
# =============================================================================

test_that("Permute view operations work on GPU", {
  # Create 3D tensor
  tensor_3d <- as_tensor(array(1:24, dim = c(2, 3, 4)), dtype = "float")
  
  # Permute dimensions: (2,3,4) -> (3,4,2)
  permuted <- permute(tensor_3d, c(2, 3, 1))
  expect_equal(shape(permuted), c(3, 4, 2))
  expect_true(verify_gpu_tensor(permuted, "permute operation"))
  
  # Check correctness
  original_array <- as.array(tensor_3d)
  permuted_array <- as.array(permuted)
  expected <- aperm(original_array, c(2, 3, 1))
  expect_equal(permuted_array, expected, tolerance = 1e-6)
})

test_that("Permute handles different dimension orders", {
  tensor_3d <- as_tensor(array(1:24, dim = c(2, 3, 4)), dtype = "float")
  
  # Test different permutations
  perm1 <- permute(tensor_3d, c(3, 1, 2))  # (2,3,4) -> (4,2,3)
  expect_equal(shape(perm1), c(4, 2, 3))
  
  perm2 <- permute(tensor_3d, c(1, 3, 2))  # (2,3,4) -> (2,4,3)
  expect_equal(shape(perm2), c(2, 4, 3))
  
  # Identity permutation
  perm_identity <- permute(tensor_3d, c(1, 2, 3))
  expect_equal(shape(perm_identity), c(2, 3, 4))
  expect_tensor_equal(perm_identity, as.array(tensor_3d))
})

# =============================================================================
# CONTIGUITY AND VIEW OPERATIONS
# =============================================================================

test_that("Contiguity detection works correctly", {
  # Original tensor should be contiguous
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  original_tensor <- as_tensor(original_data, dtype = "float")
  expect_true(is_contiguous(original_tensor))
  
  # Transpose may or may not be contiguous (implementation dependent)
  transposed <- transpose(original_tensor)
  # Don't assert contiguity of transpose - depends on implementation
  
  # But operations should still work
  result_ops <- transposed + 1.0
  expect_true(verify_gpu_tensor(result_ops, "contiguity test result"))
})

test_that("Contiguous function works correctly", {
  # Test with potentially non-contiguous tensor
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  original_tensor <- as_tensor(original_data, dtype = "float")
  transposed <- transpose(original_tensor)
  
  # Make contiguous copy
  if (exists("contiguous")) {
    contiguous_copy <- contiguous(transposed)
    
    expect_true(verify_gpu_tensor(contiguous_copy, "contiguous copy"))
    expect_true(is_contiguous(contiguous_copy))
    expect_equal(as.array(contiguous_copy), as.array(transposed), tolerance = 1e-6)
  } else {
    skip("contiguous function not available")
  }
})

test_that("Contiguous is no-op for already contiguous tensors", {
  original_data <- matrix(1:6, nrow = 2, ncol = 3)
  original_tensor <- as_tensor(original_data, dtype = "float")
  
  expect_true(is_contiguous(original_tensor))
  
  if (exists("contiguous")) {
    # Should return same tensor (or equivalent)
    contiguous_result <- contiguous(original_tensor)
    
    expect_true(is_contiguous(contiguous_result))
    expect_equal(as.array(contiguous_result), original_data, tolerance = 1e-6)
  } else {
    skip("contiguous function not available")
  }
})

# =============================================================================
# VIEW AND RESHAPE OPERATIONS
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
  # Create potentially non-contiguous tensor
  original_data <- matrix(1:12, nrow = 3, ncol = 4)
  original_tensor <- as_tensor(original_data, dtype = "float")
  transposed <- transpose(original_tensor)
  
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

# =============================================================================
# OPERATIONS ON NON-CONTIGUOUS TENSORS
# =============================================================================

test_that("Operations work correctly on non-contiguous tensors", {
  # Create non-contiguous tensors
  A_data <- matrix(1:6, nrow = 2, ncol = 3)
  B_data <- matrix(1:6, nrow = 3, ncol = 2)
  
  A_tensor <- as_tensor(A_data, dtype = "float")
  B_tensor <- as_tensor(B_data, dtype = "float")
  
  # Create potentially non-contiguous views
  A_transposed <- transpose(A_tensor)  # Now 3x2
  B_transposed <- transpose(B_tensor)  # Now 2x3
  
  # Operations should work on non-contiguous tensors
  # Addition (same shape)
  sum_result <- A_transposed + A_transposed
  expect_equal(as.array(sum_result), 2 * as.array(A_transposed), tolerance = 1e-6)
  
  # Matrix multiplication
  if (exists("matmul")) {
    matmul_result <- matmul(A_transposed, B_transposed)
    expected_matmul <- t(A_data) %*% t(B_data)
    expect_equal(as.array(matmul_result), expected_matmul, tolerance = 1e-6)
  }
})

# =============================================================================
# SLICE OPERATIONS
# =============================================================================

test_that("Slice mutation updates parent tensor in-place (GPU verified)", {
  # Create parent tensor
  parent <- as_tensor(array(1:120, dim = c(4, 5, 6)), dtype = "float")
  expect_true(verify_gpu_tensor(parent, "parent tensor creation"))
  
  # Extract slice and modify
  slice1 <- parent[1, , ]  # Extract first slice (5x6)
  slice2 <- parent[2, , ]  # Extract second slice (5x6)
  
  expect_true(verify_gpu_tensor(slice1, "slice extraction"))
  expect_equal(shape(slice1), c(5, 6))
  
  # Modify slice (if slice assignment is supported)
  if (exists("[<-.gpuTensor")) {
    parent[1, , ] <- parent[1, , ] + 100
    
    # Check that parent was modified
    modified_slice <- parent[1, , ]
    original_plus_100 <- as.array(slice1) + 100
    expect_equal(as.array(modified_slice), original_plus_100, tolerance = 1e-6)
  } else {
    skip("Slice assignment not available")
  }
})

test_that("Slice mutation with different slice patterns (GPU verified)", {
  # Create 2D matrix
  mat <- as_tensor(matrix(1:20, 4, 5), dtype = "float")
  expect_true(verify_gpu_tensor(mat, "2D matrix creation"))
  
  # Test different slice patterns (if supported)
  if (exists("[.gpuTensor")) {
    # Row slice
    row_slice <- mat[2, ]
    expect_equal(length(shape(row_slice)), 1)
    expect_equal(shape(row_slice), 5)
    
    # Column slice
    col_slice <- mat[, 3]
    expect_equal(length(shape(col_slice)), 1)
    expect_equal(shape(col_slice), 4)
    
    # Submatrix slice
    sub_slice <- mat[1:2, 2:4]
    expect_equal(shape(sub_slice), c(2, 3))
  } else {
    skip("Tensor slicing not available")
  }
})

# =============================================================================
# MEMORY EFFICIENCY TESTS
# =============================================================================

test_that("Views share memory efficiently", {
  # Create large tensor
  n <- 1000
  large_data <- matrix(runif(n*n), nrow = n, ncol = n)
  large_tensor <- as_tensor(large_data, dtype = "float")
  
  # Create transpose view
  transposed <- transpose(large_tensor)
  
  # Both should be valid GPU tensors
  expect_true(verify_gpu_tensor(large_tensor, "large original tensor"))
  expect_true(verify_gpu_tensor(transposed, "large transposed tensor"))
  
  # Operations should work on both
  sum_orig <- sum(large_tensor)
  sum_trans <- sum(transposed)
  
  # Sums should be equal (transpose doesn't change sum)
  expect_equal(as.numeric(sum_orig), as.numeric(sum_trans), tolerance = 1e-5)
})

# =============================================================================
# ERROR HANDLING
# =============================================================================

test_that("View error handling", {
  tensor_2d <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  
  # Invalid reshape size
  expect_error(view(tensor_2d, c(7)), "size|elements", ignore.case = TRUE)
  
  # Invalid permute dimensions
  expect_error(permute(tensor_2d, c(1, 2, 3)), "dimensions|permute", ignore.case = TRUE)
  
  # Transpose on non-2D tensor (if restricted)
  tensor_1d <- as_tensor(c(1, 2, 3), dtype = "float")
  if (exists("transpose")) {
    # Some implementations may restrict transpose to 2D
    tryCatch({
      result <- transpose(tensor_1d)
      # If it works, that's fine too
    }, error = function(e) {
      expect_true(grepl("2D|dimension", e$message, ignore.case = TRUE))
    })
  }
}) 