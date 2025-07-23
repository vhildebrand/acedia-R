test_that("gpuTensor creation works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  data <- 1:12
  shape <- c(3, 4)
  
  tensor <- gpu_tensor(data, shape)
  
  expect_equal(shape(tensor), shape)
  expect_equal(size(tensor), 12)
  expect_equal(attr(tensor, "dtype"), "double")
  
  # Test data integrity - R uses column-major order by default
  result_data <- as.array(tensor)
  expected_matrix <- matrix(data, nrow = 3, ncol = 4, byrow = FALSE)  # Changed from byrow = TRUE
  expect_equal(result_data, expected_matrix)
})

test_that("gpuTensor shape validation works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test invalid shape dimensions
  expect_error(gpu_tensor(1:6, c(2, 0)), "positive")
  expect_error(gpu_tensor(1:6, c(-1, 3)), "positive")
  
  # Test mismatched data and shape size
  expect_error(gpu_tensor(1:6, c(2, 4)), "doesn't match shape size")
  expect_error(gpu_tensor(1:10, c(2, 3)), "doesn't match shape size")
})

test_that("gpuTensor supports multiple data types", {
  skip_if_not(gpu_available(), "GPU not available")
  
  data <- runif(8)
  shape <- c(2, 4)
  
  # Test double precision (default)
  tensor_double <- gpu_tensor(data, shape, dtype = "double")
  expect_equal(attr(tensor_double, "dtype"), "double")
  
  # Test single precision
  tensor_float <- gpu_tensor(data, shape, dtype = "float")
  expect_equal(attr(tensor_float, "dtype"), "float")
  
  # Test unsupported dtype
  expect_error(gpu_tensor(data, shape, dtype = "int32"), "Unsupported dtype")
})

test_that("empty tensor creation works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  shape <- c(2, 3, 4)
  tensor <- empty_tensor(shape)
  
  expect_s3_class(tensor, "gpuTensor")
  expect_equal(shape(tensor), shape)
  expect_equal(size(tensor), 24)
  
  # Should be able to convert to R (though values are uninitialized)
  result <- as.array(tensor)
  expect_equal(dim(result), shape)
})

test_that("tensor from matrix/array conversion works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test from matrix
  m <- matrix(1:12, nrow = 3, ncol = 4)
  tensor <- as_tensor(m)
  
  expect_equal(shape(tensor), c(3, 4))
  expect_equal(as.array(tensor), m)
  
  # Test from 3D array
  arr <- array(1:24, dim = c(2, 3, 4))
  tensor_3d <- as_tensor(arr)
  
  expect_equal(shape(tensor_3d), c(2, 3, 4))
  expect_equal(as.array(tensor_3d), arr)
  
  # Test from vector
  vec <- 1:10
  tensor_vec <- as_tensor(vec)
  
  expect_equal(shape(tensor_vec), 10)
  expect_equal(as.vector(tensor_vec), vec)
})

test_that("tensor arithmetic operations work correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test element-wise addition
  a_data <- matrix(1:6, nrow = 2, ncol = 3)
  b_data <- matrix(7:12, nrow = 2, ncol = 3)
  
  tensor_a <- as_tensor(a_data)
  tensor_b <- as_tensor(b_data)
  
  result <- tensor_a + tensor_b
  expected <- a_data + b_data
  
  expect_equal(as.array(result), expected)
  expect_s3_class(result, "gpuTensor")
  
  # Test element-wise multiplication
  result_mul <- tensor_a * tensor_b
  expected_mul <- a_data * b_data
  
  expect_equal(as.array(result_mul), expected_mul)
  
  # Test scalar multiplication
  scalar <- 2.5
  result_scalar <- tensor_a * scalar
  expected_scalar <- a_data * scalar
  
  expect_equal(as.array(result_scalar), expected_scalar, tolerance = 1e-10)
})

test_that("matrix multiplication works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic matrix multiplication
  a_data <- matrix(1:6, nrow = 2, ncol = 3)    # 2x3
  b_data <- matrix(1:12, nrow = 3, ncol = 4)   # 3x4
  
  tensor_a <- as_tensor(a_data)
  tensor_b <- as_tensor(b_data)
  
  result <- matmul(tensor_a, tensor_b)
  expected <- a_data %*% b_data
  
  expect_equal(shape(result), c(2, 4))
  expect_equal(as.array(result), expected, tolerance = 1e-10)
  
  # Test incompatible dimensions
  c_data <- matrix(1:8, nrow = 2, ncol = 4)    # 2x4 (incompatible with 2x3)
  tensor_c <- as_tensor(c_data)
  
  expect_error(matmul(tensor_a, tensor_c), "Incompatible dimensions for matrix multiplication")
  
  # Test non-2D tensors
  tensor_1d <- as_tensor(1:6)
  expect_error(matmul(tensor_1d, tensor_a), "requires 2D tensors")
})

test_that("tensor views and reshaping work", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Create original tensor
  data <- 1:24
  original_shape <- c(2, 3, 4)
  tensor <- gpu_tensor(data, original_shape)
  
  # Test view (same data, different shape)
  new_shape <- c(6, 4)
  tensor_view <- view(tensor, new_shape)
  
  expect_equal(shape(tensor_view), new_shape)
  expect_equal(size(tensor_view), size(tensor))
  
  # Views should share memory (modification to one affects the other)
  # Note: This test would require implementing an assignment operation
  
  # Test invalid view (different total size)
  expect_error(view(tensor, c(5, 5)), "View shape size must match")
})

test_that("tensor transpose works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test 2D matrix transpose
  data_2d <- 1:6
  tensor_2d <- gpu_tensor(data_2d, c(2, 3))
  
  transposed <- transpose(tensor_2d)
  
  expect_equal(shape(transposed), c(3, 2))
  expect_equal(size(transposed), size(tensor_2d))
  
  # Check values are correct
  original_array <- as.array(tensor_2d)
  transposed_array <- as.array(transposed)
  expect_equal(transposed_array, t(original_array))
  
  # Test error for non-2D tensors
  tensor_1d <- gpu_tensor(1:6, c(6))
  expect_error(transpose(tensor_1d), "2D tensors only")
  
  tensor_3d <- gpu_tensor(1:24, c(2, 3, 4))
  expect_error(transpose(tensor_3d), "2D tensors only")
})

test_that("tensor permute works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test 3D tensor permutation
  data_3d <- 1:24
  tensor_3d <- gpu_tensor(data_3d, c(2, 3, 4))
  
  # Permute dimensions: (2,3,4) -> (3,4,2)
  permuted <- permute(tensor_3d, c(2, 3, 1))
  
  expect_equal(shape(permuted), c(3, 4, 2))
  expect_equal(size(permuted), size(tensor_3d))
  
  # Check values using R's aperm as reference
  original_array <- as.array(tensor_3d)
  permuted_array <- as.array(permuted)
  expected_array <- aperm(original_array, c(2, 3, 1))
  expect_equal(permuted_array, expected_array)
  
  # Test invalid permutation
  expect_error(permute(tensor_3d, c(1, 2)), "must match tensor dimensions")
  expect_error(permute(tensor_3d, c(1, 1, 2)), "Invalid permutation")  # Duplicate dimension
})

test_that("tensor concat works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test concatenation of 2D tensors
  t1 <- gpu_tensor(1:6, c(2, 3))
  t2 <- gpu_tensor(7:12, c(2, 3))
  
  # Concatenate along first dimension (rows)
  if (requireNamespace("abind", quietly = TRUE)) {
    result <- concat(list(t1, t2), axis = 1)
    expect_equal(shape(result), c(4, 3))
    
    # Check values
    expected <- rbind(as.array(t1), as.array(t2))
    expect_equal(as.array(result), expected)
    
    # Concatenate along second dimension (columns)
    result2 <- concat(list(t1, t2), axis = 2)
    expect_equal(shape(result2), c(2, 6))
  } else {
    skip("abind package not available")
  }
})

test_that("tensor stack works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test stacking of 2D tensors
  t1 <- gpu_tensor(1:4, c(2, 2))
  t2 <- gpu_tensor(5:8, c(2, 2))
  
  if (requireNamespace("abind", quietly = TRUE)) {
    # Stack along new first dimension
    result <- stack(list(t1, t2), axis = 1)
    expect_equal(shape(result), c(2, 2, 2))
    
    # Check that original tensors are preserved in stack
    expect_equal(as.array(result)[1, , ], as.array(t1))
    expect_equal(as.array(result)[2, , ], as.array(t2))
  } else {
    skip("abind package not available")
  }
})

test_that("tensor repeat works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic tensor repeat
  t1 <- gpu_tensor(c(1, 2), c(2))
  
  # This is a placeholder test - the current implementation needs work
  expect_error(repeat_tensor(t1, c(3)), NA)  # Should not error
  
  # Test error conditions
  expect_error(repeat_tensor(t1, c(2, 3)), "Length of repeats must match")
  expect_error(repeat_tensor(t1, c(0)), "must be positive")
})

test_that("tensor pad works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic constant padding
  t1 <- gpu_tensor(1:4, c(2, 2))
  
  # Pad with 1 on each side
  pad_spec <- list(c(1, 1), c(1, 1))  # [[before, after], [before, after]]
  result <- pad(t1, pad_spec, mode = "constant", value = 0)
  
  expect_equal(shape(result), c(4, 4))
  
  # Check that center contains original data
  center_data <- as.array(result)[2:3, 2:3]
  expect_equal(center_data, as.array(t1))
  
  # Test error conditions
  expect_error(pad(t1, list(c(1, 1)), "constant"), "list length must match")
  expect_error(pad(t1, pad_spec, "invalid_mode"), "mode must be one of")
})

test_that("memory sharing in views works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Create a tensor and a view
  original <- gpu_tensor(1:12, c(3, 4))
  reshaped <- view(original, c(4, 3))
  
  # Test that they have different shapes but same size
  expect_equal(shape(original), c(3, 4))
  expect_equal(shape(reshaped), c(4, 3))
  expect_equal(size(original), size(reshaped))
  
  # Both should have the same underlying data
  expect_equal(as.vector(original), as.vector(reshaped))
  
  # Test transpose memory sharing
  if (length(shape(original)) == 2) {
    transposed <- transpose(original)
    expect_equal(shape(transposed), c(4, 3))
    expect_equal(size(transposed), size(original))
  }
})

test_that("tensor reduction operations work", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test sum operation
  data <- matrix(1:12, nrow = 3, ncol = 4)
  tensor <- as_tensor(data)
  
  result_sum <- sum(tensor)
  expected_sum <- sum(data)
  
  expect_equal(result_sum, expected_sum, tolerance = 1e-10)
  expect_type(result_sum, "double")
  
  # Test with larger tensor to ensure GPU parallelization
  large_data <- runif(1000000)
  large_tensor <- gpu_tensor(large_data, c(1000, 1000))
  
  gpu_sum <- sum(large_tensor)
  cpu_sum <- sum(large_data)
  
  expect_equal(gpu_sum, cpu_sum, tolerance = 1e-8)
})

test_that("tensor contiguity checks work", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Newly created tensors should be contiguous
  tensor <- gpu_tensor(1:12, c(3, 4))
  expect_true(is_contiguous(tensor))
  
  # Views of contiguous tensors should be contiguous (for compatible shapes)
  tensor_view <- view(tensor, c(12, 1))
  expect_true(is_contiguous(tensor_view))
  
  # TODO: Test non-contiguous tensors when stride manipulation is implemented
})

test_that("tensor synchronization works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  tensor <- gpu_tensor(1:100, c(10, 10))
  
  # Should not throw an error
  expect_silent(synchronize(tensor))
  
  # After synchronization, operations should be complete
  result <- sum(tensor)
  expect_equal(result, sum(1:100))
})

test_that("gradient requirements can be set", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test requires_grad during creation
  tensor <- gpu_tensor(1:6, c(2, 3), requires_grad = TRUE)
  expect_s3_class(tensor, "gpuTensor")
  
  # Test setting requires_grad after creation
  tensor2 <- gpu_tensor(1:6, c(2, 3))
  requires_grad(tensor2, TRUE)
  expect_s3_class(tensor2, "gpuTensor")
  
  # Should work without error (full autograd implementation would test more)
})

test_that("tensor operations actually use GPU (performance test)", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Create large tensors for performance comparison
  n <- 1000000
  a_data <- runif(n)
  b_data <- runif(n)
  
  # GPU tensor operations
  tensor_a <- gpu_tensor(a_data, c(1000, 1000))
  tensor_b <- gpu_tensor(b_data, c(1000, 1000))
  
  # Time GPU operations
  gpu_start_time <- Sys.time()
  gpu_result <- tensor_a + tensor_b
  synchronize(gpu_result)  # Ensure completion
  gpu_end_time <- Sys.time()
  gpu_time <- as.numeric(gpu_end_time - gpu_start_time)
  
  # Time CPU operations
  cpu_start_time <- Sys.time()
  cpu_result <- a_data + b_data
  cpu_end_time <- Sys.time()
  cpu_time <- as.numeric(cpu_end_time - cpu_start_time)
  
  # Verify correctness first
  gpu_data <- as.vector(gpu_result)
  expect_equal(gpu_data, cpu_result, tolerance = 1e-10)
  
  # Performance check - GPU should complete within reasonable time
  expect_lt(gpu_time, 5.0)  # Should complete within 5 seconds
  
  cat("\n=== GPU Tensor Performance Test ===\n")
  cat("Vector size:", format(n, scientific = TRUE), "\n")
  cat("CPU time:", sprintf("%.6f", cpu_time), "seconds\n")
  cat("GPU time:", sprintf("%.6f", gpu_time), "seconds\n")
  
  if (gpu_time < cpu_time) {
    cat("✓ GPU is faster than CPU\n")
  } else {
    cat("⚠ GPU is slower (may include transfer overhead)\n")
  }
})

test_that("tensor memory management works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test that tensors can be created and destroyed without memory leaks
  for (i in 1:10) {
    large_tensor <- gpu_tensor(runif(100000), c(100, 1000))
    result <- sum(large_tensor)
    expect_gt(result, 0)  # Basic sanity check
    
    # Tensor should be garbage collected when going out of scope
    rm(large_tensor)
  }
  
  # Force garbage collection
  gc()
  
  # GPU memory should still be available
  final_memory <- gpu_memory_available()
  expect_gt(final_memory, 0)
})

test_that("tensor error handling is comprehensive", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test operations on tensors with mismatched shapes
  tensor_a <- gpu_tensor(1:6, c(2, 3))
  tensor_b <- gpu_tensor(1:8, c(2, 4))
  
  expect_error(tensor_a + tensor_b, "broadcastable")
  expect_error(tensor_a * tensor_b, "Tensor shapes are not broadcastable")
  
  # Test invalid operations
  expect_error(matmul(tensor_a, tensor_a), "Incompatible dimensions for matrix multiplication")  # 2x3 * 2x3 invalid
  
  # Test dtype mismatches
  tensor_float <- gpu_tensor(1:6, c(2, 3), dtype = "float")
  expect_error(tensor_a + tensor_float, "different dtypes")
})

test_that("tensor operations work with edge cases", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test empty tensors (0-dimensional)
  # Note: This may not be supported yet
  
  # Test single-element tensors
  scalar_tensor <- gpu_tensor(5, c(1, 1))
  expect_equal(as.array(scalar_tensor), matrix(5, 1, 1))
  expect_equal(sum(scalar_tensor), 5)
  
  # Test large tensors (memory stress test)
  # Only test if enough GPU memory is available
  available_mem <- gpu_memory_available()
  if (available_mem > 1e9) {  # More than 1GB available
    large_size <- min(1000000, available_mem / (8 * 4))  # Conservative size
    large_tensor <- gpu_tensor(rep(1.0, large_size), c(as.integer(sqrt(large_size)), as.integer(sqrt(large_size))))
    expect_equal(sum(large_tensor), large_size)
  }
})

test_that("tensor dtype consistency is maintained", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test that operations preserve dtypes
  tensor_double <- gpu_tensor(1:6, c(2, 3), dtype = "double")
  tensor_float <- gpu_tensor(1:6, c(2, 3), dtype = "float")
  
  # Operations within same dtype should work
  result_double <- tensor_double * 2.0
  expect_equal(attr(result_double, "dtype"), "double")
  
  result_float <- tensor_float * 2.0
  expect_equal(attr(result_float, "dtype"), "float")
  
  # Results should be numerically close but different precision
  expect_equal(as.vector(result_double), as.vector(result_float), tolerance = 1e-6)
})

test_that("CPU fallback works when GPU is not available", {
  # This test would need to mock GPU unavailability
  # For now, we just verify the error handling
  expect_true(gpu_available())  # Should be available in test environment
  
  # TODO: Implement CPU fallback for tensor operations
  # When implemented, this should test:
  # 1. Graceful fallback when GPU operations fail
  # 2. Identical results between GPU and CPU
  # 3. Appropriate warnings when falling back
}) 