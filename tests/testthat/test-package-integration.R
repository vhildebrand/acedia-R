test_that("package loads and provides expected functionality", {
  # Test that all main functions are available after library load
  expect_true(exists("gpu_add"))
  expect_true(exists("gpu_tensor"))
  expect_true(exists("gpu_available"))
  expect_true(exists("gpu_status"))
})

test_that("complete user workflow works end-to-end", {
  # Simulate a typical user session using modern gpuTensor approach
  
  # 1. Check GPU status
  status <- gpu_status()
  expect_type(status$available, "logical")
  expect_type(status$memory_gb, "double")
  
  # 2. Perform basic operation with high-level functions
  a <- c(1, 2, 3, 4, 5)
  b <- c(2, 3, 4, 5, 6)
  result <- gpu_add(a, b)
  expect_equal(result, c(3, 5, 7, 9, 11))
  
  # 3. Use advanced gpuTensor objects (if available)
  if (status$available) {
    tensor_a <- gpu_tensor(a, length(a))
    tensor_b <- gpu_tensor(b, length(b))
    tensor_result <- tensor_a + tensor_b
    final_result <- as.vector(tensor_result)
    expect_equal(final_result, c(3, 5, 7, 9, 11))
    
    # Test tensor-specific functionality
    expect_equal(shape(tensor_a), length(a))
    expect_equal(size(tensor_a), length(a))
  }
})

test_that("package handles edge cases gracefully", {
  # Empty vectors
  expect_equal(gpu_add(numeric(0), numeric(0)), numeric(0))
  
  # Large vectors
  n <- 10000
  large_a <- runif(n)
  large_b <- runif(n)
  large_result <- gpu_add(large_a, large_b)
  expect_length(large_result, n)
  expect_equal(large_result, large_a + large_b, tolerance = 1e-14)
  
  # Test tensor operations with large data
  if (gpu_available()) {
    large_tensor_a <- gpu_tensor(large_a, c(100, 100))
    large_tensor_b <- gpu_tensor(large_b, c(100, 100))
    large_tensor_result <- large_tensor_a + large_tensor_b
    expect_equal(as.vector(large_tensor_result), large_a + large_b, tolerance = 1e-14)
  }
}) 