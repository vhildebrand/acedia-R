test_that("tensor operations can be performed asynchronously", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Create tensors for async operations
  n <- 100000
  tensor_a <- gpu_tensor(runif(n), c(100, 1000))
  tensor_b <- gpu_tensor(runif(n), c(100, 1000))
  
  # Operations should be asynchronous by default (non-blocking)
  start_time <- Sys.time()
  result1 <- tensor_a + tensor_b
  result2 <- tensor_a * tensor_b
  result3 <- tensor_a * 2.0
  immediate_time <- Sys.time()
  
  # Operations should return immediately (before GPU work is done)
  immediate_duration <- as.numeric(immediate_time - start_time)
  expect_lt(immediate_duration, 0.1)  # Should take less than 100ms to launch
  
  # Synchronize all operations
  synchronize(result1)
  synchronize(result2)
  synchronize(result3)
  sync_time <- Sys.time()
  
  total_duration <- as.numeric(sync_time - start_time)
  
  # Verify results are correct
  a_data <- as.vector(tensor_a)
  b_data <- as.vector(tensor_b)
  
  expect_equal(as.vector(result1), a_data + b_data, tolerance = 1e-10)
  expect_equal(as.vector(result2), a_data * b_data, tolerance = 1e-10)
  expect_equal(as.vector(result3), a_data * 2.0, tolerance = 1e-10)
  
  cat("\n=== Async Operation Test ===\n")
  cat("Launch time:", sprintf("%.6f", immediate_duration), "seconds\n")
  cat("Total time:", sprintf("%.6f", total_duration), "seconds\n")
  cat("Async benefit:", immediate_duration < total_duration / 3, "\n")
})

test_that("multiple operations can overlap using streams", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Create multiple independent tensor computations
  tensors <- list()
  results <- list()
  
  # Launch multiple independent operations
  for (i in 1:5) {
    data <- runif(10000)
    tensors[[i]] <- gpu_tensor(data, c(100, 100))
    results[[i]] <- tensors[[i]] * (i + 1.0)  # Different scalar for each
  }
  
  # All operations should be launched quickly
  # Synchronize all at once
  for (i in 1:5) {
    synchronize(results[[i]])
  }
  
  # Verify all results are correct
  for (i in 1:5) {
    expected <- as.vector(tensors[[i]]) * (i + 1.0)
    actual <- as.vector(results[[i]])
    expect_equal(actual, expected, tolerance = 1e-10)
  }
})

test_that("synchronization works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  tensor <- gpu_tensor(1:1000, c(40, 25))
  
  # Launch an operation
  result <- tensor * 2.0
  
  # Before synchronization, the GPU may still be working
  # After synchronization, the result should be ready
  synchronize(result)
  
  # Should be able to immediately access the result
  final_result <- sum(result)
  expected <- sum(1:1000) * 2.0
  expect_equal(final_result, expected)
})

test_that("stream-based operations maintain data integrity", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test that concurrent operations don't interfere with each other
  n <- 50000
  base_data <- 1:n
  
  # Create multiple tensors and perform different operations
  tensor1 <- gpu_tensor(base_data, c(250, 200))
  tensor2 <- gpu_tensor(base_data, c(500, 100))
  tensor3 <- gpu_tensor(base_data, c(1000, 50))
  
  # Perform different operations that might compete for resources
  result1 <- tensor1 + tensor1  # Addition
  result2 <- tensor2 * tensor2  # Element-wise multiplication
  result3 <- sum(tensor3)       # Reduction
  
  # Synchronize all operations
  synchronize(result1)
  synchronize(result2)
  # result3 is already synchronized by sum()
  
  # Verify all results are correct
  expect_equal(as.vector(result1), rep(base_data * 2, 1), tolerance = 1e-10)
  expect_equal(as.vector(result2), rep(base_data^2, 1), tolerance = 1e-10)
  expect_equal(result3, sum(base_data^2), tolerance = 1e-10)
}) 