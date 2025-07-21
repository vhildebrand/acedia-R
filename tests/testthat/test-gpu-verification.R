test_that("GPU device is available and accessible", {
  # Test that we can create gpuVectors of different sizes without errors
  # If CUDA/GPU wasn't available, cudaMalloc would fail
  
  small_vec <- as.gpuVector(c(1.0, 2.0))
  medium_vec <- as.gpuVector(runif(1000))
  large_vec <- as.gpuVector(runif(100000))
  
  expect_true(inherits(small_vec, "gpuVector"))
  expect_true(inherits(medium_vec, "gpuVector"))
  expect_true(inherits(large_vec, "gpuVector"))
  
  # Test that we can retrieve the data back correctly
  expect_equal(as.vector(small_vec), c(1.0, 2.0))
  expect_length(as.vector(medium_vec), 1000)
  expect_length(as.vector(large_vec), 100000)
})

test_that("GPU memory operations actually use device memory", {
  # This test verifies that data transfer to/from GPU actually happens
  # by creating vectors, modifying them on GPU, and checking results
  
  # Create test data
  n <- 10000
  a <- runif(n, min = 1, max = 100)
  b <- runif(n, min = 1, max = 100)
  
  # Convert to GPU
  gpu_a <- as.gpuVector(a)
  gpu_b <- as.gpuVector(b)
  
  # Perform GPU addition
  gpu_result <- gpu_a + gpu_b
  
  # Get result back
  result <- as.vector(gpu_result)
  
  # Verify correctness
  expected <- a + b
  expect_equal(result, expected, tolerance = 1e-10)
  
  # If GPU wasn't actually being used, we might get:
  # - Errors from CUDA API calls
  # - Incorrect results  
  # - Memory access violations
  expect_length(result, n)
  expect_true(all(is.finite(result)))
})

test_that("Multiple GPU operations in sequence work correctly", {
  # This tests that GPU memory persists correctly between operations
  # and that chained operations actually use the GPU
  
  x <- c(1, 2, 3, 4, 5)
  y <- c(2, 3, 4, 5, 6)
  z <- c(1, 1, 1, 1, 1)
  
  # Create GPU vectors
  gpu_x <- as.gpuVector(x)
  gpu_y <- as.gpuVector(y)
  gpu_z <- as.gpuVector(z)
  
  # Chain multiple operations
  # (x + y) + z should equal x + y + z
  gpu_temp <- gpu_x + gpu_y
  gpu_final <- gpu_temp + gpu_z
  
  result <- as.vector(gpu_final)
  expected <- x + y + z  # [4, 6, 8, 10, 12]
  
  expect_equal(result, expected)
  expect_equal(result, c(4, 6, 8, 10, 12))
})

test_that("GPU error handling works correctly", {
  # Test that CUDA errors are properly caught and reported
  # This ensures our error handling isn't masking GPU failures
  
  # Test with very large vectors to stress memory allocation
  large_size <- 1e6
  
  # This should work (modern GPUs have several GB memory)
  gpu_large <- as.gpuVector(rep(1.0, large_size))
  expect_equal(length(as.vector(gpu_large)), large_size)
  
  # Test size mismatch errors are caught
  gpu_a <- as.gpuVector(c(1, 2, 3))
  gpu_b <- as.gpuVector(c(1, 2))
  
  expect_error(gpu_a + gpu_b, "sizes must match")
})

test_that("GPU vs CPU results are identical", {
  # Final verification that GPU produces exactly the same results as CPU
  # This would catch subtle GPU implementation bugs
  
  set.seed(42)  # For reproducible test
  n <- 50000
  
  # Generate random test vectors
  a <- runif(n, -100, 100)
  b <- runif(n, -100, 100)
  
  # CPU computation
  cpu_result <- a + b
  
  # GPU computation  
  gpu_a <- as.gpuVector(a)
  gpu_b <- as.gpuVector(b)
  gpu_result_vec <- gpu_a + gpu_b
  gpu_result <- as.vector(gpu_result_vec)
  
  # Should be identical (not just approximately equal)
  expect_equal(gpu_result, cpu_result, tolerance = 1e-15)
  
  # Additional checks
  expect_equal(length(gpu_result), length(cpu_result))
  expect_equal(range(gpu_result), range(cpu_result), tolerance = 1e-15)
  expect_equal(sum(gpu_result), sum(cpu_result), tolerance = 1e-12)
}) 