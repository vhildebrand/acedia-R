test_that("GPU availability detection works", {
  # This test verifies that we can detect GPU availability
  # and handle cases where GPU is not available
  
  # Test 1: Check if we can get GPU device count
  # This is a basic CUDA runtime test
  expect_no_error({
    # Try to create a small gpuVector - this will fail if CUDA unavailable
    small_test <- as.gpuVector(c(1.0, 2.0))
    expect_true(inherits(small_test, "gpuVector"))
  })
})

test_that("CUDA error handling is comprehensive", {
  # Test various CUDA failure scenarios
  
  # Test extremely large memory allocation (should fail gracefully)
  # Most GPUs have < 24GB memory, so this should trigger cudaMalloc failure
  if (Sys.info()[["sysname"]] != "Windows") {  # Skip on Windows due to different memory model
    expect_error({
      # Try to allocate 1TB of GPU memory (will exceed any current GPU)
      huge_size <- 1e14  # 1TB worth of doubles
      huge_vec <- as.gpuVector(rep(1.0, min(huge_size, .Machine$integer.max)))
    }, "Failed to allocate GPU memory")
  }
})

test_that("package works gracefully when GPU is busy", {
  # Test behavior when GPU resources are constrained
  
  # Create multiple large GPU vectors to test resource competition
  gpu_vectors <- list()
  
  # This should succeed (create several GPU vectors)
  for (i in 1:5) {
    gpu_vectors[[i]] <- as.gpuVector(runif(10000))
    expect_true(inherits(gpu_vectors[[i]], "gpuVector"))
  }
  
  # Verify they all work independently
  for (i in 1:5) {
    result <- as.vector(gpu_vectors[[i]])
    expect_length(result, 10000)
    expect_true(all(is.finite(result)))
  }
})

test_that("GPU memory management is robust under stress", {
  # Stress test GPU memory allocation/deallocation
  
  # Rapidly create and destroy GPU vectors
  for (i in 1:20) {
    vec_size <- sample(1000:50000, 1)
    gpu_vec <- as.gpuVector(runif(vec_size))
    
    # Do some operation
    gpu_vec2 <- as.gpuVector(runif(vec_size))
    gpu_result <- gpu_vec + gpu_vec2
    
    # Convert back (this tests the full memory lifecycle)
    final_result <- as.vector(gpu_result)
    
    expect_length(final_result, vec_size)
    expect_true(all(is.finite(final_result)))
  }
})

test_that("concurrent GPU operations work correctly", {
  # Test that multiple GPU operations can coexist
  
  # Create several GPU vectors
  n <- 5000
  gpu_vecs <- lapply(1:5, function(i) as.gpuVector(runif(n, min = i, max = i + 1)))
  
  # Perform operations on all of them
  results <- list()
  for (i in 1:5) {
    for (j in 1:5) {
      if (i != j) {
        gpu_sum <- gpu_vecs[[i]] + gpu_vecs[[j]]
        results[[paste(i, j, sep = "_")]] <- as.vector(gpu_sum)
      }
    }
  }
  
  # Verify all results are reasonable
  for (result in results) {
    expect_length(result, n)
    expect_true(all(is.finite(result)))
    expect_true(all(result > 0))  # Since we used positive random numbers
  }
})

test_that("error messages are informative for GPU failures", {
  # Test that GPU errors provide helpful information to users
  
  # Test with invalid input types
  expect_error(as.gpuVector("not_numeric"), 
               "Input must be a numeric vector",
               info = "Should provide clear error for wrong input type")
  
  expect_error(as.gpuVector(list(1, 2, 3)), 
               "Input must be a numeric vector",
               info = "Should handle list inputs gracefully")
  
  # Test size mismatches
  gpu_a <- as.gpuVector(c(1, 2, 3))
  gpu_b <- as.gpuVector(c(1, 2))
  expect_error(gpu_a + gpu_b, 
               "sizes must match",
               info = "Should provide clear size mismatch error")
}) 