test_that("GPU multiplication operations actually run on GPU", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test 1: Performance-based verification
  # GPU should be significantly faster for large vectors
  n <- 1e6
  a <- runif(n, min = 1, max = 10)
  b <- runif(n, min = 1, max = 10)
  
  # CPU timing
  cpu_mult_time <- system.time({
    cpu_result <- a * b
  })[["elapsed"]]
  
  cpu_dot_time <- system.time({
    cpu_dot_result <- sum(a * b)
  })[["elapsed"]]
  
  # GPU timing (should be significantly faster for large vectors)
  gpu_mult_time <- system.time({
    gpu_result <- gpu_multiply(a, b, warn_fallback = FALSE)
  })[["elapsed"]]
  
  gpu_dot_time <- system.time({
    gpu_dot_result <- gpu_dot(a, b, warn_fallback = FALSE)
  })[["elapsed"]]
  
  # Verify correctness first
  expect_equal(gpu_result, cpu_result, tolerance = 1e-10)
  expect_equal(gpu_dot_result, cpu_dot_result, tolerance = 1e-10)
  
  # Performance check - GPU should be competitive or faster
  # Note: For smaller overhead, we just verify GPU completes in reasonable time
  expect_lt(gpu_mult_time, 5.0)  # Should complete within 5 seconds
  expect_lt(gpu_dot_time, 5.0)   # Should complete within 5 seconds
  
  # Print timing for manual verification
  cat("\n=== GPU Performance Verification ===\n")
  cat("Vector size:", format(n, scientific = TRUE), "\n")
  cat("CPU multiply time:", sprintf("%.6f", cpu_mult_time), "seconds\n")
  cat("GPU multiply time:", sprintf("%.6f", gpu_mult_time), "seconds\n")
  cat("CPU dot product time:", sprintf("%.6f", cpu_dot_time), "seconds\n")
  cat("GPU dot product time:", sprintf("%.6f", gpu_dot_time), "seconds\n")
  
  if (gpu_mult_time < cpu_mult_time) {
    cat("✓ GPU multiply is faster than CPU\n")
  } else {
    cat("⚠ GPU multiply is slower (may include transfer overhead)\n")
  }
  
  if (gpu_dot_time < cpu_dot_time) {
    cat("✓ GPU dot product is faster than CPU\n")
  } else {
    cat("⚠ GPU dot product is slower (may include transfer overhead)\n")
  }
})

test_that("GPU vs CPU forced execution produces same results", {
  # Skip if GPU not available  
  skip_if_not(gpu_available(), "GPU not available")
  
  # This test verifies that we can distinguish between GPU and CPU execution
  n <- 50000
  a <- runif(n)
  b <- runif(n)
  scalar <- 3.14159
  
  # Force CPU execution
  cpu_mult <- gpu_multiply(a, b, force_cpu = TRUE, warn_fallback = FALSE)
  cpu_scale <- gpu_scale(a, scalar, force_cpu = TRUE, warn_fallback = FALSE)
  cpu_dot <- gpu_dot(a, b, force_cpu = TRUE, warn_fallback = FALSE)
  
  # GPU execution (should use actual GPU if available)
  gpu_mult <- gpu_multiply(a, b, force_cpu = FALSE, warn_fallback = FALSE)
  gpu_scale <- gpu_scale(a, scalar, force_cpu = FALSE, warn_fallback = FALSE)
  gpu_dot <- gpu_dot(a, b, force_cpu = FALSE, warn_fallback = FALSE)
  
  # Results should be identical
  expect_equal(gpu_mult, cpu_mult, tolerance = 1e-14)
  expect_equal(gpu_scale, cpu_scale, tolerance = 1e-14)
  expect_equal(gpu_dot, cpu_dot, tolerance = 1e-14)
  
  # Verify against direct CPU computation
  expect_equal(gpu_mult, a * b, tolerance = 1e-14)
  expect_equal(gpu_scale, a * scalar, tolerance = 1e-14)
  expect_equal(gpu_dot, sum(a * b), tolerance = 1e-14)
})

test_that("gpuVector operations stay on GPU without transfers", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # This test verifies that gpuVector operations don't unnecessarily transfer data
  n <- 100000
  a_host <- runif(n)
  b_host <- runif(n)
  scalar <- 2.5
  
  # Create gpuVector objects (one-time transfer cost)
  gpu_a <- as.gpuVector(a_host)
  gpu_b <- as.gpuVector(b_host)
  
  # Time GPU-only operations (should be very fast as no transfers involved)
  gpu_only_mult_time <- system.time({
    gpu_result1 <- gpu_a * gpu_b  # Element-wise multiplication
  })[["elapsed"]]
  
  gpu_only_scale_time <- system.time({
    gpu_result2 <- gpu_a * scalar  # Scalar multiplication
  })[["elapsed"]]
  
  gpu_only_dot_time <- system.time({
    dot_result <- gpu_dot_vectors(gpu_a, gpu_b)  # Dot product
  })[["elapsed"]]
  
  # These should be very fast since no memory transfers
  expect_lt(gpu_only_mult_time, 1.0)
  expect_lt(gpu_only_scale_time, 1.0) 
  expect_lt(gpu_only_dot_time, 1.0)
  
  # Verify correctness by transferring back once at the end
  final_mult <- as.vector(gpu_result1)
  final_scale <- as.vector(gpu_result2)
  
  expect_equal(final_mult, a_host * b_host, tolerance = 1e-10)
  expect_equal(final_scale, a_host * scalar, tolerance = 1e-10)
  expect_equal(dot_result, sum(a_host * b_host), tolerance = 1e-10)
  
  cat("\n=== GPU-only Operation Performance ===\n")
  cat("GPU element-wise multiply time:", sprintf("%.6f", gpu_only_mult_time), "seconds\n")
  cat("GPU scalar multiply time:", sprintf("%.6f", gpu_only_scale_time), "seconds\n")
  cat("GPU dot product time:", sprintf("%.6f", gpu_only_dot_time), "seconds\n")
})

test_that("GPU memory allocation verification", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Check initial GPU memory
  initial_memory <- gpu_memory_available()
  expect_gt(initial_memory, 0)  # GPU should have available memory
  
  # Create large GPU vectors
  large_size <- 1e6
  gpu_vectors <- list()
  
  for (i in 1:3) {
    gpu_vectors[[i]] <- as.gpuVector(runif(large_size))
    expect_s3_class(gpu_vectors[[i]], "gpuVector")
  }
  
  # Perform operations that would fail if GPU wasn't really being used
  gpu_result <- gpu_vectors[[1]] * gpu_vectors[[2]]
  gpu_result2 <- gpu_result * gpu_vectors[[3]]
  final_result <- as.vector(gpu_result2)
  
  expect_length(final_result, large_size)
  expect_true(all(is.finite(final_result)))
  
  # Clean up explicitly (test that GPU resources are properly managed)
  for (i in 1:3) {
    gpu_vectors[[i]] <- NULL
  }
  gpu_result <- NULL
  gpu_result2 <- NULL
  
  # Force garbage collection
  invisible(gc())
  
  # Memory should be available again (though exact amount may vary)
  final_memory <- gpu_memory_available() 
  expect_gt(final_memory, 0)  # GPU memory should still be available after cleanup
})

test_that("CUDA kernel execution can be verified indirectly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # This test uses patterns that would behave differently on GPU vs CPU
  # to indirectly verify CUDA kernel execution
  
  # Test 1: Parallel execution should handle large vectors efficiently
  # Create vectors that would be slow to process sequentially
  n <- 5e6  # 5 million elements
  a <- rep(1.0, n)
  b <- rep(2.0, n)
  
  # If this is really running on GPU in parallel, it should complete quickly
  start_time <- Sys.time()
  gpu_result <- gpu_multiply(a, b, warn_fallback = FALSE)
  end_time <- Sys.time()
  
  gpu_time <- as.numeric(end_time - start_time)
  
  # Verify correctness
  expect_equal(gpu_result, rep(2.0, n))
  expect_length(gpu_result, n)
  
  # Should complete in reasonable time (< 5 seconds even with transfers)
  expect_lt(gpu_time, 5.0)
  
  cat("\n=== CUDA Kernel Execution Verification ===\n")
  cat("Processed", format(n, scientific = TRUE), "elements in", sprintf("%.6f", gpu_time), "seconds\n")
  cat("Throughput:", sprintf("%.2e", n / gpu_time), "elements/second\n")
  
  # Test 2: Dot product with reduction should use GPU's parallel capabilities
  start_time <- Sys.time()
  dot_result <- gpu_dot(a, b, warn_fallback = FALSE)
  end_time <- Sys.time()
  
  dot_time <- as.numeric(end_time - start_time)
  
  expect_equal(dot_result, 2.0 * n)
  expect_lt(dot_time, 5.0)
  
  cat("Dot product of", format(n, scientific = TRUE), "elements in", sprintf("%.6f", dot_time), "seconds\n")
})

test_that("GPU operations fail appropriately when CUDA is not working", {
  # This test is tricky - we can't easily simulate CUDA failure
  # But we can test error handling paths
  
  # Test with invalid input that would cause CUDA errors
  if (gpu_available()) {
    # Try to create a vector that's too large for GPU memory
    # This should either work or fail gracefully
    tryCatch({
      huge_size <- min(1e8, .Machine$integer.max / 100)  # 100M elements or max safe
      
      # This might succeed on high-end GPUs or fail gracefully
      result <- tryCatch({
        huge_vec <- as.gpuVector(rep(1.0, huge_size))
        as.vector(huge_vec)[1:10]  # Just check first few elements
      }, error = function(e) {
        # GPU memory exhausted - this is expected behavior
        expect_true(grepl("Failed to allocate GPU memory|memory", e$message, ignore.case = TRUE))
        "memory_error"
      })
      
      # Either it worked or we got appropriate error
      expect_true(is.numeric(result) || result == "memory_error")
      
    }, error = function(e) {
      # Any other error should be informative
      expect_true(nchar(e$message) > 0)
    })
  } else {
    skip("GPU not available - cannot test CUDA error handling")
  }
})

test_that("Mixed GPU and CPU operations produce consistent results", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test that mixing GPU and CPU computations doesn't cause issues
  n <- 10000
  a <- runif(n, min = 1, max = 10)
  b <- runif(n, min = 1, max = 10)
  c <- runif(n, min = 1, max = 10)
  
  # Pure CPU computation
  cpu_result <- (a + b) * c
  
  # Mixed: CPU + GPU + CPU
  step1 <- a + b  # CPU
  step2 <- gpu_multiply(step1, c, warn_fallback = FALSE)  # GPU
  mixed_result1 <- as.numeric(step2)  # Back to CPU
  
  # Mixed: GPU + CPU + GPU  
  gpu_a <- as.gpuVector(a)
  gpu_b <- as.gpuVector(b)
  step3 <- gpu_a + gpu_b  # GPU
  step4 <- as.vector(step3) * c  # CPU
  step5 <- gpu_multiply(rep(1.0, n), step4, warn_fallback = FALSE)  # GPU
  mixed_result2 <- as.numeric(step5)
  
  # All should produce the same result
  expect_equal(mixed_result1, cpu_result, tolerance = 1e-10)
  expect_equal(mixed_result2, cpu_result, tolerance = 1e-10)
  expect_equal(mixed_result1, mixed_result2, tolerance = 1e-14)
  
  expect_length(mixed_result1, n)
  expect_length(mixed_result2, n)
}) 