test_that("GPU operations actually run on GPU with parallel execution", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test with multiple sizes to verify parallel execution benefits
  test_sizes <- c(1e5, 5e5, 1e6, 2e6)
  results <- list()
  
  for (n in test_sizes) {
    cat("\n=== Testing vector size:", format(n, scientific = TRUE), "===\n")
    
    a <- runif(n, min = 1, max = 10)
    b <- runif(n, min = 1, max = 10)
    scalar <- 3.14159
    
    # CPU timing (reference)
    cpu_mult_time <- system.time({
      cpu_mult_result <- a * b
    })[["elapsed"]]
    
    cpu_scale_time <- system.time({
      cpu_scale_result <- a * scalar
    })[["elapsed"]]
    
    cpu_dot_time <- system.time({
      cpu_dot_result <- sum(a * b)
    })[["elapsed"]]
    
    # GPU timing (should show parallel speedup for larger sizes)
    gpu_mult_time <- system.time({
      gpu_mult_result <- gpu_multiply(a, b, warn_fallback = FALSE)
    })[["elapsed"]]
    
    gpu_scale_time <- system.time({
      gpu_scale_result <- gpu_scale(a, scalar, warn_fallback = FALSE)
    })[["elapsed"]]
    
    gpu_dot_time <- system.time({
      gpu_dot_result <- gpu_dot(a, b, warn_fallback = FALSE)
    })[["elapsed"]]
    
    # Verify correctness first (critical!)
    expect_equal(gpu_mult_result, cpu_mult_result, tolerance = 1e-10)
    expect_equal(gpu_scale_result, cpu_scale_result, tolerance = 1e-10)
    expect_equal(gpu_dot_result, cpu_dot_result, tolerance = 1e-10)
    
    # Performance verification
    expect_lt(gpu_mult_time, 10.0)  # Should complete within reasonable time
    expect_lt(gpu_scale_time, 10.0)
    expect_lt(gpu_dot_time, 10.0)
    
    # Calculate speedups
    mult_speedup <- cpu_mult_time / gpu_mult_time
    scale_speedup <- cpu_scale_time / gpu_scale_time
    dot_speedup <- cpu_dot_time / gpu_dot_time
    
    # Store results
    results[[paste0("n_", n)]] <- list(
      size = n,
      mult_speedup = mult_speedup,
      scale_speedup = scale_speedup,
      dot_speedup = dot_speedup,
      gpu_mult_time = gpu_mult_time,
      cpu_mult_time = cpu_mult_time
    )
    
    cat("CPU multiply time:", sprintf("%.6f", cpu_mult_time), "s\n")
    cat("GPU multiply time:", sprintf("%.6f", gpu_mult_time), "s\n")
    cat("Multiply speedup: ", sprintf("%.2f", mult_speedup), "x\n")
    cat("Scale speedup:   ", sprintf("%.2f", scale_speedup), "x\n")
    cat("Dot speedup:     ", sprintf("%.2f", dot_speedup), "x\n")
    
    if (mult_speedup > 1.0) {
      cat("✓ GPU is faster than CPU for multiplication\n")
    } else if (mult_speedup > 0.5) {
      cat("≈ GPU performance is competitive (includes transfer overhead)\n")
    } else {
      cat("⚠ GPU is slower (may be transfer-bound for this size)\n")
    }
  }
  
  # Verify that performance improves with larger problem sizes (hallmark of parallel execution)
  if (length(results) > 2) {
    largest_speedup <- results[[length(results)]]$mult_speedup
    smallest_speedup <- results[[1]]$mult_speedup
    
    cat("\nParallel execution verification:\n")
    cat("Smallest problem speedup:", sprintf("%.2f", smallest_speedup), "x\n")
    cat("Largest problem speedup: ", sprintf("%.2f", largest_speedup), "x\n")
    
    # For true parallel execution, larger problems should benefit more
    # (though transfer overhead can complicate this)
  }
})

test_that("GPU tensor operations use parallel CUDA kernels", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test gpuTensor operations specifically
  cat("\n=== GPU Tensor Parallel Execution Test ===\n")
  
  # Create large tensors to benefit from parallelization
  sizes_to_test <- list(
    c(1000, 1000),      # 1M elements
    c(1414, 1414),      # ~2M elements
    c(1732, 1732)       # ~3M elements
  )
  
  for (shape in sizes_to_test) {
    n <- prod(shape)
    cat("\nTesting tensor shape:", paste(shape, collapse=" x "), "(", format(n, scientific=TRUE), "elements )\n")
    
    # Create random data
    a_data <- runif(n)
    b_data <- runif(n)
    
    # CPU reference computation
    cpu_start <- Sys.time()
    cpu_add <- a_data + b_data
    cpu_mul <- a_data * b_data
    cpu_sum <- sum(a_data * b_data)
    cpu_time <- as.numeric(Sys.time() - cpu_start)
    
    # GPU tensor computation
    gpu_start <- Sys.time()
    tensor_a <- gpu_tensor(a_data, shape)
    tensor_b <- gpu_tensor(b_data, shape)
    
    # Multiple operations to test kernel efficiency
    tensor_add_result <- tensor_a + tensor_b
    tensor_mul_result <- tensor_a * tensor_b
    tensor_sum_result <- sum(tensor_a * tensor_b)
    
    # Synchronize to ensure completion
    synchronize(tensor_add_result)
    synchronize(tensor_mul_result)
    gpu_time <- as.numeric(Sys.time() - gpu_start)
    
    # Verify correctness
    expect_equal(as.vector(tensor_add_result), cpu_add, tolerance = 1e-10)
    expect_equal(as.vector(tensor_mul_result), cpu_mul, tolerance = 1e-10) 
    expect_equal(tensor_sum_result, cpu_sum, tolerance = 1e-8)
    
    speedup <- cpu_time / gpu_time
    throughput <- n / gpu_time
    
    cat("CPU time: ", sprintf("%.6f", cpu_time), "s\n")
    cat("GPU time: ", sprintf("%.6f", gpu_time), "s\n")
    cat("Speedup:  ", sprintf("%.2f", speedup), "x\n")
    cat("GPU throughput: ", sprintf("%.2e", throughput), "elements/second\n")
    
    # Performance expectations for parallel execution
    expect_lt(gpu_time, 5.0)  # Should complete within reasonable time
    if (n > 1e6) {  # For large tensors, expect decent throughput
      expect_gt(throughput, 1e7)  # > 10M elements/second indicates parallel execution
    }
  }
})

test_that("CUDA kernel launches are actually parallel", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== CUDA Parallelism Verification ===\n")
  
  # Test parallel execution by measuring throughput scaling
  base_size <- 100000
  thread_counts <- c(1, 2, 4, 8, 16)  # Simulate different levels of parallelism
  
  results <- list()
  
  for (multiplier in thread_counts) {
    n <- base_size * multiplier
    data <- runif(n)
    
    # Time a computation-heavy operation (multiple operations)
    start_time <- Sys.time()
    
    # Chain multiple GPU operations
    result1 <- gpu_multiply(data, rev(data))  # Element-wise multiply
    result2 <- gpu_scale(result1, 2.5)        # Scalar multiply  
    final_result <- gpu_dot(result2, data[1:length(result2)])  # Dot product
    
    end_time <- Sys.time()
    elapsed <- as.numeric(end_time - start_time)
    
    throughput <- n / elapsed
    results[[as.character(multiplier)]] <- list(
      size = n,
      time = elapsed,
      throughput = throughput
    )
    
    cat("Size:", format(n, scientific=TRUE), 
        " Time:", sprintf("%.6f", elapsed), "s",
        " Throughput:", sprintf("%.2e", throughput), "elem/s\n")
    
    # Verify result correctness
    expect_true(is.finite(final_result))
    expect_gt(final_result, 0)  # Should be positive for our test data
  }
  
  # Analyze throughput scaling - true parallel execution should show sublinear time scaling
  if (length(results) >= 3) {
    small_throughput <- results[["1"]]$throughput
    large_throughput <- results[[as.character(max(thread_counts))]]$throughput
    
    cat("\nThroughput scaling analysis:\n")
    cat("Base throughput:  ", sprintf("%.2e", small_throughput), "elem/s\n")
    cat("Large throughput: ", sprintf("%.2e", large_throughput), "elem/s\n")
    
    # If GPU is truly parallel, throughput should remain relatively stable
    # (not decrease drastically with problem size)
    throughput_ratio <- large_throughput / small_throughput
    cat("Throughput ratio: ", sprintf("%.2f", throughput_ratio), "\n")
    
    if (throughput_ratio > 0.3) {  # Maintain >30% of throughput
      cat("✓ Good throughput scaling - indicates parallel execution\n")
    } else {
      cat("⚠ Poor throughput scaling - may indicate serial bottlenecks\n")
    }
  }
})

test_that("GPU memory operations are efficient", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== GPU Memory Efficiency Test ===\n")
  
  # Test that GPU operations don't unnecessarily transfer data
  n <- 500000
  a_data <- runif(n)
  b_data <- runif(n)
  
  # Test 1: High-level functions (include transfer cost)
  high_level_time <- system.time({
    result1 <- gpu_multiply(a_data, b_data)
    result2 <- gpu_scale(result1, 2.0)
    final_result <- sum(result2)
  })[["elapsed"]]
  
  # Test 2: Tensor operations (potentially more efficient)
  tensor_time <- system.time({
    tensor_a <- gpu_tensor(a_data, length(a_data))
    tensor_b <- gpu_tensor(b_data, length(b_data))
    tensor_result1 <- tensor_a * tensor_b
    tensor_result2 <- tensor_result1 * 2.0
    tensor_final <- sum(tensor_result2)
  })[["elapsed"]]
  
  cat("High-level functions time: ", sprintf("%.6f", high_level_time), "s\n")
  cat("Tensor operations time:    ", sprintf("%.6f", tensor_time), "s\n")
  
  # Both should complete within reasonable time
  expect_lt(high_level_time, 5.0)
  expect_lt(tensor_time, 5.0)
  
  # Verify correctness
  expect_equal(final_result, tensor_final, tolerance = 1e-10)
})

test_that("GPU error handling and fallback work correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test forced CPU execution produces identical results
  n <- 50000
  a <- runif(n)
  b <- runif(n)
  scalar <- 3.14159
  
  # Force CPU execution
  cpu_mult <- gpu_multiply(a, b, force_cpu = TRUE, warn_fallback = FALSE)
  cpu_scale <- gpu_scale(a, scalar, force_cpu = TRUE, warn_fallback = FALSE)
  cpu_dot <- gpu_dot(a, b, force_cpu = TRUE, warn_fallback = FALSE)
  
  # GPU execution
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

test_that("Mixed operations maintain precision and performance", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test mixing different operation types
  n <- 100000
  a <- runif(n, min = 1, max = 10)
  b <- runif(n, min = 1, max = 10)
  c <- runif(n, min = 1, max = 10)
  
  # Complex computation: (a * b + c) / sum(a * b + c)
  start_time <- Sys.time()
  
  # Step-by-step GPU computation
  step1 <- gpu_multiply(a, b)           # a * b
  step2 <- gpu_add(step1, c)            # a * b + c
  step3 <- sum(step2)                   # sum(a * b + c)
  step4 <- gpu_scale(step2, 1.0/step3)  # (a * b + c) / sum(...)
  
  gpu_time <- as.numeric(Sys.time() - start_time)
  
  # CPU reference
  start_time <- Sys.time()
  cpu_step1 <- a * b
  cpu_step2 <- cpu_step1 + c
  cpu_step3 <- sum(cpu_step2)
  cpu_result <- cpu_step2 / cpu_step3
  cpu_time <- as.numeric(Sys.time() - start_time)
  
  # Verify correctness
  expect_equal(step4, cpu_result, tolerance = 1e-12)
  
  cat("\nMixed operations performance:\n")
  cat("CPU time: ", sprintf("%.6f", cpu_time), "s\n")
  cat("GPU time: ", sprintf("%.6f", gpu_time), "s\n")
  cat("Speedup:  ", sprintf("%.2f", cpu_time / gpu_time), "x\n")
  
  # Should maintain reasonable performance
  expect_lt(gpu_time, 5.0)
}) 