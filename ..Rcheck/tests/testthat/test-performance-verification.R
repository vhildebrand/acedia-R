test_that("Comprehensive GPU vs CPU runtime comparison", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== COMPREHENSIVE GPU vs CPU PERFORMANCE BENCHMARKS ===\n")
  
  # Test different problem sizes to identify GPU advantage threshold
  sizes <- c(1e3, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6)
  operations <- c("multiply", "add", "scale", "dot")
  
  results <- data.frame()
  
  for (n in sizes) {
    if (n > 2e5) {
      cat("\n>>> Testing large problem size:", format(n, scientific=TRUE), "elements <<<\n")
    } else {
      cat("\n>> Testing size:", format(n, scientific=TRUE), "elements <<\n")
    }
    
    # Generate test data
    set.seed(42)  # Reproducible results
    a <- runif(n, -10, 10)
    b <- runif(n, -5, 15)
    scalar <- pi
    
    # === MULTIPLICATION BENCHMARK ===
    cpu_mult_times <- replicate(3, system.time(a * b)[["elapsed"]])
    cpu_mult_time <- median(cpu_mult_times)
    
    gpu_mult_times <- replicate(3, system.time(gpu_multiply(a, b, warn_fallback = FALSE))[["elapsed"]])
    gpu_mult_time <- median(gpu_mult_times)
    
    mult_speedup <- cpu_mult_time / gpu_mult_time
    mult_throughput <- n / gpu_mult_time
    
    # === ADDITION BENCHMARK ===  
    cpu_add_times <- replicate(3, system.time(a + b)[["elapsed"]])
    cpu_add_time <- median(cpu_add_times)
    
    gpu_add_times <- replicate(3, system.time(gpu_add(a, b, warn_fallback = FALSE))[["elapsed"]])
    gpu_add_time <- median(gpu_add_times)
    
    add_speedup <- cpu_add_time / gpu_add_time
    add_throughput <- n / gpu_add_time
    
    # === SCALING BENCHMARK ===
    cpu_scale_times <- replicate(3, system.time(a * scalar)[["elapsed"]])
    cpu_scale_time <- median(cpu_scale_times)
    
    gpu_scale_times <- replicate(3, system.time(gpu_scale(a, scalar, warn_fallback = FALSE))[["elapsed"]])
    gpu_scale_time <- median(gpu_scale_times)
    
    scale_speedup <- cpu_scale_time / gpu_scale_time
    scale_throughput <- n / gpu_scale_time
    
    # === DOT PRODUCT BENCHMARK ===
    cpu_dot_times <- replicate(3, system.time(sum(a * b))[["elapsed"]])
    cpu_dot_time <- median(cpu_dot_times)
    
    gpu_dot_times <- replicate(3, system.time(gpu_dot(a, b, warn_fallback = FALSE))[["elapsed"]])
    gpu_dot_time <- median(gpu_dot_times)
    
    dot_speedup <- cpu_dot_time / gpu_dot_time
    dot_throughput <- n / gpu_dot_time
    
    # Store results
    results <- rbind(results, 
      data.frame(
        size = n,
        operation = "multiply",
        cpu_time = cpu_mult_time,
        gpu_time = gpu_mult_time,
        speedup = mult_speedup,
        throughput = mult_throughput
      ),
      data.frame(
        size = n,
        operation = "add", 
        cpu_time = cpu_add_time,
        gpu_time = gpu_add_time,
        speedup = add_speedup,
        throughput = add_throughput
      ),
      data.frame(
        size = n,
        operation = "scale",
        cpu_time = cpu_scale_time,
        gpu_time = gpu_scale_time,
        speedup = scale_speedup,
        throughput = scale_throughput
      ),
      data.frame(
        size = n,
        operation = "dot",
        cpu_time = cpu_dot_time,
        gpu_time = gpu_dot_time,
        speedup = dot_speedup,
        throughput = dot_throughput
      )
    )
    
    # Print summary for this size
    cat("Multiply: CPU=", sprintf("%.6f", cpu_mult_time), "s, GPU=", sprintf("%.6f", gpu_mult_time), 
        "s, Speedup=", sprintf("%.2f", mult_speedup), "x\n")
    cat("Add:      CPU=", sprintf("%.6f", cpu_add_time), "s, GPU=", sprintf("%.6f", gpu_add_time),
        "s, Speedup=", sprintf("%.2f", add_speedup), "x\n")
    cat("Scale:    CPU=", sprintf("%.6f", cpu_scale_time), "s, GPU=", sprintf("%.6f", gpu_scale_time),
        "s, Speedup=", sprintf("%.2f", scale_speedup), "x\n")
    cat("Dot:      CPU=", sprintf("%.6f", cpu_dot_time), "s, GPU=", sprintf("%.6f", gpu_dot_time),
        "s, Speedup=", sprintf("%.2f", dot_speedup), "x\n")
    
    # Verify correctness at each size
    mult_result <- gpu_multiply(a[1:min(100, n)], b[1:min(100, n)])
    expect_equal(mult_result, (a * b)[1:min(100, n)], tolerance = 1e-12)
    
    add_result <- gpu_add(a[1:min(100, n)], b[1:min(100, n)])
    expect_equal(add_result, (a + b)[1:min(100, n)], tolerance = 1e-12)
  }
  
  # === PERFORMANCE ANALYSIS ===
  cat("\n=== PERFORMANCE ANALYSIS ===\n")
  
  # Find GPU advantage threshold for each operation
  for (op in operations) {
    op_results <- results[results$operation == op, ]
    gpu_wins <- op_results[op_results$speedup > 1.0, ]
    
    if (nrow(gpu_wins) > 0) {
      threshold <- min(gpu_wins$size)
      max_speedup <- max(op_results$speedup)
      cat("Operation '", op, "': GPU faster than CPU starting at size ", 
          format(threshold, scientific=TRUE), 
          " (max speedup: ", sprintf("%.2f", max_speedup), "x)\n", sep="")
    } else {
      cat("Operation '", op, "': CPU faster across all test sizes (transfer overhead)\n", sep="")
    }
  }
  
  # Verify throughput scaling (indicates parallel execution)
  cat("\nThroughput scaling analysis:\n")
  large_results <- results[results$size >= 1e5, ]
  for (op in operations) {
    op_large <- large_results[large_results$operation == op, ]
    if (nrow(op_large) > 0) {
      avg_throughput <- mean(op_large$throughput)
      cat("Operation '", op, "': Average throughput ", 
          sprintf("%.2e", avg_throughput), " elements/second\n", sep="")
      
      # Good throughput indicates parallel execution
      if (avg_throughput > 1e7) {  # > 10M elements/second
        cat("  ✓ High throughput suggests parallel GPU execution\n")
      } else if (avg_throughput > 1e6) {
        cat("  ≈ Moderate throughput - may include transfer overhead\n")  
      } else {
        cat("  ⚠ Low throughput - may indicate serial bottlenecks\n")
      }
    }
  }
  
  # Performance requirements
  expect_true(nrow(results) > 0)
  expect_true(all(results$gpu_time > 0))
  expect_true(all(results$cpu_time > 0))
  expect_true(all(is.finite(results$speedup)))
  
  # For large problems, GPU should maintain reasonable performance
  large_gpu_times <- results[results$size >= 1e6, "gpu_time"]
  if (length(large_gpu_times) > 0) {
    expect_true(all(large_gpu_times < 5.0))  # Should complete within 5 seconds
  }
})

test_that("GPU tensor operations scale with parallel execution", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== GPU TENSOR PARALLEL SCALING TEST ===\n")
  
  # Test different tensor shapes to verify 2D parallelization
  shapes <- list(
    c(100, 100),     # 10K elements
    c(316, 316),     # 100K elements  
    c(707, 707),     # 500K elements
    c(1000, 1000),   # 1M elements
    c(1414, 1414),   # 2M elements
    c(2000, 1000)    # 2M elements (different aspect ratio)
  )
  
  perf_results <- list()
  
  for (i in seq_along(shapes)) {
    shape <- shapes[[i]]
    n <- prod(shape)
    
    cat("\nTesting tensor shape: [", paste(shape, collapse=" x "), "] = ", 
        format(n, scientific=TRUE), " elements\n")
    
    # Generate test data
    a_data <- runif(n, -1, 1)
    b_data <- runif(n, -1, 1)
    
    # CPU baseline (1D operations)
    cpu_time <- system.time({
      cpu_add <- a_data + b_data
      cpu_mult <- a_data * b_data
      cpu_sum <- sum(a_data * b_data)
    })[["elapsed"]]
    
    # GPU tensor operations
    gpu_time <- system.time({
      tensor_a <- gpu_tensor(a_data, shape)
      tensor_b <- gpu_tensor(b_data, shape)
      
      tensor_add <- tensor_a + tensor_b
      tensor_mult <- tensor_a * tensor_b
      tensor_sum <- sum(tensor_a * tensor_b)
      
      # Synchronize to ensure completion
      synchronize(tensor_add)
      synchronize(tensor_mult)
    })[["elapsed"]]
    
    # Memory bandwidth estimation (rough)
    data_bytes <- n * 8 * 4  # 4 operations * 8 bytes per double
    bandwidth_gbps <- (data_bytes / gpu_time) / 1e9
    
    speedup <- cpu_time / gpu_time
    throughput <- n / gpu_time
    
    perf_results[[i]] <- list(
      shape = shape,
      size = n,
      cpu_time = cpu_time,
      gpu_time = gpu_time,
      speedup = speedup,
      throughput = throughput,
      bandwidth_gbps = bandwidth_gbps
    )
    
    cat("CPU time:    ", sprintf("%.6f", cpu_time), " s\n")
    cat("GPU time:    ", sprintf("%.6f", gpu_time), " s\n") 
    cat("Speedup:     ", sprintf("%.2f", speedup), "x\n")
    cat("Throughput:  ", sprintf("%.2e", throughput), " elem/s\n")
    cat("Bandwidth:   ", sprintf("%.2f", bandwidth_gbps), " GB/s\n")
    
    # Verify correctness
    expect_equal(as.vector(tensor_add), cpu_add, tolerance = 1e-12)
    expect_equal(as.vector(tensor_mult), cpu_mult, tolerance = 1e-12)
    expect_equal(tensor_sum, cpu_sum, tolerance = 1e-10)
    
    # Performance expectations
    expect_lt(gpu_time, 10.0)  # Should complete within 10 seconds
    if (n >= 1e6) {
      expect_gt(throughput, 1e6)  # At least 1M elements/second for large tensors
    }
  }
  
  # Analyze scaling behavior
  cat("\n=== PARALLEL SCALING ANALYSIS ===\n")
  
  if (length(perf_results) >= 3) {
    # Compare small vs large problem performance
    small_result <- perf_results[[1]]
    large_result <- perf_results[[length(perf_results)]]
    
    size_ratio <- large_result$size / small_result$size
    time_ratio <- large_result$gpu_time / small_result$gpu_time
    throughput_ratio <- large_result$throughput / small_result$throughput
    
    cat("Problem size increased by: ", sprintf("%.1f", size_ratio), "x\n")
    cat("GPU time increased by:     ", sprintf("%.1f", time_ratio), "x\n")
    cat("Throughput changed by:     ", sprintf("%.1f", throughput_ratio), "x\n")
    
    # Ideal parallel scaling: time should scale less than linearly with problem size
    if (time_ratio < size_ratio * 0.8) {
      cat("✓ Sublinear time scaling - indicates good parallel execution\n")
    } else if (time_ratio < size_ratio * 1.2) {
      cat("≈ Near-linear time scaling - moderate parallel efficiency\n")
    } else {
      cat("⚠ Super-linear time scaling - may indicate memory bottlenecks\n")
    }
    
    # Check for consistent throughput (indicates parallel efficiency)
    if (throughput_ratio > 0.5) {
      cat("✓ Throughput maintained well across problem sizes\n")
    } else {
      cat("⚠ Throughput drops significantly for larger problems\n") 
    }
  }
})

test_that("Memory bandwidth utilization indicates parallel execution", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== MEMORY BANDWIDTH ANALYSIS ===\n")
  
  # Test operations with different computational intensities
  test_cases <- list(
    list(name = "Simple Add", n = 2e6, ops = 1),
    list(name = "Multiply", n = 2e6, ops = 1), 
    list(name = "Fused Mult-Add", n = 1e6, ops = 2),
    list(name = "Complex Chain", n = 5e5, ops = 4)
  )
  
  for (test_case in test_cases) {
    n <- test_case$n
    ops <- test_case$ops
    name <- test_case$name
    
    cat("\n", name, " test (", format(n, scientific=TRUE), " elements):\n", sep="")
    
    # Generate data
    a <- runif(n)
    b <- runif(n)
    
    if (name == "Simple Add") {
      gpu_time <- system.time({
        result <- gpu_add(a, b)
      })[["elapsed"]]
      
      cpu_time <- system.time({
        cpu_result <- a + b
      })[["elapsed"]]
      
    } else if (name == "Multiply") {
      gpu_time <- system.time({
        result <- gpu_multiply(a, b)
      })[["elapsed"]]
      
      cpu_time <- system.time({
        cpu_result <- a * b
      })[["elapsed"]]
      
    } else if (name == "Fused Mult-Add") {
      gpu_time <- system.time({
        temp <- gpu_multiply(a, b)
        result <- gpu_add(temp, a)
      })[["elapsed"]]
      
      cpu_time <- system.time({
        cpu_result <- a * b + a
      })[["elapsed"]]
      
    } else if (name == "Complex Chain") {
      scalar <- 2.5
      gpu_time <- system.time({
        temp1 <- gpu_multiply(a, b)
        temp2 <- gpu_scale(temp1, scalar)  
        temp3 <- gpu_add(temp2, a)
        result <- gpu_dot(temp3, b)
      })[["elapsed"]]
      
      cpu_time <- system.time({
        cpu_result <- sum((a * b * scalar + a) * b)
      })[["elapsed"]]
    }
    
    # Estimate memory bandwidth
    # Each double is 8 bytes, assume 3 memory operations per arithmetic op (2 reads + 1 write)
    bytes_transferred <- n * 8 * 3 * ops
    gpu_bandwidth_gbps <- bytes_transferred / gpu_time / 1e9
    cpu_bandwidth_gbps <- bytes_transferred / cpu_time / 1e9
    
    speedup <- cpu_time / gpu_time
    compute_rate <- (n * ops) / gpu_time
    
    cat("CPU time:        ", sprintf("%.6f", cpu_time), " s\n")
    cat("GPU time:        ", sprintf("%.6f", gpu_time), " s\n")
    cat("Speedup:         ", sprintf("%.2f", speedup), "x\n")
    cat("GPU bandwidth:   ", sprintf("%.1f", gpu_bandwidth_gbps), " GB/s\n")
    cat("Compute rate:    ", sprintf("%.2e", compute_rate), " ops/s\n")
    
    # Modern GPUs should achieve reasonable bandwidth utilization
    if (gpu_bandwidth_gbps > 50) {
      cat("✓ High bandwidth utilization - indicates parallel memory access\n")
    } else if (gpu_bandwidth_gbps > 10) {
      cat("≈ Moderate bandwidth utilization\n")
    } else {
      cat("⚠ Low bandwidth utilization - may indicate inefficient access patterns\n")
    }
    
    # Performance requirements
    expect_lt(gpu_time, 5.0)  # Should complete within 5 seconds
    expect_gt(compute_rate, 1e6)  # At least 1M operations per second
    
    # Verify numerical correctness
    if (name == "Complex Chain") {
      expect_true(is.finite(result))
      expect_true(is.finite(cpu_result))
      expect_equal(result, cpu_result, tolerance = 1e-10)
    } else {
      expect_equal(result, cpu_result, tolerance = 1e-12)
    }
  }
}) 