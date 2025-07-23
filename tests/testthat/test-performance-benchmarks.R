test_that("Enhanced GPU performance benchmarks with verification", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== ENHANCED GPU PERFORMANCE BENCHMARKS ===\n")
  
  # Test different problem sizes to identify GPU advantage threshold
  test_sizes <- c(1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6)
  operations <- c("add", "multiply", "scalar_multiply", "matmul", "reductions")
  results_df <- data.frame()
  
  for (n in test_sizes) {
    cat(sprintf("\n>>> TESTING PROBLEM SIZE: %s ELEMENTS <<<\n", 
                format(n, scientific=TRUE)))
    
    # Generate reproducible test data
    set.seed(42)
    a <- runif(n, -10, 10)
    b <- runif(n, -5, 15)
    scalar <- pi
    
    # === ELEMENT-WISE ADDITION BENCHMARK ===
    cat("• Element-wise Addition: ")
    # Calculate results for correctness check
    cpu_add_result <- a + b
    tensor_a <- as_tensor(a, dtype = "float32")
    tensor_b <- as_tensor(b, dtype = "float32") 
    gpu_add_result <- as.array(tensor_a + tensor_b)
    
    # Time CPU operation
    cpu_add_times <- replicate(5, system.time({
      result <- a + b
    })[["elapsed"]])
    cpu_add_time <- median(cpu_add_times)
    
    # Time GPU operation
    gpu_add_times <- replicate(5, system.time({
      tensor_a_temp <- as_tensor(a, dtype = "float32")
      tensor_b_temp <- as_tensor(b, dtype = "float32")
      result <- as.array(tensor_a_temp + tensor_b_temp)
    })[["elapsed"]])
    gpu_add_time <- median(gpu_add_times)
    
    # Verify correctness
    expect_equal(gpu_add_result, cpu_add_result, tolerance = 1e-6)
    
    add_speedup <- cpu_add_time / gpu_add_time
    add_throughput <- n / gpu_add_time / 1e9  # Giga-elements per second
    
    cat(sprintf("GPU: %.4fs, CPU: %.4fs, Speedup: %.2fx, Throughput: %.2f GE/s\n",
                gpu_add_time, cpu_add_time, add_speedup, add_throughput))
    
    # === ELEMENT-WISE MULTIPLICATION BENCHMARK ===
    cat("• Element-wise Multiplication: ")
    cpu_mul_times <- replicate(5, system.time({
      cpu_mul_result <- a * b
    })[["elapsed"]])
    cpu_mul_time <- median(cpu_mul_times)
    
    gpu_mul_times <- replicate(5, system.time({
      tensor_a <- as_tensor(a, dtype = "float32")
      tensor_b <- as_tensor(b, dtype = "float32") 
      gpu_mul_result_tensor <- tensor_a * tensor_b
      gpu_mul_result <- as.array(gpu_mul_result_tensor)
    })[["elapsed"]])
    gpu_mul_time <- median(gpu_mul_times)
    
    # Verify correctness
    expect_equal(gpu_mul_result, cpu_mul_result, tolerance = 1e-6)
    
    mul_speedup <- cpu_mul_time / gpu_mul_time
    mul_throughput <- n / gpu_mul_time / 1e9
    
    cat(sprintf("GPU: %.4fs, CPU: %.4fs, Speedup: %.2fx, Throughput: %.2f GE/s\n",
                gpu_mul_time, cpu_mul_time, mul_speedup, mul_throughput))
    
    # === SCALAR MULTIPLICATION BENCHMARK ===
    cat("• Scalar Multiplication: ")
    cpu_scalar_times <- replicate(5, system.time({
      cpu_scalar_result <- a * scalar
    })[["elapsed"]])
    cpu_scalar_time <- median(cpu_scalar_times)
    
    gpu_scalar_times <- replicate(5, system.time({
      tensor_a <- as_tensor(a, dtype = "float32")
      gpu_scalar_result_tensor <- tensor_a * scalar
      gpu_scalar_result <- as.array(gpu_scalar_result_tensor)
    })[["elapsed"]])
    gpu_scalar_time <- median(gpu_scalar_times)
    
    # Verify correctness
    expect_equal(gpu_scalar_result, cpu_scalar_result, tolerance = 1e-6)
    
    scalar_speedup <- cpu_scalar_time / gpu_scalar_time
    scalar_throughput <- n / gpu_scalar_time / 1e9
    
    cat(sprintf("GPU: %.4fs, CPU: %.4fs, Speedup: %.2fx, Throughput: %.2f GE/s\n",
                gpu_scalar_time, cpu_scalar_time, scalar_speedup, scalar_throughput))
    
    # === REDUCTION OPERATIONS BENCHMARK ===
    cat("• Sum Reduction: ")
    cpu_sum_times <- replicate(5, system.time({
      cpu_sum_result <- sum(a)
    })[["elapsed"]])
    cpu_sum_time <- median(cpu_sum_times)
    
    gpu_sum_times <- replicate(5, system.time({
      tensor_a <- as_tensor(a, dtype = "float32")
      gpu_sum_result <- sum(tensor_a)
    })[["elapsed"]])
    gpu_sum_time <- median(gpu_sum_times)
    
    # Verify correctness
    expect_equal(gpu_sum_result, cpu_sum_result, tolerance = 1e-5)
    
    sum_speedup <- cpu_sum_time / gpu_sum_time
    sum_throughput <- n / gpu_sum_time / 1e9
    
    cat(sprintf("GPU: %.4fs, CPU: %.4fs, Speedup: %.2fx, Throughput: %.2f GE/s\n",
                gpu_sum_time, cpu_sum_time, sum_speedup, sum_throughput))
    
    # === MATRIX MULTIPLICATION BENCHMARK (for square matrices) ===
    if (n <= 1e6) {  # Only test matmul for reasonable sizes
      mat_size <- floor(sqrt(n))
      if (mat_size >= 10) {  # Only test if we have at least 10x10 matrices
        cat(sprintf("• Matrix Multiplication (%dx%d): ", mat_size, mat_size))
        
        A_data <- matrix(runif(mat_size^2, -1, 1), nrow = mat_size)
        B_data <- matrix(runif(mat_size^2, -1, 1), nrow = mat_size)
        
        cpu_matmul_times <- replicate(3, system.time({
          cpu_matmul_result <- A_data %*% B_data
        })[["elapsed"]])
        cpu_matmul_time <- median(cpu_matmul_times)
        
        if (exists("matmul")) {
          gpu_matmul_times <- replicate(3, system.time({
            A_tensor <- as_tensor(A_data, dtype = "float32")
            B_tensor <- as_tensor(B_data, dtype = "float32")
            gpu_matmul_result_tensor <- matmul(A_tensor, B_tensor)
            gpu_matmul_result <- as.array(gpu_matmul_result_tensor)
          })[["elapsed"]])
          gpu_matmul_time <- median(gpu_matmul_times)
          
          # Verify correctness
          expect_equal(gpu_matmul_result, cpu_matmul_result, tolerance = 1e-4)
          
          matmul_speedup <- cpu_matmul_time / gpu_matmul_time
          # Calculate FLOPS (2*N^3 operations for NxN matrix multiplication)
          flops <- 2 * mat_size^3
          matmul_gflops <- flops / gpu_matmul_time / 1e9
          
          cat(sprintf("GPU: %.4fs, CPU: %.4fs, Speedup: %.2fx, GFLOPS: %.2f\n",
                      gpu_matmul_time, cpu_matmul_time, matmul_speedup, matmul_gflops))
        } else {
          cat("MATMUL function not available\n")
        }
      }
    }
    
    # Store results for analysis
    results_df <- rbind(results_df, data.frame(
      size = n,
      add_speedup = add_speedup,
      mul_speedup = mul_speedup,
      scalar_speedup = scalar_speedup,
      sum_speedup = sum_speedup,
      add_throughput = add_throughput,
      mul_throughput = mul_throughput,
      scalar_throughput = scalar_throughput,
      sum_throughput = sum_throughput
    ))
    
    # Performance assertions
    if (n >= 1e5) {  # For large problems, expect reasonable performance
      expect_lt(gpu_add_time, 1.0, 
                info = sprintf("GPU addition should complete in <1s for %s elements", format(n, scientific=TRUE)))
      expect_lt(gpu_mul_time, 1.0,
                info = sprintf("GPU multiplication should complete in <1s for %s elements", format(n, scientific=TRUE)))
      
      # Expect reasonable throughput (at least 1 GE/s for large problems)
      if (n >= 5e5) {
        expect_gt(add_throughput, 1.0,
                  info = sprintf("GPU addition throughput should be >1 GE/s for %s elements", format(n, scientific=TRUE)))
        expect_gt(mul_throughput, 1.0,
                  info = sprintf("GPU multiplication throughput should be >1 GE/s for %s elements", format(n, scientific=TRUE)))
      }
    }
  }
  
  cat("\n=== PERFORMANCE ANALYSIS SUMMARY ===\n")
  
  # Find crossover points where GPU becomes advantageous
  gpu_advantage_add <- results_df$size[which(results_df$add_speedup > 1.0)[1]]
  gpu_advantage_mul <- results_df$size[which(results_df$mul_speedup > 1.0)[1]]
  
  if (!is.na(gpu_advantage_add)) {
    cat(sprintf("GPU Addition Advantage: %s elements and above\n", 
                format(gpu_advantage_add, scientific=TRUE)))
  }
  if (!is.na(gpu_advantage_mul)) {
    cat(sprintf("GPU Multiplication Advantage: %s elements and above\n", 
                format(gpu_advantage_mul, scientific=TRUE)))
  }
  
  # Peak throughput
  max_add_throughput <- max(results_df$add_throughput)
  max_mul_throughput <- max(results_df$mul_throughput)
  
  cat(sprintf("Peak Addition Throughput: %.2f GE/s\n", max_add_throughput))
  cat(sprintf("Peak Multiplication Throughput: %.2f GE/s\n", max_mul_throughput))
  
  # Memory bandwidth estimation (assuming single-precision floats = 4 bytes)
  # Add operation: reads 2 arrays + writes 1 = 12 bytes per element
  max_add_bandwidth <- max_add_throughput * 12  # GB/s
  cat(sprintf("Estimated Memory Bandwidth (Addition): %.2f GB/s\n", max_add_bandwidth))
  
  cat("\n✓ Performance benchmark tests completed!\n")
})

test_that("Memory usage and scaling verification", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== MEMORY USAGE AND SCALING VERIFICATION ===\n")
  
  # Test memory scaling
  memory_sizes <- c(1e4, 1e5, 5e5, 1e6, 2e6)
  
  for (size in memory_sizes) {
    cat(sprintf("Testing memory allocation for %s elements: ", format(size, scientific=TRUE)))
    
    # Create tensor and measure creation time
    creation_time <- system.time({
      tensor <- as_tensor(runif(size, -1, 1), dtype = "float32")
    })[["elapsed"]]
    
    # Verify tensor properties
    expect_equal(as.numeric(size(tensor)), size)
    expect_true(is_contiguous(tensor))
    
    # Test a simple operation to ensure the tensor is functional
    operation_time <- system.time({
      result <- tensor + 1.0
      final_result <- as.array(result)
    })[["elapsed"]]
    
    expect_equal(length(final_result), size)
    
    # Memory estimation (4 bytes per float32)
    memory_mb <- size * 4 / 1024 / 1024
    
    cat(sprintf("Create: %.4fs, Op: %.4fs, Memory: %.1f MB\n",
                creation_time, operation_time, memory_mb))
    
    # Performance expectations
    expect_lt(creation_time, 2.0) # Tensor creation should be fast
    expect_lt(operation_time, 2.0) # Simple operations should be fast
  }
  
  cat("\n✓ Memory scaling tests passed!\n")
}) 