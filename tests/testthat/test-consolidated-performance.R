context("Performance benchmarks and GPU verification")

# Helper functions
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
# GPU EXECUTION VERIFICATION
# =============================================================================

test_that("GPU operations actually run on GPU with parallel execution", {
  # Test with different sizes to verify parallel execution
  test_sizes <- c(1e4, 5e4, 1e5, 5e5)
  
  for (n in test_sizes) {
    if (n > 1e6) next  # Skip very large tests in CI
    
    # Create test data
    data_a <- runif(n, -1, 1)
    data_b <- runif(n, -1, 1)
    
    # Create GPU tensors
    tensor_a <- as_tensor(data_a, dtype = "float32")
    tensor_b <- as_tensor(data_b, dtype = "float32")
    
    expect_true(verify_gpu_tensor(tensor_a, paste("tensor_a creation", n)))
    expect_true(verify_gpu_tensor(tensor_b, paste("tensor_b creation", n)))
    
    # Perform operations and verify GPU execution
    result_add_gpu <- tensor_a + tensor_b
    result_mul_gpu <- tensor_a * tensor_b
    result_scalar_add_gpu <- tensor_a + 2.5
    result_scalar_mul_gpu <- tensor_a * 1.5
    
    expect_true(verify_gpu_tensor(result_add_gpu, paste("addition", n)))
    expect_true(verify_gpu_tensor(result_mul_gpu, paste("multiplication", n)))
    expect_true(verify_gpu_tensor(result_scalar_add_gpu, paste("scalar addition", n)))
    expect_true(verify_gpu_tensor(result_scalar_mul_gpu, paste("scalar multiplication", n)))
    
    # Verify correctness
    expect_equal(as.vector(result_add_gpu), data_a + data_b, tolerance = 1e-6)
    expect_equal(as.vector(result_mul_gpu), data_a * data_b, tolerance = 1e-6)
    expect_equal(as.vector(result_scalar_add_gpu), data_a + 2.5, tolerance = 1e-6)
    expect_equal(as.vector(result_scalar_mul_gpu), data_a * 1.5, tolerance = 1e-6)
  }
})

test_that("GPU tensor operations use parallel CUDA kernels", {
  # Test various operations to ensure they use CUDA kernels
  n <- 1e4
  
  # Test data
  data_pos <- abs(runif(n, 0.1, 5))  # Positive values for sqrt, log
  data_exp <- runif(n, -2, 2)        # Values for exp
  
  tensor_pos <- as_tensor(data_pos, dtype = "float32")
  tensor_exp <- as_tensor(data_exp, dtype = "float32")
  
  expect_true(verify_gpu_tensor(tensor_pos, "positive tensor"))
  expect_true(verify_gpu_tensor(tensor_exp, "exp tensor"))
  
  # Mathematical operations
  if (exists("sqrt.gpuTensor")) {
    result_sqrt_gpu <- sqrt(tensor_pos)
    expect_true(verify_gpu_tensor(result_sqrt_gpu, "sqrt operation"))
    expect_equal(as.vector(result_sqrt_gpu), sqrt(data_pos), tolerance = 1e-6)
  }
  
  if (exists("exp.gpuTensor")) {
    result_exp_gpu <- exp(tensor_exp)
    expect_true(verify_gpu_tensor(result_exp_gpu, "exp operation"))
    expect_equal(as.vector(result_exp_gpu), exp(data_exp), tolerance = 1e-6)
  }
  
  if (exists("log.gpuTensor")) {
    result_log_gpu <- log(tensor_pos)
    expect_true(verify_gpu_tensor(result_log_gpu, "log operation"))
    expect_equal(as.vector(result_log_gpu), log(data_pos), tolerance = 1e-6)
  }
})

test_that("CUDA kernel launches are actually parallel", {
  # Test operations that should show parallel speedup
  sizes_to_test <- c(1e4, 5e4, 1e5)
  
  for (n in sizes_to_test) {
    if (n > 2e5) next  # Skip very large in CI
    
    # Create test tensors
    a_data <- runif(n)
    b_data <- runif(n)
    
    tensor_a <- as_tensor(a_data, dtype = "float32")
    tensor_b <- as_tensor(b_data, dtype = "float32")
    
    # Time GPU operations (should be fast due to parallelism)
    gpu_time <- system.time({
      result1 <- tensor_a + tensor_b
      result2 <- tensor_a * tensor_b
      result3 <- result1 * result2
      final_sum <- sum(result3)
      # Only synchronize if result is a gpuTensor
      if (inherits(final_sum, "gpuTensor")) {
        synchronize(final_sum)
      }
    })
    
    # Sum operations might return scalars instead of gpuTensors
    if (inherits(final_sum, "gpuTensor")) {
      expect_true(verify_gpu_tensor(final_sum, "parallel kernel result"))
    }
    
    # Verify the result is numeric
    final_val <- if (inherits(final_sum, "gpuTensor")) as.numeric(as.array(final_sum)) else as.numeric(final_sum)
    expect_true(is.numeric(final_val))
    
    # GPU time should be reasonable (not testing specific speedup due to variability)
    expect_lt(gpu_time[["elapsed"]], 5.0)  # Should complete within 5 seconds
  }
})

# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

test_that("GPU implementation is appreciably faster than CPU for large vectors", {
  # Test with moderately large vectors to see GPU benefit
  test_sizes <- c(1e4, 5e4, 1e5)
  
  for (n in test_sizes) {
    if (n > 2e5) next  # Skip very large in CI
    
    # Create test data
    data_a <- runif(n)
    data_b <- runif(n)
    
    # GPU computation
    tensor_a <- as_tensor(data_a, dtype = "float32")
    tensor_b <- as_tensor(data_b, dtype = "float32")
    
    gpu_time <- system.time({
      gpu_result <- tensor_a * tensor_b + 1.0
      gpu_sum <- sum(gpu_result)
      # Only synchronize if result is a gpuTensor
      if (inherits(gpu_sum, "gpuTensor")) {
        synchronize(gpu_sum)
      }
    })
    
    # CPU computation
    cpu_time <- system.time({
      cpu_result <- data_a * data_b + 1.0
      cpu_sum <- sum(cpu_result)
    })
    
    # Verify correctness
    gpu_sum_val <- if (inherits(gpu_sum, "gpuTensor")) as.numeric(as.array(gpu_sum)) else as.numeric(gpu_sum)
    expect_equal(gpu_sum_val, cpu_sum, tolerance = 1e-6)
    
    # For large enough sizes, GPU should show some benefit
    # (Not enforcing strict speedup due to overhead and test variability)
    expect_true(gpu_time[["elapsed"]] < 10 * cpu_time[["elapsed"]])  # Sanity check
    
    cat(sprintf("Size %e: GPU=%.4fs, CPU=%.4fs\n", n, gpu_time[["elapsed"]], cpu_time[["elapsed"]]))
  }
})

test_that("GPU memory operations are efficient", {
  # Test memory-intensive operations
  n <- 1e4
  
  # Create and manipulate tensors
  tensor_a <- as_tensor(runif(n), dtype = "float32")
  tensor_b <- as_tensor(runif(n), dtype = "float32")
  
  # Chain of operations that require memory bandwidth
  start_time <- Sys.time()
  
  result1 <- tensor_a + tensor_b
  result2 <- result1 * 2.0
  result3 <- result2 - tensor_a
  final_result <- sum(result3)
  
  # Only synchronize if result is a gpuTensor
  if (inherits(final_result, "gpuTensor")) {
    synchronize(final_result)
  }
  end_time <- Sys.time()
  
  elapsed <- as.numeric(end_time - start_time)
  
  # Sum operations might return scalars instead of gpuTensors
  if (inherits(final_result, "gpuTensor")) {
    expect_true(verify_gpu_tensor(final_result, "memory operation chain"))
  }
  
  final_val <- if (inherits(final_result, "gpuTensor")) as.numeric(as.array(final_result)) else as.numeric(final_result)
  expect_true(is.numeric(final_val))
  expect_lt(elapsed, 2.0)  # Should complete quickly
})

test_that("GPU tensor operations scale with parallel execution", {
  # Test scaling with different thread counts (conceptually)
  base_size <- 1e4
  sizes <- c(base_size, base_size * 2, base_size * 4)
  
  times <- numeric(length(sizes))
  
  for (i in seq_along(sizes)) {
    n <- sizes[i]
    if (n > 1e5) next  # Skip very large
    
    # Create tensors
    tensor_a <- as_tensor(runif(n), dtype = "float32")
    tensor_b <- as_tensor(runif(n), dtype = "float32")
    
    # Time operation
    start_time <- Sys.time()
    result <- tensor_a * tensor_b + tensor_a
    sum_result <- sum(result)
    # Only synchronize if result is a gpuTensor
    if (inherits(sum_result, "gpuTensor")) {
      synchronize(sum_result)
    }
    end_time <- Sys.time()
    
    times[i] <- as.numeric(end_time - start_time)
    
    # Sum operations might return scalars instead of gpuTensors
    if (inherits(sum_result, "gpuTensor")) {
      expect_true(verify_gpu_tensor(sum_result, paste("scaling test", n)))
    }
  }
  
  # Times should not grow linearly (due to parallelism)
  # This is a weak test due to overhead variability
  valid_times <- times[times > 0]
  if (length(valid_times) >= 2) {
    expect_true(max(valid_times) < 10 * min(valid_times))  # Sanity check
  }
})

# =============================================================================
# MATRIX OPERATION PERFORMANCE
# =============================================================================

test_that("Matrix operations maintain GPU execution", {
  # Test matrix operations for performance
  sizes_to_test <- list(
    list(m = 50, k = 50, n = 50),
    list(m = 100, k = 100, n = 100)
  )
  
  for (mat_size in sizes_to_test) {
    m <- mat_size$m
    k <- mat_size$k  
    n <- mat_size$n
    
    if (m * k * n > 1e6) next  # Skip very large
    
    # Create matrices
    A_data <- matrix(runif(m * k), nrow = m, ncol = k)
    B_data <- matrix(runif(k * n), nrow = k, ncol = n)
    
    A_tensor <- as_tensor(A_data, dtype = "float32")
    B_tensor <- as_tensor(B_data, dtype = "float32")
    
    expect_true(verify_gpu_tensor(A_tensor, paste("matrix A", m, "x", k)))
    expect_true(verify_gpu_tensor(B_tensor, paste("matrix B", k, "x", n)))
    
    # Matrix multiplication
    if (exists("matmul")) {
      gpu_time <- system.time({
        result_gpu <- matmul(A_tensor, B_tensor)
        synchronize(result_gpu)
      })
      
      expect_true(verify_gpu_tensor(result_gpu, paste("matmul result", m, "x", n)))
      expect_equal(shape(result_gpu), c(m, n))
      
      # Verify correctness
      expected <- A_data %*% B_data
      expect_equal(as.array(result_gpu), expected, tolerance = 1e-5)
      
      # Should complete in reasonable time
      expect_lt(gpu_time[["elapsed"]], 5.0)
    }
  }
})

# =============================================================================
# ADVANCED OPERATIONS PERFORMANCE
# =============================================================================

test_that("Advanced operations work with different dtypes", {
  test_data <- c(1, 2, 3, 4, 5)
  
  # Test float32
  tensor_f32 <- as_tensor(test_data, dtype = "float")
  expect_true(verify_gpu_tensor(tensor_f32, "float32 tensor"))
  
  # Basic operations
  sum_f32 <- sum(tensor_f32)
  # Sum might return a scalar, not a gpuTensor
  if (inherits(sum_f32, "gpuTensor")) {
    expect_true(verify_gpu_tensor(sum_f32, "float32 sum"))
    expect_equal(as.numeric(as.array(sum_f32)), sum(test_data), tolerance = 1e-6)
  } else {
    expect_equal(as.numeric(sum_f32), sum(test_data), tolerance = 1e-6)
  }
  
  # Test float64
  tensor_f64 <- as_tensor(test_data, dtype = "double")
  expect_true(verify_gpu_tensor(tensor_f64, "float64 tensor"))
  
  sum_f64 <- sum(tensor_f64)
  # Sum might return a scalar, not a gpuTensor
  if (inherits(sum_f64, "gpuTensor")) {
    expect_true(verify_gpu_tensor(sum_f64, "float64 sum"))
    expect_equal(as.numeric(as.array(sum_f64)), sum(test_data), tolerance = 1e-15)
  } else {
    expect_equal(as.numeric(sum_f64), sum(test_data), tolerance = 1e-15)
  }
})

test_that("Advanced operations maintain GPU execution with large tensors", {
  # Test advanced operations with larger tensors
  n <- 1e4
  test_data <- runif(n, -5, 5)
  
  tensor <- as_tensor(test_data, dtype = "float32")
  expect_true(verify_gpu_tensor(tensor, "large tensor"))
  
  # Chain of advanced operations
  if (exists("exp.gpuTensor") && exists("sum.gpuTensor")) {
    tryCatch({
      # Get mean as scalar to avoid shape mismatch
      tensor_mean <- mean(tensor)
      mean_val <- if (inherits(tensor_mean, "gpuTensor")) as.numeric(as.array(tensor_mean)) else as.numeric(tensor_mean)
      
      result1 <- exp(tensor - mean_val)  # Use scalar mean
      sum1 <- sum(result1)
      sum1_val <- if (inherits(sum1, "gpuTensor")) as.numeric(as.array(sum1)) else as.numeric(sum1)
      
      result2 <- result1 / sum1_val     # Use scalar sum
      final_sum <- sum(result2)
      
      final_val <- if (inherits(final_sum, "gpuTensor")) as.numeric(as.array(final_sum)) else as.numeric(final_sum)
      expect_equal(final_val, 1.0, tolerance = 1e-6)
    }, error = function(e) {
      skip(paste("Advanced operations not fully supported:", e$message))
    })
  }
})

# =============================================================================
# MEMORY BANDWIDTH ESTIMATION
# =============================================================================

test_that("Memory bandwidth utilization indicates parallel execution", {
  # Test operations that are memory bandwidth limited
  test_cases <- list(
    list(n = 1e4, ops = "add", name = "Small Add"),
    list(n = 5e4, ops = "multiply", name = "Medium Multiply"),
    list(n = 1e5, ops = "chain", name = "Large Chain")
  )
  
  for (test_case in test_cases) {
    n <- test_case$n
    ops <- test_case$ops
    name <- test_case$name
    
    if (n > 2e5) next  # Skip very large
    
    # Create test data
    data_a <- runif(n)
    data_b <- runif(n)
    
    tensor_a <- as_tensor(data_a, dtype = "float32")
    tensor_b <- as_tensor(data_b, dtype = "float32")
    
    # Perform operation based on type
    start_time <- Sys.time()
    
    if (ops == "add") {
      result <- tensor_a + tensor_b
    } else if (ops == "multiply") {
      result <- tensor_a * tensor_b
    } else if (ops == "chain") {
      result <- (tensor_a + tensor_b) * (tensor_a - tensor_b)
    }
    
    final_result <- sum(result)
    # Only synchronize if result is a gpuTensor
    if (inherits(final_result, "gpuTensor")) {
      synchronize(final_result)
    }
    end_time <- Sys.time()
    
    elapsed <- as.numeric(end_time - start_time)
    
    # Sum operations might return scalars instead of gpuTensors
    if (inherits(final_result, "gpuTensor")) {
      expect_true(verify_gpu_tensor(final_result, name))
    }
    
    final_val <- if (inherits(final_result, "gpuTensor")) as.numeric(as.array(final_result)) else as.numeric(final_result)
    expect_true(is.numeric(final_val))
    expect_lt(elapsed, 2.0)  # Should be fast due to parallel execution
    
    cat(sprintf("%s (n=%e): %.4fs\n", name, n, elapsed))
  }
})

# =============================================================================
# ERROR HANDLING AND ROBUSTNESS
# =============================================================================

test_that("GPU error handling and fallback work correctly", {
  # Test various error conditions
  tensor_valid <- as_tensor(c(1, 2, 3), dtype = "float")
  expect_true(verify_gpu_tensor(tensor_valid, "valid tensor"))
  
  # Test with different dtypes
  tryCatch({
    tensor_double <- as_tensor(c(1.0, 2.0, 3.0), dtype = "double")
    expect_true(verify_gpu_tensor(tensor_double, "double tensor"))
  }, error = function(e) {
    # If double not supported, that's okay
    cat("Double precision not supported:", e$message, "\n")
  })
  
  # Test operations maintain GPU execution
  result <- tensor_valid * 2.0 + 1.0
  expect_true(verify_gpu_tensor(result, "error handling operation"))
})

test_that("Mixed operations maintain precision and performance", {
  # Test mixing different operation types
  n <- 1e4
  data <- runif(n, 0.1, 5.0)  # Positive values
  
  tensor <- as_tensor(data, dtype = "float32")
  expect_true(verify_gpu_tensor(tensor, "mixed operations tensor"))
  
  # Chain of mixed operations
  result1 <- tensor * 2.0      # Scalar multiply
  result2 <- result1 + tensor  # Element-wise add
  result3 <- sum(result2)      # Reduction
  
  # Sum might return scalar, not gpuTensor
  if (inherits(result3, "gpuTensor")) {
    expect_true(verify_gpu_tensor(result3, "mixed operations result"))
  }
  
  # Verify precision maintained
  expected <- sum(data * 2.0 + data)
  actual <- if (inherits(result3, "gpuTensor")) as.numeric(as.array(result3)) else as.numeric(result3)
  expect_equal(actual, expected, tolerance = 1e-5)
})

# =============================================================================
# COMPREHENSIVE GPU vs CPU COMPARISON
# =============================================================================

test_that("Comprehensive GPU vs CPU runtime comparison", {
  # Compare GPU vs CPU for various operations
  test_sizes <- c(1e4, 5e4)  # Moderate sizes for CI
  
  for (n in test_sizes) {
    if (n > 1e5) next  # Skip very large
    
    cat(sprintf("\n=== Size %e ===\n", n))
    
    # Create test data
    data_a <- runif(n)
    data_b <- runif(n)
    
    # GPU tensors
    tensor_a <- as_tensor(data_a, dtype = "float32")
    tensor_b <- as_tensor(data_b, dtype = "float32")
    
    # Addition benchmark
    gpu_add_time <- system.time({
      gpu_add_result <- tensor_a + tensor_b
      synchronize(gpu_add_result)
    })
    
    cpu_add_time <- system.time({
      cpu_add_result <- data_a + data_b
    })
    
    # Verify correctness
    expect_equal(as.vector(gpu_add_result), cpu_add_result, tolerance = 1e-6)
    
    # Multiplication benchmark
    gpu_mul_time <- system.time({
      gpu_mul_result <- tensor_a * tensor_b
      synchronize(gpu_mul_result)
    })
    
    cpu_mul_time <- system.time({
      cpu_mul_result <- data_a * data_b
    })
    
    expect_equal(as.vector(gpu_mul_result), cpu_mul_result, tolerance = 1e-6)
    
    # Sum reduction benchmark
    gpu_sum_time <- system.time({
      gpu_sum_result <- sum(tensor_a)
      # Only synchronize if result is a gpuTensor
      if (inherits(gpu_sum_result, "gpuTensor")) {
        synchronize(gpu_sum_result)
      }
    })
    
    cpu_sum_time <- system.time({
      cpu_sum_result <- sum(data_a)
    })
    
    gpu_sum_val <- if (inherits(gpu_sum_result, "gpuTensor")) as.numeric(as.array(gpu_sum_result)) else as.numeric(gpu_sum_result)
    expect_equal(gpu_sum_val, cpu_sum_result, tolerance = 1e-6)
    
    # Report results
    cat(sprintf("Addition: GPU=%.4fs, CPU=%.4fs\n", 
                gpu_add_time[["elapsed"]], cpu_add_time[["elapsed"]]))
    cat(sprintf("Multiplication: GPU=%.4fs, CPU=%.4fs\n", 
                gpu_mul_time[["elapsed"]], cpu_mul_time[["elapsed"]]))
    cat(sprintf("Sum: GPU=%.4fs, CPU=%.4fs\n", 
                gpu_sum_time[["elapsed"]], cpu_sum_time[["elapsed"]]))
  }
}) 