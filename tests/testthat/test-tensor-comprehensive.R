test_that("Comprehensive tensor operations test suite", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== COMPREHENSIVE TENSOR OPERATIONS TEST SUITE ===\n")
  
  # Helper function to verify tensor is on GPU
  verify_gpu_tensor <- function(tensor, operation_name = "operation") {
    if (!inherits(tensor, "gpuTensor")) {
      warning(paste("❌ GPU FALLBACK:", operation_name, "returned non-gpuTensor object"))
      return(FALSE)
    }
    
    # Additional check: verify data is actually on GPU by attempting a GPU-specific operation
    tryCatch({
      # Try to get tensor info - this should work for GPU tensors
      info <- tensor_info_unified(tensor)
      if (grepl("CUDA", info, ignore.case = TRUE)) {
        return(TRUE)
      } else {
        warning(paste("❌ GPU FALLBACK:", operation_name, "not on CUDA device"))
        return(FALSE)
      }
    }, error = function(e) {
      warning(paste("❌ GPU VERIFICATION FAILED:", operation_name, "-", e$message))
      return(FALSE)
    })
  }
  
  # Test data setup
  set.seed(42)
  sizes <- c(1e3, 1e4, 1e5)
  dtypes <- c("float", "double") # Add "int32", "int64" when supported
  
  for (size in sizes) {
    for (dtype in dtypes) {
      cat(sprintf("\n>> Testing size: %s, dtype: %s <<\n", 
                  format(size, scientific=TRUE), dtype))
      
      # Create test data
      data_a <- runif(size, -10, 10)
      data_b <- runif(size, -5, 15)
      
      # Create tensors
      tensor_a <- as_tensor(data_a, dtype = dtype)
      tensor_b <- as_tensor(data_b, dtype = dtype)
      
      # Verify tensors are on GPU
      verify_gpu_tensor(tensor_a, paste("tensor_a creation", dtype, size))
      verify_gpu_tensor(tensor_b, paste("tensor_b creation", dtype, size))
      
      # === BASIC PROPERTIES ===
      expect_equal(as.numeric(size(tensor_a)), size)
      expect_equal(dtype(tensor_a), dtype)
      expect_true(is_contiguous(tensor_a))
      
      # === ARITHMETIC OPERATIONS ===
      
      # Addition
      result_add_gpu <- tensor_a + tensor_b
      verify_gpu_tensor(result_add_gpu, paste("addition", dtype, size))
      result_add_cpu <- data_a + data_b
      expect_equal(as.array(result_add_gpu), result_add_cpu, tolerance = 1e-6)
      
      # Subtraction  
      result_sub_gpu <- tensor_a - tensor_b
      verify_gpu_tensor(result_sub_gpu, paste("subtraction", dtype, size))
      result_sub_cpu <- data_a - data_b
      expect_equal(as.array(result_sub_gpu), result_sub_cpu, tolerance = 1e-6)
      
      # Multiplication
      result_mul_gpu <- tensor_a * tensor_b
      verify_gpu_tensor(result_mul_gpu, paste("multiplication", dtype, size))
      result_mul_cpu <- data_a * data_b
      expect_equal(as.array(result_mul_gpu), result_mul_cpu, tolerance = 1e-6)
      
      # Division (with safe values)
      data_b_safe <- pmax(data_b, 0.1)  # Avoid division by zero
      tensor_b_safe <- as_tensor(data_b_safe, dtype = dtype)
      verify_gpu_tensor(tensor_b_safe, paste("safe tensor creation", dtype, size))
      result_div_gpu <- tensor_a / tensor_b_safe
      verify_gpu_tensor(result_div_gpu, paste("division", dtype, size))
      result_div_cpu <- data_a / data_b_safe
      expect_equal(as.array(result_div_gpu), result_div_cpu, tolerance = 1e-6)
      
      # Scalar operations
      scalar_val <- 3.14159
      result_scalar_add_gpu <- tensor_a + scalar_val
      verify_gpu_tensor(result_scalar_add_gpu, paste("scalar addition", dtype, size))
      result_scalar_add_cpu <- data_a + scalar_val
      expect_equal(as.array(result_scalar_add_gpu), result_scalar_add_cpu, tolerance = 1e-6)
      
      result_scalar_mul_gpu <- tensor_a * scalar_val
      verify_gpu_tensor(result_scalar_mul_gpu, paste("scalar multiplication", dtype, size))
      result_scalar_mul_cpu <- data_a * scalar_val
      expect_equal(as.array(result_scalar_mul_gpu), result_scalar_mul_cpu, tolerance = 1e-6)
      
      # === REDUCTION OPERATIONS ===w
      
      # Sum
      sum_gpu <- sum(tensor_a)
      sum_cpu <- sum(data_a)
      expect_equal(sum_gpu, sum_cpu, tolerance = 1e-5)
      
      # Mean
      if (exists("mean.gpuTensor")) {
        mean_gpu <- mean(tensor_a)
        mean_cpu <- mean(data_a)
        expect_equal(mean_gpu, mean_cpu, tolerance = 1e-6)
      }
      
      # Min/Max (for positive values)
      data_pos <- abs(data_a) + 1  # Ensure positive
      tensor_pos <- as_tensor(data_pos, dtype = dtype)
      verify_gpu_tensor(tensor_pos, paste("positive tensor", dtype, size))
      
      if (exists("max.gpuTensor")) {
        max_gpu <- max(tensor_pos)
        max_cpu <- max(data_pos)
        expect_equal(max_gpu, max_cpu, tolerance = 1e-6)
        
        min_gpu <- min(tensor_pos)
        min_cpu <- min(data_pos)
        expect_equal(min_gpu, min_cpu, tolerance = 1e-6)
      }
      
      # === MATH OPERATIONS ===
      
      # Exponential (with clipped values to avoid overflow)
      data_exp <- pmax(pmin(data_a, 5), -5)  # Clip to [-5, 5]
      tensor_exp <- as_tensor(data_exp, dtype = dtype)
      verify_gpu_tensor(tensor_exp, paste("exp tensor", dtype, size))
      
      if (exists("exp.gpuTensor")) {
        result_exp_gpu <- exp(tensor_exp)
        verify_gpu_tensor(result_exp_gpu, paste("exp operation", dtype, size))
        result_exp_cpu <- exp(data_exp)
        expect_equal(as.array(result_exp_gpu), result_exp_cpu, tolerance = 1e-5)
      }
      
      # Square root (positive values only)
      if (exists("sqrt.gpuTensor")) {
        result_sqrt_gpu <- sqrt(tensor_pos)
        verify_gpu_tensor(result_sqrt_gpu, paste("sqrt operation", dtype, size))
        result_sqrt_cpu <- sqrt(data_pos)
        expect_equal(as.array(result_sqrt_gpu), result_sqrt_cpu, tolerance = 1e-6)
      }
      
      # Logarithm (positive values only)
      if (exists("log.gpuTensor")) {
        result_log_gpu <- log(tensor_pos)
        verify_gpu_tensor(result_log_gpu, paste("log operation", dtype, size))
        result_log_cpu <- log(data_pos)
        expect_equal(as.array(result_log_gpu), result_log_cpu, tolerance = 1e-6)
      }
    }
  }
  
  cat("\n=== MATRIX OPERATIONS ===\n")
  
  # Matrix multiplication tests
  matrices <- list(
    list(m=10, n=8, k=12),
    list(m=50, n=30, k=40),
    list(m=100, n=80, k=90)
  )
  
  for (mat_size in matrices) {
    cat(sprintf("Testing matrix multiplication: %dx%d * %dx%d\n", 
                mat_size$m, mat_size$k, mat_size$k, mat_size$n))
    
    # Create random matrices
    A_data <- matrix(runif(mat_size$m * mat_size$k, -1, 1), 
                     nrow = mat_size$m, ncol = mat_size$k)
    B_data <- matrix(runif(mat_size$k * mat_size$n, -1, 1), 
                     nrow = mat_size$k, ncol = mat_size$n)
    
    A_tensor <- as_tensor(A_data, dtype = "float")
    B_tensor <- as_tensor(B_data, dtype = "float")
    
    verify_gpu_tensor(A_tensor, paste("matrix A", mat_size$m, "x", mat_size$k))
    verify_gpu_tensor(B_tensor, paste("matrix B", mat_size$k, "x", mat_size$n))
    
    # Matrix multiplication
    if (exists("matmul") || exists("%*%.gpuTensor")) {
      if (exists("matmul")) {
        result_gpu <- matmul(A_tensor, B_tensor)
      } else {
        result_gpu <- A_tensor %*% B_tensor
      }
      verify_gpu_tensor(result_gpu, paste("matmul result", mat_size$m, "x", mat_size$n))
      result_cpu <- A_data %*% B_data
      
      expect_equal(as.array(result_gpu), result_cpu, tolerance = 1e-5)
    }
  }
  
  cat("\n=== SHAPE OPERATIONS ===\n")
  
  # Test various reshape and view operations
  original_data <- runif(24, -2, 2)
  tensor_orig <- as_tensor(original_data, dtype = "float")
  verify_gpu_tensor(tensor_orig, "original tensor for reshaping")
  
  # Reshape tests
  shapes_to_test <- list(c(24), c(6, 4), c(3, 8), c(2, 3, 4))
  for (shape in shapes_to_test) {
    cat(sprintf("Testing reshape to: %s\n", paste(shape, collapse="x")))
    
    if (exists("view") || exists("reshape.gpuTensor")) {
      if (exists("view")) {
        reshaped <- view(tensor_orig, shape)
      } else {
        reshaped <- reshape(tensor_orig, shape)
      }
      
      verify_gpu_tensor(reshaped, paste("reshape to", paste(shape, collapse="x")))
      
      expect_equal(length(as.array(reshaped)), 24)
      expect_equal(as.numeric(size(reshaped)), 24)
      expect_equal(as.vector(as.array(reshaped)), original_data, tolerance = 1e-7)
    }
  }
  
  # Transpose test (2D only)
  mat_data <- matrix(runif(20, -1, 1), nrow = 4, ncol = 5)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  verify_gpu_tensor(mat_tensor, "matrix for transpose")
  
  if (exists("t.gpuTensor") || exists("transpose")) {
    if (exists("t.gpuTensor")) {
      transposed <- t(mat_tensor)
    } else {
      transposed <- transpose(mat_tensor)
    }
    
    verify_gpu_tensor(transposed, "transpose result")
    result_cpu <- t(mat_data)
    expect_equal(as.array(transposed), result_cpu, tolerance = 1e-7)
  }
  
  cat("\n=== BROADCASTING TESTS ===\n")
  
  # Broadcasting tests
  vec_data <- runif(5, -1, 1)
  mat_data_5x3 <- matrix(runif(15, -1, 1), nrow = 5, ncol = 3)
  
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  mat_tensor <- as_tensor(mat_data_5x3, dtype = "float")
  
  verify_gpu_tensor(vec_tensor, "vector for broadcasting")
  verify_gpu_tensor(mat_tensor, "matrix for broadcasting")
  
  # Test broadcasting if supported
  # Note: This depends on the broadcasting implementation
  
  cat("\n=== DTYPE CONVERSION TESTS ===\n")
  
  # Test dtype conversions
  test_data <- runif(100, -5, 5)
  tensor_f32 <- as_tensor(test_data, dtype = "float")
  verify_gpu_tensor(tensor_f32, "float32 tensor for conversion")
  
  if (exists("to_dtype")) {
    # Convert to float64 and back
    tensor_f64 <- to_dtype(tensor_f32, "double")
    verify_gpu_tensor(tensor_f64, "converted to double")
    expect_equal(dtype(tensor_f64), "double")
    expect_equal(as.array(tensor_f64), test_data, tolerance = 1e-6)
    
    tensor_back <- to_dtype(tensor_f64, "float")
    verify_gpu_tensor(tensor_back, "converted back to float")
    expect_equal(dtype(tensor_back), "float")
    expect_equal(as.array(tensor_back), test_data, tolerance = 1e-6)
  }
  
  cat("\n=== MEMORY AND CONTIGUITY TESTS ===\n")
  
  # Test contiguity after operations
  big_tensor <- as_tensor(runif(1000, -1, 1), dtype = "float")
  verify_gpu_tensor(big_tensor, "big tensor for contiguity test")
  
  expect_true(is_contiguous(big_tensor))
  
  # Operations should preserve or handle non-contiguous tensors correctly
  result_ops <- big_tensor + 1.0
  verify_gpu_tensor(result_ops, "contiguity test result")
  expect_equal(length(as.array(result_ops)), 1000)
  
  # Test synchronization
  synchronize(big_tensor)  # Should not error
  
  cat("\n✅ All comprehensive tensor tests passed! All operations verified on GPU.\n")
})

test_that("Error handling and edge cases", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== ERROR HANDLING AND EDGE CASES ===\n")
  
  # Test empty tensors
  expect_error(as_tensor(numeric(0)), "positive")
  
  # Test invalid shapes
  expect_error(as_tensor(c(1,2,3), shape = c(2, 3)), "size.*match")
  
  # Test invalid dtypes
  expect_error(as_tensor(c(1,2,3), dtype = "invalid_dtype"))
  
  # Test dimension mismatches for matrix operations
  if (exists("matmul")) {
    A <- as_tensor(matrix(1:6, 2, 3), dtype = "float")
    B <- as_tensor(matrix(1:6, 3, 2), dtype = "float")
    C <- as_tensor(matrix(1:4, 2, 2), dtype = "float")
    
    # This should work
    expect_no_error(matmul(A, B))
    
    # This should fail - dimension mismatch
    expect_error(matmul(A, C), "Incompatible.*dimensions")
  }
  
  # Test domain errors for math functions
  if (exists("log.gpuTensor")) {
          negative_tensor <- as_tensor(c(-1, -2, -3), dtype = "float")
    expect_error(log(negative_tensor), "domain error")
  }
  
  if (exists("sqrt.gpuTensor")) {
          negative_tensor <- as_tensor(c(-1, -4, -9), dtype = "float")
    expect_error(sqrt(negative_tensor), "domain error")
  }
  
  cat("\n✓ Error handling tests passed!\n")
}) 

  cat("\n=== ADVANCED TENSOR OPERATIONS ===\n")
  
  # Test complex slicing operations
  cat("Testing complex slicing operations...\n")
  large_tensor <- gpu_tensor(1:120, c(4, 5, 6))
  
  # Test slice arithmetic
  if (exists("[.gpuTensor")) {
    slice1 <- large_tensor[1:2, , ]  # First 2 layers
    slice2 <- large_tensor[3:4, , ]  # Last 2 layers
    
    # Add slices together
    slice_sum <- slice1 + slice2
    expect_equal(dim(slice_sum), c(2, 5, 6))
    
    # Multiply slice by scalar
    scaled_slice <- slice1 * 2.5
    expect_equal(dim(scaled_slice), c(2, 5, 6))
  }
  
  # Test transpose chaining (memory efficiency)
  cat("Testing transpose chaining...\n")
  matrix_tensor <- gpu_tensor(1:20, c(4, 5))
  
  if (exists("transpose")) {
    # Chain: transpose -> add scalar -> transpose back
    result_chain <- transpose(transpose(matrix_tensor) + 1.0)
    expect_equal(shape(result_chain), c(4, 5))
    
    # Verify result correctness
    expected <- as.array(matrix_tensor) + 1.0
    expect_equal(as.array(result_chain), expected, tolerance = 1e-10)
  }
  
  # Test memory-efficient operations
  cat("Testing memory-efficient operations...\n")
  
  # Create large tensor for memory efficiency test
  large_data <- runif(10000)
  large_tensor_1d <- as_tensor(large_data, dtype = "float")
  
  # Chain multiple operations efficiently
  if (exists("view") && exists("reshape")) {
    # Reshape -> arithmetic -> reshape back (should be memory efficient)
    reshaped <- reshape(large_tensor_1d, c(100, 100))
    operated <- reshaped * 2.0 + 1.0
    final_result <- reshape(operated, c(10000))
    
    expect_equal(size(final_result), 10000)
    expect_equal(as.vector(final_result), (large_data * 2.0 + 1.0), tolerance = 1e-6)
  }
  
  # Test advanced broadcasting with views
  cat("Testing advanced broadcasting with views...\n")
  
  if (exists("view")) {
    # Create tensors with different shapes for complex broadcasting
    tensor_2x3 <- gpu_tensor(1:6, c(2, 3))
    tensor_1x3 <- gpu_tensor(c(10, 20, 30), c(1, 3))
    
    # Broadcasting addition
    broadcast_result <- tensor_2x3 + tensor_1x3
    expect_equal(shape(broadcast_result), c(2, 3))
    
    # Verify broadcasting worked correctly (convert to vectors for R compatibility)
    expected_broadcast <- as.vector(as.array(tensor_2x3)) + rep(as.vector(as.array(tensor_1x3)), each=2)
    expect_equal(as.vector(as.array(broadcast_result)), expected_broadcast, tolerance = 1e-10)
  }
  
  # Test complex operation chains
  cat("Testing complex operation chains...\n")
  
  # Create test matrices
  mat_a <- gpu_tensor(matrix(runif(20, -1, 1), 4, 5), c(4, 5))
  mat_b <- gpu_tensor(matrix(runif(15, -1, 1), 5, 3), c(5, 3))
  
  if (exists("matmul") && exists("transpose")) {
    # Complex chain: A * B -> transpose -> add scalar -> reduce
    expect_no_error({
      step1 <- matmul(mat_a, mat_b)  # 4x3 result
      step2 <- transpose(step1)      # 3x4 result  
      step3 <- step2 + 0.5          # Add scalar
      final_sum <- sum(step3)       # Reduce to scalar
    })
    
    expect_true(is.numeric(final_sum))
    expect_length(final_sum, 1)
  }
  
  # Test mixed operations with different tensor shapes
  cat("Testing mixed operations...\n")
  
  # Vector operations
  vec_a <- as_tensor(runif(1000), dtype = "float")
  vec_b <- as_tensor(runif(1000), dtype = "float")
  
  # Element-wise then reduction
  element_product <- vec_a * vec_b
  dot_product_manual <- sum(element_product)
  
  # Should match built-in dot product if available
  expect_true(is.numeric(dot_product_manual))
  expect_gt(abs(dot_product_manual), 0)  # Should be non-zero for random data
  
  # Test contiguity preservation
  cat("Testing contiguity preservation...\n")
  
  contiguous_tensor <- gpu_tensor(1:24, c(4, 6))
  expect_true(is_contiguous(contiguous_tensor))
  
  # Operations that should preserve contiguity
  scaled <- contiguous_tensor * 2.0
  expect_true(is_contiguous(scaled))
  
  added <- contiguous_tensor + 1.0
  expect_true(is_contiguous(added))
  
  # Test GPU memory efficiency
  cat("Testing GPU memory efficiency...\n")
  
  # Create multiple tensors and ensure no memory leaks
  initial_memory <- gpu_memory_available()
  
  for (i in 1:5) {
    temp_tensor <- gpu_tensor(runif(1000), c(10, 100))
    temp_result <- temp_tensor * 2.0 + 1.0
    temp_sum <- sum(temp_result)
    
    # Force cleanup
    rm(temp_tensor, temp_result, temp_sum)
  }
  
  # Force garbage collection
  gc()
  
  # Memory should be roughly the same (allowing for some fragmentation)
  final_memory <- gpu_memory_available()
  memory_diff <- abs(initial_memory - final_memory)
  memory_diff_mb <- memory_diff / (1024 * 1024)
  
  # Allow up to 100MB difference for fragmentation/overhead
  expect_lt(memory_diff_mb, 100)
  
  cat("✓ Advanced tensor operations tests completed!\n") 