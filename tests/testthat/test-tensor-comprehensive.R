test_that("Comprehensive tensor operations test suite", {
  skip_if_not(gpu_available(), "GPU not available")
  
  cat("\n=== COMPREHENSIVE TENSOR OPERATIONS TEST SUITE ===\n")
  
  # Test data setup
  set.seed(42)
  sizes <- c(1e3, 1e4, 1e5)
  dtypes <- c("float32", "float64") # Add "int32", "int64" when supported
  
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
      
      # === BASIC PROPERTIES ===
      expect_equal(as.numeric(size(tensor_a)), size)
      expect_equal(dtype(tensor_a), dtype)
      expect_true(is_contiguous(tensor_a))
      
      # === ARITHMETIC OPERATIONS ===
      
      # Addition
      result_add_gpu <- tensor_a + tensor_b
      result_add_cpu <- data_a + data_b
      expect_equal(as.array(result_add_gpu), result_add_cpu, tolerance = 1e-6)
      
      # Subtraction  
      result_sub_gpu <- tensor_a - tensor_b
      result_sub_cpu <- data_a - data_b
      expect_equal(as.array(result_sub_gpu), result_sub_cpu, tolerance = 1e-6)
      
      # Multiplication
      result_mul_gpu <- tensor_a * tensor_b
      result_mul_cpu <- data_a * data_b
      expect_equal(as.array(result_mul_gpu), result_mul_cpu, tolerance = 1e-6)
      
      # Division (with safe values)
      data_b_safe <- pmax(data_b, 0.1)  # Avoid division by zero
      tensor_b_safe <- as_tensor(data_b_safe, dtype = dtype)
      result_div_gpu <- tensor_a / tensor_b_safe
      result_div_cpu <- data_a / data_b_safe
      expect_equal(as.array(result_div_gpu), result_div_cpu, tolerance = 1e-6)
      
      # Scalar operations
      scalar_val <- 3.14159
      result_scalar_add_gpu <- tensor_a + scalar_val
      result_scalar_add_cpu <- data_a + scalar_val
      expect_equal(as.array(result_scalar_add_gpu), result_scalar_add_cpu, tolerance = 1e-6)
      
      result_scalar_mul_gpu <- tensor_a * scalar_val
      result_scalar_mul_cpu <- data_a * scalar_val
      expect_equal(as.array(result_scalar_mul_gpu), result_scalar_mul_cpu, tolerance = 1e-6)
      
      # === REDUCTION OPERATIONS ===
      
      # Sum
      sum_gpu <- sum(tensor_a)
      sum_cpu <- sum(data_a)
      expect_equal(sum_gpu, sum_cpu, tolerance = 1e-6)
      
      # Mean
      if (exists("mean.gpuTensor")) {
        mean_gpu <- mean(tensor_a)
        mean_cpu <- mean(data_a)
        expect_equal(mean_gpu, mean_cpu, tolerance = 1e-6)
      }
      
      # Min/Max (for positive values)
      data_pos <- abs(data_a) + 1  # Ensure positive
      tensor_pos <- as_tensor(data_pos, dtype = dtype)
      
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
      
      if (exists("exp.gpuTensor")) {
        result_exp_gpu <- exp(tensor_exp)
        result_exp_cpu <- exp(data_exp)
        expect_equal(as.array(result_exp_gpu), result_exp_cpu, tolerance = 1e-5)
      }
      
      # Square root (positive values only)
      if (exists("sqrt.gpuTensor")) {
        result_sqrt_gpu <- sqrt(tensor_pos)
        result_sqrt_cpu <- sqrt(data_pos)
        expect_equal(as.array(result_sqrt_gpu), result_sqrt_cpu, tolerance = 1e-6)
      }
      
      # Logarithm (positive values only)
      if (exists("log.gpuTensor")) {
        result_log_gpu <- log(tensor_pos)
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
    
    A_tensor <- as_tensor(A_data, dtype = "float32")
    B_tensor <- as_tensor(B_data, dtype = "float32")
    
    # Matrix multiplication
    if (exists("matmul") || exists("%*%.gpuTensor")) {
      if (exists("matmul")) {
        result_gpu <- matmul(A_tensor, B_tensor)
      } else {
        result_gpu <- A_tensor %*% B_tensor
      }
      result_cpu <- A_data %*% B_data
      
      expect_equal(as.array(result_gpu), result_cpu, tolerance = 1e-5)
    }
  }
  
  cat("\n=== SHAPE OPERATIONS ===\n")
  
  # Test various reshape and view operations
  original_data <- runif(24, -2, 2)
  tensor_orig <- as_tensor(original_data, dtype = "float32")
  
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
      
      expect_equal(length(as.array(reshaped)), 24)
      expect_equal(as.numeric(size(reshaped)), 24)
      expect_equal(as.array(reshaped), original_data, tolerance = 1e-7)
    }
  }
  
  # Transpose test (2D only)
  mat_data <- matrix(runif(20, -1, 1), nrow = 4, ncol = 5)
  mat_tensor <- as_tensor(mat_data, dtype = "float32")
  
  if (exists("t.gpuTensor") || exists("transpose")) {
    if (exists("t.gpuTensor")) {
      transposed <- t(mat_tensor)
    } else {
      transposed <- transpose(mat_tensor)
    }
    
    result_cpu <- t(mat_data)
    expect_equal(as.array(transposed), result_cpu, tolerance = 1e-7)
  }
  
  cat("\n=== BROADCASTING TESTS ===\n")
  
  # Broadcasting tests
  vec_data <- runif(5, -1, 1)
  mat_data_5x3 <- matrix(runif(15, -1, 1), nrow = 5, ncol = 3)
  
  vec_tensor <- as_tensor(vec_data, dtype = "float32")
  mat_tensor <- as_tensor(mat_data_5x3, dtype = "float32")
  
  # Test broadcasting if supported
  # Note: This depends on the broadcasting implementation
  
  cat("\n=== DTYPE CONVERSION TESTS ===\n")
  
  # Test dtype conversions
  test_data <- runif(100, -5, 5)
  tensor_f32 <- as_tensor(test_data, dtype = "float32")
  
  if (exists("to_dtype")) {
    # Convert to float64 and back
    tensor_f64 <- to_dtype(tensor_f32, "float64")
    expect_equal(dtype(tensor_f64), "float64")
    expect_equal(as.array(tensor_f64), test_data, tolerance = 1e-6)
    
    tensor_back <- to_dtype(tensor_f64, "float32")
    expect_equal(dtype(tensor_back), "float32")
    expect_equal(as.array(tensor_back), test_data, tolerance = 1e-6)
  }
  
  cat("\n=== MEMORY AND CONTIGUITY TESTS ===\n")
  
  # Test contiguity after operations
  big_tensor <- as_tensor(runif(1000, -1, 1), dtype = "float32")
  
  expect_true(is_contiguous(big_tensor))
  
  # Operations should preserve or handle non-contiguous tensors correctly
  result_ops <- big_tensor + 1.0
  expect_equal(length(as.array(result_ops)), 1000)
  
  # Test synchronization
  synchronize(big_tensor)  # Should not error
  
  cat("\n✓ All comprehensive tensor tests passed!\n")
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
    A <- as_tensor(matrix(1:6, 2, 3), dtype = "float32")
    B <- as_tensor(matrix(1:6, 3, 2), dtype = "float32")
    C <- as_tensor(matrix(1:4, 2, 2), dtype = "float32")
    
    # This should work
    expect_no_error(matmul(A, B))
    
    # This should fail - dimension mismatch
    expect_error(matmul(A, C), "Incompatible.*dimensions")
  }
  
  # Test domain errors for math functions
  if (exists("log.gpuTensor")) {
    negative_tensor <- as_tensor(c(-1, -2, -3), dtype = "float32")
    expect_error(log(negative_tensor), "domain error")
  }
  
  if (exists("sqrt.gpuTensor")) {
    negative_tensor <- as_tensor(c(-1, -4, -9), dtype = "float32")
    expect_error(sqrt(negative_tensor), "domain error")
  }
  
  cat("\n✓ Error handling tests passed!\n")
}) 