context("Comprehensive dtype coverage for all operations")

# Helper functions
expect_tensor_equal <- function(tensor, expected, tolerance = 1e-6) {
  expect_equal(as.array(tensor), expected, tolerance = tolerance)
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

# Test helper to run operation on both dtypes
test_both_dtypes <- function(operation_name, test_func) {
  test_that(paste(operation_name, "works with both float32 and float64"), {
    # Test with float32
    result_f32 <- test_func("float")
    expect_s3_class(result_f32, "gpuTensor")
    expect_equal(dtype(result_f32), "float")
    
    # Test with float64
    result_f64 <- test_func("double")
    expect_s3_class(result_f64, "gpuTensor")
    expect_equal(dtype(result_f64), "double")
    
    # Results should be numerically similar (within precision limits)
    expect_equal(as.array(result_f32), as.array(result_f64), tolerance = 1e-6)
  })
}

# =============================================================================
# ARITHMETIC OPERATIONS DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Addition", function(dtype) {
  a <- as_tensor(c(1.1, 2.2, 3.3), dtype = dtype)
  b <- as_tensor(c(0.9, 1.8, 2.7), dtype = dtype)
  return(a + b)
})

test_both_dtypes("Subtraction", function(dtype) {
  a <- as_tensor(c(5.5, 4.4, 3.3), dtype = dtype)
  b <- as_tensor(c(1.1, 2.2, 1.1), dtype = dtype)
  return(a - b)
})

test_both_dtypes("Multiplication", function(dtype) {
  a <- as_tensor(c(2.5, 3.0, 1.5), dtype = dtype)
  b <- as_tensor(c(2.0, 1.5, 4.0), dtype = dtype)
  return(a * b)
})

test_both_dtypes("Division", function(dtype) {
  a <- as_tensor(c(10.0, 15.0, 8.0), dtype = dtype)
  b <- as_tensor(c(2.0, 3.0, 4.0), dtype = dtype)
  return(a / b)
})

test_both_dtypes("Scalar Addition", function(dtype) {
  a <- as_tensor(c(1.0, 2.0, 3.0), dtype = dtype)
  return(a + 5.5)
})

test_both_dtypes("Scalar Multiplication", function(dtype) {
  a <- as_tensor(c(1.0, 2.0, 3.0), dtype = dtype)
  return(a * 2.5)
})

# =============================================================================
# COMPARISON OPERATIONS DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Greater Than Comparison", function(dtype) {
  a <- as_tensor(c(1.1, 3.3, 2.2), dtype = dtype)
  b <- as_tensor(c(2.0, 2.0, 2.0), dtype = dtype)
  return(a > b)
})

test_both_dtypes("Less Than Comparison", function(dtype) {
  a <- as_tensor(c(1.1, 3.3, 2.2), dtype = dtype)
  b <- as_tensor(c(2.0, 2.0, 2.0), dtype = dtype)
  return(a < b)
})

test_both_dtypes("Equality Comparison", function(dtype) {
  a <- as_tensor(c(2.0, 3.0, 2.0), dtype = dtype)
  b <- as_tensor(c(2.0, 2.0, 2.0), dtype = dtype)
  return(a == b)
})

test_both_dtypes("Inequality Comparison", function(dtype) {
  a <- as_tensor(c(2.0, 3.0, 2.0), dtype = dtype)
  b <- as_tensor(c(2.0, 2.0, 2.0), dtype = dtype)
  return(a != b)
})

test_both_dtypes("Greater Than or Equal Comparison", function(dtype) {
  a <- as_tensor(c(1.9, 2.0, 2.1), dtype = dtype)
  b <- as_tensor(c(2.0, 2.0, 2.0), dtype = dtype)
  return(a >= b)
})

test_both_dtypes("Less Than or Equal Comparison", function(dtype) {
  a <- as_tensor(c(1.9, 2.0, 2.1), dtype = dtype)
  b <- as_tensor(c(2.0, 2.0, 2.0), dtype = dtype)
  return(a <= b)
})

# =============================================================================
# MATHEMATICAL FUNCTIONS DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Exponential Function", function(dtype) {
  a <- as_tensor(c(0.5, 1.0, 1.5), dtype = dtype)
  return(exp(a))
})

test_both_dtypes("Natural Logarithm", function(dtype) {
  a <- as_tensor(c(1.0, 2.0, 3.0), dtype = dtype)
  return(log(a))
})

test_both_dtypes("Square Root", function(dtype) {
  a <- as_tensor(c(1.0, 4.0, 9.0), dtype = dtype)
  return(sqrt(a))
})

test_both_dtypes("Sine Function", function(dtype) {
  a <- as_tensor(c(0.0, pi/2, pi), dtype = dtype)
  return(sin(a))
})

test_both_dtypes("Cosine Function", function(dtype) {
  a <- as_tensor(c(0.0, pi/2, pi), dtype = dtype)
  return(cos(a))
})

test_both_dtypes("Hyperbolic Tangent", function(dtype) {
  a <- as_tensor(c(-1.0, 0.0, 1.0), dtype = dtype)
  return(tanh(a))
})

test_both_dtypes("Absolute Value", function(dtype) {
  a <- as_tensor(c(-2.5, 0.0, 3.5), dtype = dtype)
  return(abs(a))
})

test_both_dtypes("Floor Function", function(dtype) {
  a <- as_tensor(c(-2.7, -0.3, 2.8), dtype = dtype)
  return(floor(a))
})

test_both_dtypes("Ceiling Function", function(dtype) {
  a <- as_tensor(c(-2.7, -0.3, 2.8), dtype = dtype)
  return(ceiling(a))
})

test_both_dtypes("Round Function", function(dtype) {
  a <- as_tensor(c(-2.7, -0.3, 2.8), dtype = dtype)
  return(round(a))
})

test_both_dtypes("Error Function", function(dtype) {
  a <- as_tensor(c(-1.0, 0.0, 1.0), dtype = dtype)
  return(erf(a))
})

test_both_dtypes("Power Function (Scalar)", function(dtype) {
  a <- as_tensor(c(2.0, 3.0, 4.0), dtype = dtype)
  return(a ^ 2.5)
})

# =============================================================================
# ACTIVATION FUNCTIONS DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Sigmoid Function", function(dtype) {
  a <- as_tensor(c(-2.0, 0.0, 2.0), dtype = dtype)
  return(sigmoid(a))
})

test_both_dtypes("ReLU Function", function(dtype) {
  a <- as_tensor(c(-2.0, 0.0, 2.0), dtype = dtype)
  return(relu(a))
})

test_both_dtypes("Softmax Function", function(dtype) {
  a <- as_tensor(c(1.0, 2.0, 3.0), dtype = dtype)
  return(softmax(a))
})

# =============================================================================
# REDUCTION OPERATIONS DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Sum Reduction", function(dtype) {
  a <- as_tensor(c(1.1, 2.2, 3.3, 4.4), dtype = dtype)
  result <- sum(a)
  # Sum might return scalar, wrap in tensor for consistency
  if (!inherits(result, "gpuTensor")) {
    result <- as_tensor(c(result), dtype = dtype)
  }
  return(result)
})

test_both_dtypes("Mean Reduction", function(dtype) {
  a <- as_tensor(c(1.0, 2.0, 3.0, 4.0), dtype = dtype)
  result <- mean(a)
  if (!inherits(result, "gpuTensor")) {
    result <- as_tensor(c(result), dtype = dtype)
  }
  return(result)
})

test_both_dtypes("Max Reduction", function(dtype) {
  a <- as_tensor(c(1.5, 4.2, 2.8, 3.1), dtype = dtype)
  result <- max(a)
  if (!inherits(result, "gpuTensor")) {
    result <- as_tensor(c(result), dtype = dtype)
  }
  return(result)
})

test_both_dtypes("Min Reduction", function(dtype) {
  a <- as_tensor(c(3.5, 1.2, 4.8, 2.1), dtype = dtype)
  result <- min(a)
  if (!inherits(result, "gpuTensor")) {
    result <- as_tensor(c(result), dtype = dtype)
  }
  return(result)
})

test_both_dtypes("Product Reduction", function(dtype) {
  a <- as_tensor(c(1.1, 1.2, 1.3), dtype = dtype)  # Small values to avoid overflow
  result <- prod(a)
  if (!inherits(result, "gpuTensor")) {
    result <- as_tensor(c(result), dtype = dtype)
  }
  return(result)
})

test_both_dtypes("Variance Reduction", function(dtype) {
  a <- as_tensor(c(1.0, 2.0, 3.0, 4.0, 5.0), dtype = dtype)
  result <- var(a)
  if (!inherits(result, "gpuTensor")) {
    result <- as_tensor(c(result), dtype = dtype)
  }
  return(result)
})

# =============================================================================
# LINEAR ALGEBRA OPERATIONS DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Matrix Multiplication", function(dtype) {
  A <- as_tensor(matrix(c(1.1, 2.2, 3.3, 4.4), nrow = 2, ncol = 2), dtype = dtype)
  B <- as_tensor(matrix(c(0.5, 1.5, 2.5, 3.5), nrow = 2, ncol = 2), dtype = dtype)
  return(matmul(A, B))
})

test_both_dtypes("Matrix-Vector Multiplication", function(dtype) {
  A <- as_tensor(matrix(c(1.0, 2.0, 3.0, 4.0), nrow = 2, ncol = 2), dtype = dtype)
  v <- as_tensor(c(0.5, 1.5), dtype = dtype)
  return(matvec(A, v))
})

test_both_dtypes("Vector-Matrix Multiplication", function(dtype) {
  v <- as_tensor(c(2.0, 3.0), dtype = dtype)
  A <- as_tensor(matrix(c(1.0, 2.0, 3.0, 4.0), nrow = 2, ncol = 2), dtype = dtype)
  return(vecmat(v, A))
})

test_both_dtypes("Outer Product", function(dtype) {
  a <- as_tensor(c(1.0, 2.0), dtype = dtype)
  b <- as_tensor(c(3.0, 4.0, 5.0), dtype = dtype)
  return(outer_product(a, b))
})

# =============================================================================
# ELEMENT-WISE BINARY OPERATIONS DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Element-wise Maximum (pmax)", function(dtype) {
  a <- as_tensor(c(1.5, 3.2, 2.1), dtype = dtype)
  b <- as_tensor(c(2.0, 2.5, 3.0), dtype = dtype)
  return(pmax(a, b))
})

test_both_dtypes("Element-wise Minimum (pmin)", function(dtype) {
  a <- as_tensor(c(1.5, 3.2, 2.1), dtype = dtype)
  b <- as_tensor(c(2.0, 2.5, 3.0), dtype = dtype)
  return(pmin(a, b))
})

test_both_dtypes("Element-wise Power", function(dtype) {
  a <- as_tensor(c(2.0, 3.0, 4.0), dtype = dtype)
  b <- as_tensor(c(2.0, 2.0, 0.5), dtype = dtype)
  return(tensor_pow(a, b))
})

# =============================================================================
# TENSOR MANIPULATION DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Transpose", function(dtype) {
  A <- as_tensor(matrix(c(1.1, 2.2, 3.3, 4.4, 5.5, 6.6), nrow = 2, ncol = 3), dtype = dtype)
  return(transpose(A))
})

test_both_dtypes("Reshape", function(dtype) {
  a <- as_tensor(c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype = dtype)
  return(reshape(a, c(2, 3)))
})

test_both_dtypes("View", function(dtype) {
  a <- as_tensor(c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype = dtype)
  return(view(a, c(3, 2)))
})

# =============================================================================
# CUMULATIVE OPERATIONS DTYPE COVERAGE
# =============================================================================

test_both_dtypes("Cumulative Sum", function(dtype) {
  a <- as_tensor(c(1.1, 2.2, 3.3, 4.4), dtype = dtype)
  return(cumsum(a))
})

test_both_dtypes("Cumulative Product", function(dtype) {
  a <- as_tensor(c(1.01, 1.02, 1.03), dtype = dtype)  # Small values to avoid overflow
  return(cumprod(a))
})

test_both_dtypes("Differences", function(dtype) {
  a <- as_tensor(c(1.0, 3.0, 6.0, 10.0), dtype = dtype)
  return(diff(a))
})

# =============================================================================
# TENSOR CREATION DTYPE COVERAGE
# =============================================================================

test_that("empty_tensor works with both dtypes", {
  shape <- c(3, 4)
  
  # Test float32
  tensor_f32 <- empty_tensor(shape, dtype = "float")
  expect_equal(dtype(tensor_f32), "float")
  expect_equal(shape(tensor_f32), shape)
  
  # Test float64
  tensor_f64 <- empty_tensor(shape, dtype = "double")
  expect_equal(dtype(tensor_f64), "double")
  expect_equal(shape(tensor_f64), shape)
})

test_that("create_ones_like preserves dtype", {
  # Test float32
  original_f32 <- as_tensor(c(1.5, 2.5, 3.5), dtype = "float")
  ones_f32 <- create_ones_like(original_f32)
  expect_equal(dtype(ones_f32), "float")
  expect_tensor_equal(ones_f32, c(1, 1, 1))
  
  # Test float64
  original_f64 <- as_tensor(c(1.5, 2.5, 3.5), dtype = "double")
  ones_f64 <- create_ones_like(original_f64)
  expect_equal(dtype(ones_f64), "double")
  expect_tensor_equal(ones_f64, c(1, 1, 1))
})

# =============================================================================
# RANDOM GENERATION DTYPE COVERAGE
# =============================================================================

test_that("random tensor generation works with both dtypes", {
  shape <- c(100)
  
  # Test uniform generation
  uniform_f32 <- rand_tensor_uniform(shape, dtype = "float")
  expect_equal(dtype(uniform_f32), "float")
  expect_equal(shape(uniform_f32), shape)
  
  uniform_f64 <- rand_tensor_uniform(shape, dtype = "double")
  expect_equal(dtype(uniform_f64), "double")
  expect_equal(shape(uniform_f64), shape)
  
  # Test normal generation
  normal_f32 <- rand_tensor_normal(shape, dtype = "float")
  expect_equal(dtype(normal_f32), "float")
  expect_equal(shape(normal_f32), shape)
  
  normal_f64 <- rand_tensor_normal(shape, dtype = "double")
  expect_equal(dtype(normal_f64), "double")
  expect_equal(shape(normal_f64), shape)
})

# =============================================================================
# MIXED DTYPE ERROR HANDLING
# =============================================================================

test_that("mixed dtype operations handle errors appropriately", {
  tensor_f32 <- as_tensor(c(1.0, 2.0, 3.0), dtype = "float")
  tensor_f64 <- as_tensor(c(1.0, 2.0, 3.0), dtype = "double")
  
  # Test that mixed dtype operations either work or give clear errors
  tryCatch({
    result <- tensor_f32 + tensor_f64
    # If this succeeds, result should have a consistent dtype
    expect_s3_class(result, "gpuTensor")
    expect_true(dtype(result) %in% c("float", "double"))
  }, error = function(e) {
    # If this errors, it should mention dtype mismatch
    expect_true(grepl("dtype", e$message, ignore.case = TRUE))
  })
  
  # Test mixed dtype matrix operations
  mat_f32 <- as_tensor(matrix(c(1, 2, 3, 4), nrow = 2), dtype = "float")
  mat_f64 <- as_tensor(matrix(c(1, 2, 3, 4), nrow = 2), dtype = "double")
  
  tryCatch({
    result_matmul <- matmul(mat_f32, mat_f64)
    expect_s3_class(result_matmul, "gpuTensor")
  }, error = function(e) {
    expect_true(grepl("dtype", e$message, ignore.case = TRUE))
  })
})

# =============================================================================
# PRECISION TESTS
# =============================================================================

test_that("dtype precision differences are respected", {
  # Test that float64 maintains higher precision than float32
  precise_value <- 1.23456789012345
  
  tensor_f32 <- as_tensor(c(precise_value), dtype = "float")
  tensor_f64 <- as_tensor(c(precise_value), dtype = "double")
  
  # Extract values
  val_f32 <- as.numeric(as.array(tensor_f32))
  val_f64 <- as.numeric(as.array(tensor_f64))
  
  # Float64 should be closer to the original precise value
  expect_true(abs(val_f64 - precise_value) <= abs(val_f32 - precise_value))
  
  # Test precision in computations
  small_increment <- 1e-10
  
  # This increment might be lost in float32 but preserved in float64
  result_f32 <- as.numeric(as.array(tensor_f32 + small_increment))
  result_f64 <- as.numeric(as.array(tensor_f64 + small_increment))
  
  # Float64 should show the increment more accurately
  diff_f32 <- abs((result_f32 - val_f32) - small_increment)
  diff_f64 <- abs((result_f64 - val_f64) - small_increment)
  
  expect_true(diff_f64 <= diff_f32)
})

# =============================================================================
# COMPREHENSIVE INTEGRATION TEST
# =============================================================================

test_that("complex operations work correctly with both dtypes", {
  # Test a complex chain of operations with both dtypes
  test_complex_chain <- function(dtype) {
    # Create test data
    A <- as_tensor(matrix(stats::runif(9, 0.5, 2.0), nrow = 3, ncol = 3), dtype = dtype)
    v <- as_tensor(c(1.5, 2.5, 3.5), dtype = dtype)
    
    # Complex operation chain
    # 1. Matrix-vector multiplication
    result1 <- matvec(A, v)
    
    # 2. Apply activation function
    result2 <- tanh(result1)
    
    # 3. Element-wise operations
    result3 <- result2 * 2.0 + 1.0
    
    # 4. Comparison and masking
    mask <- result3 > 1.5
    
    # 5. Reduction
    final_result <- sum(result3 * mask)
    
    return(list(
      intermediate = result3,
      mask = mask,
      final = final_result
    ))
  }
  
  # Test with both dtypes
  result_f32 <- test_complex_chain("float")
  result_f64 <- test_complex_chain("double")
  
  # Verify dtypes are preserved throughout
  expect_equal(dtype(result_f32$intermediate), "float")
  expect_equal(dtype(result_f64$intermediate), "double")
  
  # Results should be numerically similar
  expect_equal(as.array(result_f32$intermediate), 
               as.array(result_f64$intermediate), 
               tolerance = 1e-6)
  
  # Final scalar results should be similar
  final_f32 <- if (inherits(result_f32$final, "gpuTensor")) as.numeric(result_f32$final) else result_f32$final
  final_f64 <- if (inherits(result_f64$final, "gpuTensor")) as.numeric(result_f64$final) else result_f64$final
  
  expect_equal(final_f32, final_f64, tolerance = 1e-6)
}) 