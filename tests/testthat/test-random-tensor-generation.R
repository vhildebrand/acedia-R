context("Random tensor generation: rand_tensor_uniform, rand_tensor_normal, rnorm/runif methods")

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

# =============================================================================
# RAND_TENSOR_UNIFORM TESTS
# =============================================================================

test_that("rand_tensor_uniform creates tensors with correct shape and dtype", {
  # Test 1D tensor
  shape_1d <- c(100)
  tensor_1d <- rand_tensor_uniform(shape_1d, dtype = "float")
  
  expect_s3_class(tensor_1d, "gpuTensor")
  expect_equal(shape(tensor_1d), shape_1d)
  expect_equal(size(tensor_1d), prod(shape_1d))
  expect_equal(dtype(tensor_1d), "float")
  
  # Test 2D tensor
  shape_2d <- c(10, 20)
  tensor_2d <- rand_tensor_uniform(shape_2d, dtype = "float")
  
  expect_equal(shape(tensor_2d), shape_2d)
  expect_equal(size(tensor_2d), prod(shape_2d))
  
  # Test 3D tensor
  shape_3d <- c(5, 4, 3)
  tensor_3d <- rand_tensor_uniform(shape_3d, dtype = "float")
  
  expect_equal(shape(tensor_3d), shape_3d)
  expect_equal(size(tensor_3d), prod(shape_3d))
})

test_that("rand_tensor_uniform works with different dtypes", {
  shape <- c(50, 50)
  
  # Test float32
  tensor_f32 <- rand_tensor_uniform(shape, dtype = "float")
  expect_equal(dtype(tensor_f32), "float")
  expect_equal(shape(tensor_f32), shape)
  
  # Test float64
  tensor_f64 <- rand_tensor_uniform(shape, dtype = "double")
  expect_equal(dtype(tensor_f64), "double")
  expect_equal(shape(tensor_f64), shape)
})

test_that("rand_tensor_uniform respects min/max parameters", {
  shape <- c(1000)
  min_val <- 2.0
  max_val <- 5.0
  
  tensor <- rand_tensor_uniform(shape, dtype = "float", min = min_val, max = max_val)
  values <- as.vector(tensor)
  
  # All values should be within [min, max)
  expect_true(all(values >= min_val))
  expect_true(all(values < max_val))  # max is exclusive in uniform distribution
  
  # Test with negative range
  tensor_neg <- rand_tensor_uniform(shape, dtype = "float", min = -3.0, max = -1.0)
  values_neg <- as.vector(tensor_neg)
  
  expect_true(all(values_neg >= -3.0))
  expect_true(all(values_neg < -1.0))
})

test_that("rand_tensor_uniform produces reasonable statistical properties", {
  # Generate large sample for statistical tests
  n <- 10000
  shape <- c(n)
  min_val <- 0.0
  max_val <- 1.0
  
  tensor <- rand_tensor_uniform(shape, dtype = "float", min = min_val, max = max_val)
  values <- as.vector(tensor)
  
  # Check basic statistics (with some tolerance for randomness)
  sample_mean <- mean(values)
  expected_mean <- (min_val + max_val) / 2
  expect_equal(sample_mean, expected_mean, tolerance = 0.05)
  
  sample_var <- var(values)
  expected_var <- (max_val - min_val)^2 / 12  # Variance of uniform distribution
  expect_equal(sample_var, expected_var, tolerance = 0.01)
  
  # Check range
  expect_true(min(values) >= min_val)
  expect_true(max(values) < max_val)
})

test_that("rand_tensor_uniform default parameters work correctly", {
  shape <- c(100)
  
  # Default should be [0, 1)
  tensor_default <- rand_tensor_uniform(shape)
  values_default <- as.vector(tensor_default)
  
  expect_true(all(values_default >= 0.0))
  expect_true(all(values_default < 1.0))
  expect_equal(dtype(tensor_default), "float")  # Default dtype
})

# =============================================================================
# RAND_TENSOR_NORMAL TESTS
# =============================================================================

test_that("rand_tensor_normal creates tensors with correct shape and dtype", {
  # Test 1D tensor
  shape_1d <- c(100)
  tensor_1d <- rand_tensor_normal(shape_1d, dtype = "float")
  
  expect_s3_class(tensor_1d, "gpuTensor")
  expect_equal(shape(tensor_1d), shape_1d)
  expect_equal(size(tensor_1d), prod(shape_1d))
  expect_equal(dtype(tensor_1d), "float")
  
  # Test 2D tensor
  shape_2d <- c(10, 20)
  tensor_2d <- rand_tensor_normal(shape_2d, dtype = "float")
  
  expect_equal(shape(tensor_2d), shape_2d)
  expect_equal(size(tensor_2d), prod(shape_2d))
  
  # Test 3D tensor
  shape_3d <- c(5, 4, 3)
  tensor_3d <- rand_tensor_normal(shape_3d, dtype = "float")
  
  expect_equal(shape(tensor_3d), shape_3d)
  expect_equal(size(tensor_3d), prod(shape_3d))
})

test_that("rand_tensor_normal works with different dtypes", {
  shape <- c(50, 50)
  
  # Test float32
  tensor_f32 <- rand_tensor_normal(shape, dtype = "float")
  expect_equal(dtype(tensor_f32), "float")
  expect_equal(shape(tensor_f32), shape)
  
  # Test float64
  tensor_f64 <- rand_tensor_normal(shape, dtype = "double")
  expect_equal(dtype(tensor_f64), "double")
  expect_equal(shape(tensor_f64), shape)
})

test_that("rand_tensor_normal respects mean/sd parameters", {
  shape <- c(5000)  # Large sample for statistical testing
  mean_val <- 2.5
  sd_val <- 1.5
  
  tensor <- rand_tensor_normal(shape, mean = mean_val, sd = sd_val, dtype = "float")
  values <- as.vector(tensor)
  
  # Check statistical properties (with tolerance for randomness)
  sample_mean <- mean(values)
  expect_equal(sample_mean, mean_val, tolerance = 0.1)
  
  sample_sd <- sd(values)
  expect_equal(sample_sd, sd_val, tolerance = 0.1)
  
  # Test with different parameters
  tensor_neg <- rand_tensor_normal(shape, mean = -1.0, sd = 0.5, dtype = "float")
  values_neg <- as.vector(tensor_neg)
  
  sample_mean_neg <- mean(values_neg)
  expect_equal(sample_mean_neg, -1.0, tolerance = 0.05)
  
  sample_sd_neg <- sd(values_neg)
  expect_equal(sample_sd_neg, 0.5, tolerance = 0.05)
})

test_that("rand_tensor_normal produces reasonable statistical properties", {
  # Generate large sample for statistical tests
  n <- 10000
  shape <- c(n)
  
  tensor <- rand_tensor_normal(shape, mean = 0, sd = 1, dtype = "float")
  values <- as.vector(tensor)
  
  # Check basic statistics for standard normal
  sample_mean <- mean(values)
  expect_equal(sample_mean, 0.0, tolerance = 0.05)
  
  sample_sd <- sd(values)
  expect_equal(sample_sd, 1.0, tolerance = 0.05)
  
  # Check that values span reasonable range (most should be within 3 standard deviations)
  within_3sd <- sum(abs(values) <= 3) / length(values)
  expect_gt(within_3sd, 0.99)  # Should be > 99%
})

test_that("rand_tensor_normal default parameters work correctly", {
  shape <- c(1000)
  
  # Default should be mean=0, sd=1
  tensor_default <- rand_tensor_normal(shape)
  values_default <- as.vector(tensor_default)
  
  sample_mean <- mean(values_default)
  sample_sd <- sd(values_default)
  
  expect_equal(sample_mean, 0.0, tolerance = 0.1)
  expect_equal(sample_sd, 1.0, tolerance = 0.1)
  expect_equal(dtype(tensor_default), "float")  # Default dtype
})

# =============================================================================
# RNORM.GPUTENSOR METHOD TESTS
# =============================================================================

test_that("rnorm.gpuTensor method works correctly", {
  # Create template tensor
  template <- as_tensor(matrix(0, nrow = 10, ncol = 20), dtype = "float")
  
  # Generate random normal tensor with same shape
  result <- rnorm(template, mean = 1.5, sd = 0.8)
  
  expect_s3_class(result, "gpuTensor")
  expect_equal(shape(result), shape(template))
  expect_equal(dtype(result), dtype(template))
  
  # Check statistical properties
  values <- as.vector(result)
  sample_mean <- mean(values)
  sample_sd <- sd(values)
  
  expect_equal(sample_mean, 1.5, tolerance = 0.2)
  expect_equal(sample_sd, 0.8, tolerance = 0.2)
})

test_that("rnorm.gpuTensor preserves tensor properties", {
  # Test with different dtypes
  template_f32 <- as_tensor(c(1, 2, 3, 4, 5), dtype = "float")
  result_f32 <- rnorm(template_f32)
  
  expect_equal(shape(result_f32), shape(template_f32))
  expect_equal(dtype(result_f32), "float")
  
  template_f64 <- as_tensor(c(1, 2, 3, 4, 5), dtype = "double")
  result_f64 <- rnorm(template_f64)
  
  expect_equal(shape(result_f64), shape(template_f64))
  expect_equal(dtype(result_f64), "double")
})

test_that("rnorm.gpuTensor works with different tensor shapes", {
  # Test various shapes
  shapes_to_test <- list(
    c(50),           # 1D
    c(5, 10),        # 2D
    c(2, 3, 4),      # 3D
    c(2, 2, 2, 2)    # 4D
  )
  
  for (shape in shapes_to_test) {
    template <- empty_tensor(shape, dtype = "float")
    result <- rnorm(template, mean = 0, sd = 1)
    
    expect_equal(shape(result), shape)
    expect_equal(size(result), prod(shape))
    expect_s3_class(result, "gpuTensor")
  }
})

# =============================================================================
# RUNIF.GPUTENSOR METHOD TESTS
# =============================================================================

test_that("runif.gpuTensor method works correctly", {
  # Create template tensor
  template <- as_tensor(matrix(0, nrow = 10, ncol = 20), dtype = "float")
  
  # Generate random uniform tensor with same shape
  result <- runif(template, min = 2.0, max = 5.0)
  
  expect_s3_class(result, "gpuTensor")
  expect_equal(shape(result), shape(template))
  expect_equal(dtype(result), dtype(template))
  
  # Check range
  values <- as.vector(result)
  expect_true(all(values >= 2.0))
  expect_true(all(values < 5.0))
  
  # Check statistical properties
  sample_mean <- mean(values)
  expected_mean <- (2.0 + 5.0) / 2
  expect_equal(sample_mean, expected_mean, tolerance = 0.1)
})

test_that("runif.gpuTensor preserves tensor properties", {
  # Test with different dtypes
  template_f32 <- as_tensor(array(0, dim = c(3, 4, 5)), dtype = "float")
  result_f32 <- runif(template_f32)
  
  expect_equal(shape(result_f32), shape(template_f32))
  expect_equal(dtype(result_f32), "float")
  
  template_f64 <- as_tensor(array(0, dim = c(3, 4, 5)), dtype = "double")
  result_f64 <- runif(template_f64)
  
  expect_equal(shape(result_f64), shape(template_f64))
  expect_equal(dtype(result_f64), "double")
})

test_that("runif.gpuTensor works with different tensor shapes", {
  # Test various shapes
  shapes_to_test <- list(
    c(100),          # 1D
    c(10, 10),       # 2D
    c(2, 5, 4),      # 3D
    c(2, 2, 2, 2)    # 4D
  )
  
  for (shape in shapes_to_test) {
    template <- empty_tensor(shape, dtype = "float")
    result <- runif(template, min = 0, max = 1)
    
    expect_equal(shape(result), shape)
    expect_equal(size(result), prod(shape))
    expect_s3_class(result, "gpuTensor")
    
    # Check range
    values <- as.vector(result)
    expect_true(all(values >= 0.0))
    expect_true(all(values < 1.0))
  }
})

# =============================================================================
# REPRODUCIBILITY TESTS
# =============================================================================

test_that("random generation respects set.seed()", {
  skip("Random seed reproducibility not yet implemented")
  
  # Note: This test is skipped because GPU random generation
  # may not respect R's set.seed() depending on implementation
  
  # Future implementation should test:
  # set.seed(123)
  # tensor1 <- rand_tensor_uniform(c(100), dtype = "float")
  # 
  # set.seed(123)  
  # tensor2 <- rand_tensor_uniform(c(100), dtype = "float")
  # 
  # expect_tensor_equal(tensor1, as.array(tensor2))
})

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

test_that("random tensor generation error handling", {
  # Invalid shape
  expect_error(rand_tensor_uniform(c(0)), "positive")
  expect_error(rand_tensor_normal(c(-5)), "positive")
  
  # Invalid dtype
  expect_error(rand_tensor_uniform(c(10), dtype = "invalid"), "dtype")
  expect_error(rand_tensor_normal(c(10), dtype = "invalid"), "dtype")
  
  # Invalid parameters for uniform
  expect_error(rand_tensor_uniform(c(10), min = 5, max = 2), "min.*max|max.*min")
  
  # Invalid parameters for normal
  expect_error(rand_tensor_normal(c(10), sd = -1), "positive|sd")
  expect_error(rand_tensor_normal(c(10), sd = 0), "positive|sd")
})

test_that("random tensor methods error handling", {
  # Non-tensor input for methods
  expect_error(rnorm(c(1, 2, 3)), "gpuTensor")
  expect_error(runif(matrix(1:6, nrow = 2)), "gpuTensor")
  
  # Invalid parameters
  template <- as_tensor(c(1, 2, 3), dtype = "float")
  expect_error(rnorm(template, sd = -1), "positive|sd")
  expect_error(runif(template, min = 5, max = 2), "min.*max|max.*min")
})

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test_that("random tensors can be used in computations", {
  # Generate random tensors and use them in operations
  shape <- c(100, 50)
  
  uniform_tensor <- rand_tensor_uniform(shape, min = 0, max = 1, dtype = "float")
  normal_tensor <- rand_tensor_normal(shape, mean = 0, sd = 1, dtype = "float")
  
  # Arithmetic operations
  sum_tensor <- uniform_tensor + normal_tensor
  expect_s3_class(sum_tensor, "gpuTensor")
  expect_equal(shape(sum_tensor), shape)
  
  # Element-wise operations
  product_tensor <- uniform_tensor * normal_tensor
  expect_s3_class(product_tensor, "gpuTensor")
  expect_equal(shape(product_tensor), shape)
  
  # Reductions
  mean_uniform <- mean(uniform_tensor)
  mean_normal <- mean(normal_tensor)
  
  # Uniform [0,1] should have mean ~0.5
  expect_equal(mean_uniform, 0.5, tolerance = 0.1)
  # Normal(0,1) should have mean ~0
  expect_equal(mean_normal, 0.0, tolerance = 0.1)
})

test_that("random generation works with views and reshapes", {
  # Generate random tensor, then create views
  original_shape <- c(60)
  tensor <- rand_tensor_uniform(original_shape, dtype = "float")
  
  # Reshape to 2D
  reshaped <- reshape(tensor, c(10, 6))
  expect_equal(shape(reshaped), c(10, 6))
  expect_equal(size(reshaped), size(tensor))
  
  # Create view
  viewed <- view(tensor, c(6, 10))
  expect_equal(shape(viewed), c(6, 10))
  expect_equal(size(viewed), size(tensor))
  
  # Generate random tensor from reshaped template
  template_2d <- as_tensor(matrix(0, nrow = 5, ncol = 8), dtype = "float")
  random_from_template <- rnorm(template_2d)
  
  expect_equal(shape(random_from_template), c(5, 8))
  expect_s3_class(random_from_template, "gpuTensor")
})

test_that("random generation maintains GPU execution", {
  # Test with larger tensors to ensure GPU path
  large_shape <- c(1000, 500)
  
  # Time random generation
  uniform_time <- system.time({
    large_uniform <- rand_tensor_uniform(large_shape, dtype = "float")
  })
  
  normal_time <- system.time({
    large_normal <- rand_tensor_normal(large_shape, dtype = "float")
  })
  
  # Operations should complete reasonably quickly
  expect_lt(uniform_time[["elapsed"]], 5.0)
  expect_lt(normal_time[["elapsed"]], 5.0)
  
  # Verify results are GPU tensors
  expect_s3_class(large_uniform, "gpuTensor")
  expect_s3_class(large_normal, "gpuTensor")
  expect_equal(shape(large_uniform), large_shape)
  expect_equal(shape(large_normal), large_shape)
  
  # Spot check statistical properties
  uniform_sample <- as.vector(large_uniform[1:1000, 1])
  normal_sample <- as.vector(large_normal[1:1000, 1])
  
  expect_true(all(uniform_sample >= 0.0 & uniform_sample < 1.0))
  expect_equal(mean(normal_sample), 0.0, tolerance = 0.1)
})

test_that("random generation works with different memory layouts", {
  # Test with various tensor shapes and memory patterns
  shapes_to_test <- list(
    c(1000),         # 1D - contiguous
    c(50, 20),       # 2D - row-major
    c(10, 10, 10),   # 3D - standard layout
    c(2, 2, 2, 2, 2, 2, 2), # High-dimensional
    c(1, 1000),      # Thin matrix
    c(1000, 1),      # Tall matrix
    c(1, 1, 1000)    # Mostly singleton dimensions
  )
  
  for (shape in shapes_to_test) {
    # Test both uniform and normal generation
    uniform_tensor <- rand_tensor_uniform(shape, dtype = "float")
    normal_tensor <- rand_tensor_normal(shape, dtype = "float")
    
    expect_equal(shape(uniform_tensor), shape)
    expect_equal(shape(normal_tensor), shape)
    expect_equal(size(uniform_tensor), prod(shape))
    expect_equal(size(normal_tensor), prod(shape))
    
    # Verify they're contiguous (new tensors should be)
    expect_true(is_contiguous(uniform_tensor))
    expect_true(is_contiguous(normal_tensor))
  }
})

test_that("random tensor generation handles edge cases", {
  # Very small tensors
  tiny_shape <- c(1)
  tiny_uniform <- rand_tensor_uniform(tiny_shape, dtype = "float")
  tiny_normal <- rand_tensor_normal(tiny_shape, dtype = "float")
  
  expect_equal(shape(tiny_uniform), tiny_shape)
  expect_equal(shape(tiny_normal), tiny_shape)
  expect_equal(size(tiny_uniform), 1)
  expect_equal(size(tiny_normal), 1)
  
  # Check single values are in expected ranges
  uniform_val <- as.numeric(tiny_uniform)
  expect_true(uniform_val >= 0.0 && uniform_val < 1.0)
  
  # Normal value should be finite
  normal_val <- as.numeric(tiny_normal)
  expect_true(is.finite(normal_val))
  
  # Test with extreme parameters (but still valid)
  extreme_uniform <- rand_tensor_uniform(c(100), min = -1000, max = 1000, dtype = "float")
  extreme_values <- as.vector(extreme_uniform)
  expect_true(all(extreme_values >= -1000))
  expect_true(all(extreme_values < 1000))
  
  extreme_normal <- rand_tensor_normal(c(100), mean = 0, sd = 100, dtype = "float")
  extreme_normal_values <- as.vector(extreme_normal)
  expect_true(all(is.finite(extreme_normal_values)))
}) 