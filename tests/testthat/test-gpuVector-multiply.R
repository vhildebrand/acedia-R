test_that("gpuVector multiplication operator works correctly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test element-wise multiplication
  a_vec <- c(2, 3, 4, 5)
  b_vec <- c(3, 4, 5, 6)
  expected <- a_vec * b_vec
  
  gpu_a <- as.gpuVector(a_vec)
  gpu_b <- as.gpuVector(b_vec)
  
  gpu_result <- gpu_a * gpu_b
  result <- as.vector(gpu_result)
  
  expect_equal(result, expected)
  expect_s3_class(gpu_result, "gpuVector")
})

test_that("gpuVector scalar multiplication operator works correctly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test scalar multiplication
  a_vec <- c(2, 3, 4, 5)
  scalar <- 2.5
  expected <- a_vec * scalar
  
  gpu_a <- as.gpuVector(a_vec)
  
  gpu_result <- gpu_a * scalar
  result <- as.vector(gpu_result)
  
  expect_equal(result, expected)
  expect_s3_class(gpu_result, "gpuVector")
})

test_that("gpuVector * operator handles edge cases", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test with empty vectors
  gpu_empty_a <- as.gpuVector(numeric(0))
  gpu_empty_b <- as.gpuVector(numeric(0))
  gpu_result <- gpu_empty_a * gpu_empty_b
  result <- as.vector(gpu_result)
  expect_equal(result, numeric(0))
  
  # Test scalar multiplication with zero
  gpu_a <- as.gpuVector(c(1, 2, 3, 4))
  gpu_result <- gpu_a * 0
  result <- as.vector(gpu_result)
  expect_equal(result, c(0, 0, 0, 0))
  
  # Test scalar multiplication with one
  gpu_result <- gpu_a * 1
  result <- as.vector(gpu_result)
  expect_equal(result, c(1, 2, 3, 4))
})

test_that("gpuVector * operator validation works", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  gpu_a <- as.gpuVector(c(1, 2, 3))
  
  # Test error for missing operand would be a syntax error, so skip this test
  
  # Test error for invalid second operand
  expect_error(gpu_a * "invalid", "Second operand must be either a gpuVector or a numeric scalar")
  expect_error(gpu_a * c(1, 2), "Second operand must be either a gpuVector or a numeric scalar")
})

test_that("gpu_multiply_vectors works correctly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic functionality
  a_vec <- c(1, 2, 3, 4, 5)
  b_vec <- c(2, 3, 4, 5, 6)
  expected <- a_vec * b_vec
  
  gpu_a <- as.gpuVector(a_vec)
  gpu_b <- as.gpuVector(b_vec)
  
  gpu_result <- gpu_multiply_vectors(gpu_a, gpu_b)
  result <- as.vector(gpu_result)
  
  expect_equal(result, expected)
  expect_s3_class(gpu_result, "gpuVector")
})

test_that("gpu_multiply_vectors handles larger vectors", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  n <- 10000
  a_vec <- runif(n)
  b_vec <- runif(n)
  expected <- a_vec * b_vec
  
  gpu_a <- as.gpuVector(a_vec)
  gpu_b <- as.gpuVector(b_vec)
  
  gpu_result <- gpu_multiply_vectors(gpu_a, gpu_b)
  result <- as.vector(gpu_result)
  
  expect_equal(length(result), n)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpu_scale_vector works correctly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic functionality
  a_vec <- c(1, 2, 3, 4, 5)
  scalar <- 3.5
  expected <- a_vec * scalar
  
  gpu_a <- as.gpuVector(a_vec)
  
  gpu_result <- gpu_scale_vector(gpu_a, scalar)
  result <- as.vector(gpu_result)
  
  expect_equal(result, expected)
  expect_s3_class(gpu_result, "gpuVector")
})

test_that("gpu_scale_vector handles larger vectors", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  n <- 10000
  a_vec <- runif(n)
  scalar <- 2.718
  expected <- a_vec * scalar
  
  gpu_a <- as.gpuVector(a_vec)
  
  gpu_result <- gpu_scale_vector(gpu_a, scalar)
  result <- as.vector(gpu_result)
  
  expect_equal(length(result), n)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpu_dot_vectors works correctly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic functionality
  a_vec <- c(1, 2, 3, 4)
  b_vec <- c(2, 3, 4, 5)
  expected <- sum(a_vec * b_vec)
  
  gpu_a <- as.gpuVector(a_vec)
  gpu_b <- as.gpuVector(b_vec)
  
  result <- gpu_dot_vectors(gpu_a, gpu_b)
  
  expect_equal(result, expected)
  expect_type(result, "double")
})

test_that("gpu_dot_vectors handles larger vectors", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  n <- 10000
  a_vec <- runif(n)
  b_vec <- runif(n)
  expected <- sum(a_vec * b_vec)
  
  gpu_a <- as.gpuVector(a_vec)
  gpu_b <- as.gpuVector(b_vec)
  
  result <- gpu_dot_vectors(gpu_a, gpu_b)
  
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpu_dot_vectors handles special cases", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test empty vectors
  gpu_empty_a <- as.gpuVector(numeric(0))
  gpu_empty_b <- as.gpuVector(numeric(0))
  result <- gpu_dot_vectors(gpu_empty_a, gpu_empty_b)
  expect_equal(result, 0.0)
  
  # Test orthogonal vectors
  gpu_a <- as.gpuVector(c(1, 0, 0))
  gpu_b <- as.gpuVector(c(0, 1, 0))
  result <- gpu_dot_vectors(gpu_a, gpu_b)
  expect_equal(result, 0.0)
  
  # Test identical vectors (squared norm)
  a_vec <- c(2, 3, 4)
  gpu_a <- as.gpuVector(a_vec)
  result <- gpu_dot_vectors(gpu_a, gpu_a)
  expected <- sum(a_vec^2)
  expect_equal(result, expected)
})

test_that("chained gpuVector operations work correctly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test chained operations: (a + b) * c * scalar
  a_vec <- c(1, 2, 3, 4)
  b_vec <- c(2, 3, 4, 5)
  c_vec <- c(1.5, 2.5, 3.5, 4.5)
  scalar <- 2.0
  
  # Expected result calculated on CPU
  expected <- (a_vec + b_vec) * c_vec * scalar
  
  # GPU computation
  gpu_a <- as.gpuVector(a_vec)
  gpu_b <- as.gpuVector(b_vec)
  gpu_c <- as.gpuVector(c_vec)
  
  gpu_temp <- gpu_a + gpu_b  # Addition from Sprint 2
  gpu_temp2 <- gpu_temp * gpu_c  # Element-wise multiplication
  gpu_result <- gpu_temp2 * scalar  # Scalar multiplication
  
  result <- as.vector(gpu_result)
  
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpuVector multiplication memory management works correctly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test that multiple operations don't cause memory leaks
  # (This is mainly to ensure proper RAII behavior)
  for (i in 1:100) {
    a_vec <- runif(1000)
    b_vec <- runif(1000)
    scalar <- runif(1)
    
    gpu_a <- as.gpuVector(a_vec)
    gpu_b <- as.gpuVector(b_vec)
    
    # Perform operations that create temporary objects
    gpu_result1 <- gpu_a * gpu_b
    gpu_result2 <- gpu_a * scalar
    gpu_dot_result <- gpu_dot_vectors(gpu_a, gpu_b)
    
    # Convert results back (this tests the full pipeline)
    result1 <- as.vector(gpu_result1)
    result2 <- as.vector(gpu_result2)
    
    # Verify correctness
    expect_equal(result1, a_vec * b_vec, tolerance = 1e-10)
    expect_equal(result2, a_vec * scalar, tolerance = 1e-10)
    expect_equal(gpu_dot_result, sum(a_vec * b_vec), tolerance = 1e-10)
  }
  
  # If we reach here without crashes, memory management is working
  expect_true(TRUE)
})

test_that("gpuVector operations handle edge cases properly", {
  # Skip if GPU not available
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test with single element vectors
  gpu_a <- as.gpuVector(5.0)
  gpu_b <- as.gpuVector(3.0)
  
  gpu_mult_result <- gpu_a * gpu_b
  gpu_scale_result <- gpu_a * 2.5
  dot_result <- gpu_dot_vectors(gpu_a, gpu_b)
  
  expect_equal(as.vector(gpu_mult_result), 15.0)
  expect_equal(as.vector(gpu_scale_result), 12.5)
  expect_equal(dot_result, 15.0)
  
  # Test with special floating point values
  special_vec <- c(1.0, Inf, -Inf, 0.0)
  gpu_special <- as.gpuVector(special_vec)
  gpu_ones <- as.gpuVector(c(1.0, 1.0, 1.0, 1.0))
  
  gpu_result <- gpu_special * gpu_ones
  result <- as.vector(gpu_result)
  expected <- special_vec * c(1.0, 1.0, 1.0, 1.0)
  
  expect_equal(result, expected)
}) 