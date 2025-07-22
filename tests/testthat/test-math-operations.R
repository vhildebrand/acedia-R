# Comprehensive tests for unary math operations and reductions

test_that("exp() operation works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic exp
  t1 <- gpu_tensor(c(0, 1, 2), c(3))
  result <- exp(t1)
  expected <- exp(c(0, 1, 2))
  
  expect_equal(as.vector(result), expected, tolerance = 1e-6)
  expect_equal(shape(result), c(3))
  expect_true(inherits(result, "gpuTensor"))
  
  # Test exp with different shapes
  t2 <- gpu_tensor(c(0, 1, 2, 3), c(2, 2))
  result2 <- exp(t2)
  expected2 <- exp(c(0, 1, 2, 3))
  
  expect_equal(as.vector(result2), expected2, tolerance = 1e-6)
  expect_equal(shape(result2), c(2, 2))
})

test_that("log() operation works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic log
  t1 <- gpu_tensor(c(1, 2.718, 7.389), c(3))
  result <- log(t1)
  expected <- log(c(1, 2.718, 7.389))
  
  expect_equal(as.vector(result), expected, tolerance = 1e-3)
  expect_equal(shape(result), c(3))
  expect_true(inherits(result, "gpuTensor"))
  
  # Test log with matrix
  t2 <- gpu_tensor(c(1, 2, 4, 8), c(2, 2))
  result2 <- log(t2)
  expected2 <- log(c(1, 2, 4, 8))
  
  expect_equal(as.vector(result2), expected2, tolerance = 1e-6)
})

test_that("sqrt() operation works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic sqrt
  t1 <- gpu_tensor(c(1, 4, 9, 16), c(4))
  result <- sqrt(t1)
  expected <- c(1, 2, 3, 4)
  
  expect_equal(as.vector(result), expected, tolerance = 1e-6)
  expect_equal(shape(result), c(4))
  expect_true(inherits(result, "gpuTensor"))
  
  # Test sqrt with 2D tensor
  t2 <- gpu_tensor(c(0, 1, 4, 9), c(2, 2))
  result2 <- sqrt(t2)
  expected2 <- c(0, 1, 2, 3)
  
  expect_equal(as.vector(result2), expected2, tolerance = 1e-6)
})

test_that("mean() reduction works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic mean
  t1 <- gpu_tensor(c(1, 2, 3, 4), c(4))
  result <- mean(t1)
  expected <- mean(c(1, 2, 3, 4))
  
  expect_equal(result, expected, tolerance = 1e-6)
  expect_true(is.numeric(result))
  expect_length(result, 1)
  
  # Test mean with 2D tensor
  t2 <- gpu_tensor(c(1, 2, 3, 4, 5, 6), c(2, 3))
  result2 <- mean(t2)
  expected2 <- mean(c(1, 2, 3, 4, 5, 6))
  
  expect_equal(result2, expected2, tolerance = 1e-6)
})

test_that("max() reduction works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic max
  t1 <- gpu_tensor(c(3, 1, 4, 2), c(4))
  result <- max(t1)
  expected <- 4
  
  expect_equal(result, expected, tolerance = 1e-6)
  expect_true(is.numeric(result))
  expect_length(result, 1)
  
  # Test max with negative values
  t2 <- gpu_tensor(c(-5, -2, -8, -1), c(2, 2))
  result2 <- max(t2)
  expected2 <- -1
  
  expect_equal(result2, expected2, tolerance = 1e-6)
})

test_that("min() reduction works correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test basic min
  t1 <- gpu_tensor(c(3, 1, 4, 2), c(4))
  result <- min(t1)
  expected <- 1
  
  expect_equal(result, expected, tolerance = 1e-6)
  expect_true(is.numeric(result))
  expect_length(result, 1)
  
  # Test min with negative values
  t2 <- gpu_tensor(c(-5, -2, -8, -1), c(2, 2))
  result2 <- min(t2)
  expected2 <- -8
  
  expect_equal(result2, expected2, tolerance = 1e-6)
})

test_that("unary operations work with non-contiguous tensors", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Create non-contiguous tensor via transpose
  t1 <- gpu_tensor(c(1, 4, 9, 16, 25, 36), c(2, 3))
  t1_transposed <- transpose(t1)
  
  # Test that unary operations work on non-contiguous tensors
  expect_no_error(result_exp <- exp(t1_transposed))
  expect_no_error(result_log <- log(t1_transposed))
  expect_no_error(result_sqrt <- sqrt(t1_transposed))
  
  # Verify results are correct
  original_data <- as.vector(t1_transposed)
  expect_equal(as.vector(result_exp), exp(original_data), tolerance = 1e-6)
  expect_equal(as.vector(result_sqrt), sqrt(original_data), tolerance = 1e-6)
})

test_that("reduction operations work with non-contiguous tensors", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Create non-contiguous tensor
  t1 <- gpu_tensor(c(1, 2, 3, 4, 5, 6), c(2, 3))
  t1_transposed <- transpose(t1)
  
  # Test reductions on non-contiguous tensors
  expect_no_error(result_sum <- sum(t1_transposed))
  expect_no_error(result_mean <- mean(t1_transposed))
  expect_no_error(result_max <- max(t1_transposed))
  expect_no_error(result_min <- min(t1_transposed))
  
  # Verify results are correct
  original_data <- as.vector(t1_transposed)
  expect_equal(result_sum, sum(original_data), tolerance = 1e-6)
  expect_equal(result_mean, mean(original_data), tolerance = 1e-6)
  expect_equal(result_max, max(original_data), tolerance = 1e-6)
  expect_equal(result_min, min(original_data), tolerance = 1e-6)
}) 