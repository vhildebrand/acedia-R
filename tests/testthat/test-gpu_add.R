test_that("gpu_add works correctly", {
  # Test basic functionality with small vectors
  a <- c(1, 2, 3, 4, 5)
  b <- c(2, 3, 4, 5, 6)
  expected <- c(3, 5, 7, 9, 11)
  
  result <- gpu_add(a, b)
  expect_equal(result, expected)
  expect_equal(result, a + b)
})

test_that("gpu_add handles larger vectors", {
  # Test with larger vectors
  n <- 10000
  a <- runif(n)
  b <- runif(n)
  
  result <- gpu_add(a, b)
  expected <- a + b
  
  expect_equal(length(result), n)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpu_add input validation works", {
  # Test error handling for non-numeric inputs
  expect_error(gpu_add("a", c(1, 2, 3)), "Both arguments must be numeric vectors")
  expect_error(gpu_add(c(1, 2, 3), "b"), "Both arguments must be numeric vectors")
  
  # Test error handling for mismatched lengths
  expect_error(gpu_add(c(1, 2, 3), c(1, 2)), "Input vectors must have the same length")
  expect_error(gpu_add(c(1, 2), c(1, 2, 3, 4)), "Input vectors must have the same length")
})

test_that("gpu_add handles edge cases", {
  # Test empty vectors
  expect_equal(gpu_add(numeric(0), numeric(0)), numeric(0))
  
  # Test single element vectors
  expect_equal(gpu_add(5, 3), 8)
  
  # Test vectors with special values
  a <- c(1, 2, Inf, -Inf, 0)
  b <- c(2, 3, 1, 1, 0)
  result <- gpu_add(a, b)
  expected <- a + b
  
  expect_equal(result, expected)
})

test_that("gpu_add performance comparison", {
  # Skip if microbenchmark is not available
  skip_if_not_installed("microbenchmark")
  
  n <- 1e6
  a <- runif(n)
  b <- runif(n)
  
  # Ensure correctness first
  gpu_result <- gpu_add(a, b)
  cpu_result <- a + b
  expect_equal(gpu_result, cpu_result, tolerance = 1e-10)
  
  # Basic timing comparison (not strict performance test)
  cpu_time <- system.time(a + b)[["elapsed"]]
  gpu_time <- system.time(gpu_add(a, b))[["elapsed"]]
  
  # Just ensure both complete successfully
  expect_true(cpu_time > 0)
  expect_true(gpu_time > 0)
}) 