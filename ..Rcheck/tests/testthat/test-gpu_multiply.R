test_that("gpu_multiply works correctly", {
  # Test basic functionality with small vectors
  a <- c(2, 3, 4, 5, 6)
  b <- c(3, 4, 5, 6, 7)
  expected <- c(6, 12, 20, 30, 42)
  
  result <- gpu_multiply(a, b)
  expect_equal(result, expected)
  expect_equal(result, a * b)
})

test_that("gpu_multiply handles larger vectors", {
  # Test with larger vectors
  n <- 10000
  a <- runif(n)
  b <- runif(n)
  
  result <- gpu_multiply(a, b)
  expected <- a * b
  
  expect_equal(length(result), n)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpu_multiply input validation works", {
  # Test error handling for non-numeric inputs
  expect_error(gpu_multiply("a", c(1, 2, 3)), "Both arguments must be numeric vectors")
  expect_error(gpu_multiply(c(1, 2, 3), "b"), "Both arguments must be numeric vectors")
  
  # Test error handling for mismatched lengths
  expect_error(gpu_multiply(c(1, 2, 3), c(1, 2)), "Input vectors must have the same length")
  expect_error(gpu_multiply(c(1, 2), c(1, 2, 3, 4)), "Input vectors must have the same length")
})

test_that("gpu_multiply handles edge cases", {
  # Test empty vectors
  expect_equal(gpu_multiply(numeric(0), numeric(0)), numeric(0))
  
  # Test single element vectors
  expect_equal(gpu_multiply(5, 3), 15)
  
  # Test vectors with special values
  a <- c(1, 2, Inf, -Inf, 0)
  b <- c(2, 3, 1, 1, 5)
  result <- gpu_multiply(a, b)
  expected <- a * b
  
  expect_equal(result, expected)
})

test_that("gpu_multiply with zeros and ones", {
  # Test multiplication by zero
  a <- c(1, 2, 3, 4, 5)
  b <- c(0, 0, 0, 0, 0)
  result <- gpu_multiply(a, b)
  expect_equal(result, c(0, 0, 0, 0, 0))
  
  # Test multiplication by one
  a <- c(1, 2, 3, 4, 5)
  b <- c(1, 1, 1, 1, 1)
  result <- gpu_multiply(a, b)
  expect_equal(result, a)
})

test_that("gpu_scale works correctly", {
  # Test basic functionality with small vector
  x <- c(1, 2, 3, 4, 5)
  scalar <- 3
  expected <- c(3, 6, 9, 12, 15)
  
  result <- gpu_scale(x, scalar)
  expect_equal(result, expected)
  expect_equal(result, x * scalar)
})

test_that("gpu_scale handles larger vectors", {
  # Test with larger vectors
  n <- 10000
  x <- runif(n)
  scalar <- 2.5
  
  result <- gpu_scale(x, scalar)
  expected <- x * scalar
  
  expect_equal(length(result), n)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpu_scale input validation works", {
  # Test error handling for non-numeric inputs
  expect_error(gpu_scale("a", 2), "x must be a numeric vector")
  expect_error(gpu_scale(c(1, 2, 3), "b"), "scalar must be a single numeric value")
  
  # Test error handling for non-scalar input
  expect_error(gpu_scale(c(1, 2, 3), c(1, 2)), "scalar must be a single numeric value")
  expect_error(gpu_scale(c(1, 2, 3), numeric(0)), "scalar must be a single numeric value")
})

test_that("gpu_scale handles edge cases", {
  # Test empty vector
  expect_equal(gpu_scale(numeric(0), 5), numeric(0))
  
  # Test single element vector
  expect_equal(gpu_scale(7, 3), 21)
  
  # Test scaling by zero
  x <- c(1, 2, 3, 4, 5)
  result <- gpu_scale(x, 0)
  expect_equal(result, c(0, 0, 0, 0, 0))
  
  # Test scaling by one
  x <- c(1, 2, 3, 4, 5)
  result <- gpu_scale(x, 1)
  expect_equal(result, x)
  
  # Test scaling by negative value
  x <- c(1, 2, 3)
  result <- gpu_scale(x, -2)
  expect_equal(result, c(-2, -4, -6))
  
  # Test with special values
  x <- c(1, Inf, -Inf, 0)
  scalar <- 2
  result <- gpu_scale(x, scalar)
  expected <- x * scalar
  expect_equal(result, expected)
})

test_that("gpu_dot works correctly", {
  # Test basic functionality with small vectors
  a <- c(1, 2, 3, 4)
  b <- c(2, 3, 4, 5)
  expected <- sum(a * b)  # 2 + 6 + 12 + 20 = 40
  
  result <- gpu_dot(a, b)
  expect_equal(result, expected)
  expect_equal(result, 40)
})

test_that("gpu_dot handles larger vectors", {
  # Test with larger vectors
  n <- 10000
  a <- runif(n)
  b <- runif(n)
  
  result <- gpu_dot(a, b)
  expected <- sum(a * b)
  
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpu_dot input validation works", {
  # Test error handling for non-numeric inputs
  expect_error(gpu_dot("a", c(1, 2, 3)), "Both arguments must be numeric vectors")
  expect_error(gpu_dot(c(1, 2, 3), "b"), "Both arguments must be numeric vectors")
  
  # Test error handling for mismatched lengths
  expect_error(gpu_dot(c(1, 2, 3), c(1, 2)), "Input vectors must have the same length")
  expect_error(gpu_dot(c(1, 2), c(1, 2, 3, 4)), "Input vectors must have the same length")
})

test_that("gpu_dot handles edge cases", {
  # Test empty vectors
  expect_equal(gpu_dot(numeric(0), numeric(0)), 0.0)
  
  # Test single element vectors
  expect_equal(gpu_dot(5, 3), 15)
  
  # Test orthogonal vectors
  a <- c(1, 0)
  b <- c(0, 1)
  result <- gpu_dot(a, b)
  expect_equal(result, 0)
  
  # Test identical vectors
  a <- c(2, 3, 4)
  result <- gpu_dot(a, a)
  expected <- sum(a^2)
  expect_equal(result, expected)
  
  # Test with zeros
  a <- c(1, 2, 3)
  b <- c(0, 0, 0)
  result <- gpu_dot(a, b)
  expect_equal(result, 0)
})

test_that("gpu operations performance comparison", {
  # Skip if microbenchmark is not available
  skip_if_not_installed("microbenchmark")
  
  n <- 1e6
  a <- runif(n)
  b <- runif(n)
  scalar <- 3.14
  
  # Ensure correctness first
  gpu_mult_result <- gpu_multiply(a, b)
  cpu_mult_result <- a * b
  expect_equal(gpu_mult_result, cpu_mult_result, tolerance = 1e-10)
  
  gpu_scale_result <- gpu_scale(a, scalar)
  cpu_scale_result <- a * scalar
  expect_equal(gpu_scale_result, cpu_scale_result, tolerance = 1e-10)
  
  gpu_dot_result <- gpu_dot(a, b)
  cpu_dot_result <- sum(a * b)
  expect_equal(gpu_dot_result, cpu_dot_result, tolerance = 1e-10)
  
  # Basic timing comparison (not strict performance test)
  cpu_mult_time <- system.time(a * b)[["elapsed"]]
  gpu_mult_time <- system.time(gpu_multiply(a, b))[["elapsed"]]
  
  cpu_dot_time <- system.time(sum(a * b))[["elapsed"]]
  gpu_dot_time <- system.time(gpu_dot(a, b))[["elapsed"]]
  
  # Just ensure both complete successfully
  expect_true(cpu_mult_time > 0)
  expect_true(gpu_mult_time > 0)
  expect_true(cpu_dot_time > 0)
  expect_true(gpu_dot_time > 0)
})

test_that("gpu operations force_cpu option works", {
  # Test force_cpu flag for all operations
  a <- c(1, 2, 3, 4, 5)
  b <- c(2, 3, 4, 5, 6)
  scalar <- 2.5
  
  # Test multiplication
  result_cpu <- gpu_multiply(a, b, force_cpu = TRUE)
  result_normal <- gpu_multiply(a, b)
  expect_equal(result_cpu, result_normal)
  expect_equal(result_cpu, a * b)
  
  # Test scaling
  result_cpu <- gpu_scale(a, scalar, force_cpu = TRUE)
  result_normal <- gpu_scale(a, scalar)
  expect_equal(result_cpu, result_normal)
  expect_equal(result_cpu, a * scalar)
  
  # Test dot product
  result_cpu <- gpu_dot(a, b, force_cpu = TRUE)
  result_normal <- gpu_dot(a, b)
  expect_equal(result_cpu, result_normal)
  expect_equal(result_cpu, sum(a * b))
}) 