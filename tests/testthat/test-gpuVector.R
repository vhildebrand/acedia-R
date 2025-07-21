test_that("gpuVector creation from R vector works", {
  # Test basic creation
  x <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  gpu_x <- as.gpuVector(x)
  
  expect_true(inherits(gpu_x, "gpuVector"))
  expect_equal(class(gpu_x), "gpuVector")
  
  # Test empty vector
  empty_x <- numeric(0)
  gpu_empty <- as.gpuVector(empty_x)
  expect_true(inherits(gpu_empty, "gpuVector"))
  expect_equal(class(gpu_empty), "gpuVector")
})

test_that("gpuVector to R vector conversion works", {
  # Test basic conversion
  x <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  gpu_x <- as.gpuVector(x)
  y <- as.vector(gpu_x)
  
  expect_equal(y, x, tolerance = 1e-10)
  expect_type(y, "double")
  
  # Test empty vector conversion
  empty_x <- numeric(0)
  gpu_empty <- as.gpuVector(empty_x)
  y_empty <- as.vector(gpu_empty)
  
  expect_equal(y_empty, empty_x)
  expect_length(y_empty, 0)
})

test_that("gpuVector roundtrip preserves data", {
  # Test various vector sizes
  test_vectors <- list(
    small = c(1.0, 2.0, 3.0),
    medium = runif(100),
    large = runif(10000),
    single = 42.0,
    negative = c(-1.0, -2.5, -3.14159),
    mixed = c(-5.0, 0.0, 5.0, 1e-10, 1e10)
  )
  
  for (name in names(test_vectors)) {
    x <- test_vectors[[name]]
    gpu_x <- as.gpuVector(x)
    y <- as.vector(gpu_x)
    
    expect_equal(y, x, tolerance = 1e-10, 
                 info = paste("Failed for", name, "vector"))
  }
})

test_that("gpu_add_vectors works correctly", {
  # Test basic vector addition
  a <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  b <- c(2.0, 3.0, 4.0, 5.0, 6.0)
  expected <- a + b
  
  gpu_a <- as.gpuVector(a)
  gpu_b <- as.gpuVector(b)
  gpu_result <- gpu_add_vectors(gpu_a, gpu_b)
  result <- as.vector(gpu_result)
  
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpuVector addition operator works", {
  # Test + operator overload
  a <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  b <- c(2.0, 3.0, 4.0, 5.0, 6.0)
  expected <- a + b
  
  gpu_a <- as.gpuVector(a)
  gpu_b <- as.gpuVector(b)
  gpu_result <- gpu_a + gpu_b
  result <- as.vector(gpu_result)
  
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("gpu_add with use_gpuvector=TRUE works", {
  # Test the enhanced gpu_add function
  a <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  b <- c(2.0, 3.0, 4.0, 5.0, 6.0)
  expected <- a + b
  
  # Test new implementation
  result_new <- gpu_add(a, b, use_gpuvector = TRUE)
  expect_equal(result_new, expected, tolerance = 1e-10)
  
  # Test backward compatibility (old implementation)
  result_old <- gpu_add(a, b, use_gpuvector = FALSE)
  expect_equal(result_old, expected, tolerance = 1e-10)
  
  # Results should be identical
  expect_equal(result_new, result_old, tolerance = 1e-10)
})

test_that("gpuVector handles large vectors", {
  # Test performance with larger vectors
  n <- 100000
  a <- runif(n)
  b <- runif(n)
  expected <- a + b
  
  gpu_a <- as.gpuVector(a)
  gpu_b <- as.gpuVector(b)
  gpu_result <- gpu_add_vectors(gpu_a, gpu_b)
  result <- as.vector(gpu_result)
  
  expect_equal(result, expected, tolerance = 1e-10)
  expect_length(result, n)
})

test_that("error handling works correctly", {
  # Test non-numeric input
  expect_error(as.gpuVector("not_numeric"), "Input must be a numeric vector")
  expect_error(as.gpuVector(c(TRUE, FALSE)), "Input must be a numeric vector")
  
  # Test size mismatch in addition (this error is handled in C++ code)
  a <- as.gpuVector(c(1.0, 2.0, 3.0))
  b <- as.gpuVector(c(1.0, 2.0))
  
  expect_error(gpu_add_vectors(a, b))
})

test_that("gpuVector print method works", {
  # Test print functionality (mostly just ensure it doesn't crash)
  x <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  gpu_x <- as.gpuVector(x)
  
  # Capture output to ensure print method is called
  output <- capture.output(print(gpu_x))
  expect_true(any(grepl("gpuVector", output)))
})

test_that("empty gpuVector operations work", {
  # Test operations with empty vectors
  empty_a <- as.gpuVector(numeric(0))
  empty_b <- as.gpuVector(numeric(0))
  
  result <- gpu_add_vectors(empty_a, empty_b)
  result_vec <- as.vector(result)
  
  expect_length(result_vec, 0)
  expect_type(result_vec, "double")
})

test_that("gpuVector memory management is robust", {
  # Test that multiple operations don't cause memory issues
  n <- 1000
  x <- runif(n)
  
  # Create many gpuVectors and let them go out of scope
  for (i in 1:10) {
    gpu_x <- as.gpuVector(x)
    gpu_y <- as.gpuVector(x)
    gpu_result <- gpu_x + gpu_y
    result <- as.vector(gpu_result)
    
    expect_length(result, n)
    expect_equal(result, x + x, tolerance = 1e-10)
  }
}) 