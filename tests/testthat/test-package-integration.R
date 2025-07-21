test_that("package loads and provides expected functionality", {
  # Test that all main functions are available after library load
  expect_true(exists("gpu_add"))
  expect_true(exists("as.gpuVector"))
  expect_true(exists("gpu_available"))
  expect_true(exists("gpu_status"))
})

test_that("complete user workflow works end-to-end", {
  # Simulate a typical user session
  
  # 1. Check GPU status
  status <- gpu_status()
  expect_type(status$available, "logical")
  expect_type(status$memory_gb, "double")
  
  # 2. Perform basic operation
  a <- c(1, 2, 3, 4, 5)
  b <- c(2, 3, 4, 5, 6)
  result <- gpu_add(a, b)
  expect_equal(result, c(3, 5, 7, 9, 11))
  
  # 3. Use advanced GPU objects (if available)
  if (status$available) {
    gpu_a <- as.gpuVector(a)
    gpu_b <- as.gpuVector(b)
    gpu_result <- gpu_a + gpu_b
    final_result <- as.vector(gpu_result)
    expect_equal(final_result, c(3, 5, 7, 9, 11))
  }
})

test_that("package handles edge cases gracefully", {
  # Empty vectors
  expect_equal(gpu_add(numeric(0), numeric(0)), numeric(0))
  
  # Large vectors
  n <- 10000
  large_a <- runif(n)
  large_b <- runif(n)
  large_result <- gpu_add(large_a, large_b)
  expect_length(large_result, n)
  expect_equal(large_result, large_a + large_b, tolerance = 1e-14)
}) 