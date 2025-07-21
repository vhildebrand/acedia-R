test_that("gpu_add vs gpuVector approaches work and produce identical results", {
  # Test both architectural approaches with various vector sizes
  
  test_cases <- list(
    small = c(1, 2, 3),
    medium = runif(1000),
    large = runif(50000),
    negative = c(-5.5, -2.1, -0.1, 0.0, 0.1, 2.1, 5.5),
    single = 42.0,
    precision = c(1.123456789012345, 2.987654321098765)
  )
  
  for (name in names(test_cases)) {
    a <- test_cases[[name]]
    b <- rev(a)  # Different but same length
    
    # CPU reference
    cpu_result <- a + b
    
    # Simple GPU function (CPU↔GPU)
    simple_result <- gpu_add(a, b)
    
    # GPU object approach (GPU-resident)
    gpu_a <- as.gpuVector(a)
    gpu_b <- as.gpuVector(b)
    gpu_result <- gpu_a + gpu_b
    object_result <- as.vector(gpu_result)
    
    # All should be identical
    expect_equal(simple_result, cpu_result, tolerance = 1e-15, 
                 info = paste("Simple GPU function failed for", name, "vectors"))
    expect_equal(object_result, cpu_result, tolerance = 1e-15,
                 info = paste("GPU object approach failed for", name, "vectors"))
    expect_equal(simple_result, object_result, tolerance = 1e-15,
                 info = paste("Simple vs Object approaches differ for", name, "vectors"))
  }
})

test_that("gpu_add and gpuVector error handling works consistently", {
  # Test that both approaches handle errors properly
  
  # Non-numeric inputs  
  expect_error(gpu_add("a", c(1, 2)), 
               "Both arguments must be numeric vectors")
  expect_error(as.gpuVector("not_numeric"), 
               "Input must be a numeric vector")
  
  # Size mismatches  
  expect_error(gpu_add(c(1, 2, 3), c(1, 2)),
               "Input vectors must have the same length")
  
  # Size mismatch for gpuVector objects
  gpu_a <- as.gpuVector(c(1, 2, 3))
  gpu_b <- as.gpuVector(c(1, 2))
  expect_error(gpu_a + gpu_b, "sizes must match")
  
  # Empty vectors (should work)
  expect_equal(gpu_add(numeric(0), numeric(0)), numeric(0))
  
  empty_a <- as.gpuVector(numeric(0))
  empty_b <- as.gpuVector(numeric(0))
  empty_result <- empty_a + empty_b
  expect_equal(as.vector(empty_result), numeric(0))
})

test_that("gpu_add vs gpuVector performance comparison", {
  # Test that both approaches can handle large vectors efficiently
  n <- 100000
  a <- runif(n)
  b <- runif(n)
  cpu_result <- a + b
  
  # Time simple GPU function (includes CPU↔GPU transfers)
  start_time <- Sys.time()
  simple_result <- gpu_add(a, b)
  simple_time <- as.numeric(Sys.time() - start_time)
  
  # Time GPU object approach (also includes transfers for fair comparison)
  start_time <- Sys.time()
  gpu_a <- as.gpuVector(a)
  gpu_b <- as.gpuVector(b)
  gpu_result <- gpu_a + gpu_b
  object_result <- as.vector(gpu_result)
  object_time <- as.numeric(Sys.time() - start_time)
  
  # Both should be correct
  expect_equal(simple_result, cpu_result, tolerance = 1e-14)
  expect_equal(object_result, cpu_result, tolerance = 1e-14)
  expect_equal(simple_result, object_result, tolerance = 1e-15)
  
  # Both should complete in reasonable time (< 1 second for 100k elements)
  expect_lt(simple_time, 1.0)
  expect_lt(object_time, 1.0)
  
  cat("Simple GPU function time:", simple_time, "seconds\n")
  cat("GPU object approach time:", object_time, "seconds\n")
})

test_that("clean architecture: each approach serves its purpose", {
  # Verify the clean separation of concerns
  a <- c(1, 2, 3)
  b <- c(4, 5, 6)
  expected <- c(5, 7, 9)
  
  # Simple approach: good for one-off operations  
  simple_result <- gpu_add(a, b)
  expect_equal(simple_result, expected)
  
  # Object approach: good for chained operations
  gpu_a <- as.gpuVector(a)
  gpu_b <- as.gpuVector(b)
  
  # Can chain multiple operations efficiently
  gpu_temp <- gpu_a + gpu_b
  gpu_final <- gpu_temp + gpu_a  # Another operation on same GPU data
  final_result <- as.vector(gpu_final)
  
  expect_equal(final_result, expected + a)  # [6, 9, 12]
})

test_that("edge cases work correctly in both approaches", {
  # Test edge cases to ensure robustness
  
  edge_cases <- list(
    "very_small" = c(1e-15, 2e-15),
    "very_large" = c(1e15, 2e15), 
    "mixed_magnitudes" = c(1e-10, 1e10),
    "repeated_values" = rep(42.0, 1000),
    "alternating" = rep(c(-1, 1), 500)
  )
  
  for (name in names(edge_cases)) {
    a <- edge_cases[[name]]
    b <- rev(a)
    
    cpu_result <- a + b
    simple_result <- gpu_add(a, b)
    
    gpu_a <- as.gpuVector(a)
    gpu_b <- as.gpuVector(b)
    gpu_result <- gpu_a + gpu_b
    object_result <- as.vector(gpu_result)
    
    expect_equal(simple_result, cpu_result, tolerance = 1e-14,
                 info = paste("Simple GPU failed for edge case:", name))
    expect_equal(object_result, cpu_result, tolerance = 1e-14, 
                 info = paste("GPU object failed for edge case:", name))
  }
}) 