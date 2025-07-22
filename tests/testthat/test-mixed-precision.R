test_that("mixed precision tensor creation works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test float16 tensor creation
  t1 <- gpu_tensor(c(1.0, 2.0, 3.0), c(3), dtype='float16')
  expect_s3_class(t1, "gpuTensor")
  expect_equal(as.array(t1), c(1, 2, 3))
  
  # Test float32 tensor creation
  t2 <- gpu_tensor(c(1.0, 2.0, 3.0), c(3), dtype='float32')
  expect_s3_class(t2, "gpuTensor")
  expect_equal(as.array(t2), c(1, 2, 3))
  
  # Test float64 tensor creation
  t3 <- gpu_tensor(c(1.0, 2.0, 3.0), c(3), dtype='float64')
  expect_s3_class(t3, "gpuTensor")
  expect_equal(as.array(t3), c(1, 2, 3))
  
  # Test int8 tensor creation
  t4 <- gpu_tensor(c(1, 2, 3), c(3), dtype='int8')
  expect_s3_class(t4, "gpuTensor")
  expect_equal(as.array(t4), c(1, 2, 3))
  
  # Test int32 tensor creation
  t5 <- gpu_tensor(c(1, 2, 3), c(3), dtype='int32')
  expect_s3_class(t5, "gpuTensor")
  expect_equal(as.array(t5), c(1, 2, 3))
  
  # Test int64 tensor creation
  t6 <- gpu_tensor(c(1, 2, 3), c(3), dtype='int64')
  expect_s3_class(t6, "gpuTensor")
  expect_equal(as.array(t6), c(1, 2, 3))
})

test_that("tensor dtypes are correctly detected", {
  skip_if_not(gpu_available(), "GPU not available")
  
  t1 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float16')
  t2 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float32')
  t3 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float64')
  
  # Test dtype detection (this would require implementing tensor_dtype in R)
  # expect_equal(dtype(t1), "float16")  # Future implementation
  # expect_equal(dtype(t2), "float32")
  # expect_equal(dtype(t3), "float64")
})

test_that("scalar operations work with different precisions", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test scalar multiplication with different dtypes
  t1 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float16')
  result1 <- t1 * 2.0
  expect_equal(as.array(result1), c(2, 4, 6))
  
  t2 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float32')
  result2 <- t2 * 2.0
  expect_equal(as.array(result2), c(2, 4, 6))
  
  t3 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float64')
  result3 <- t3 * 2.0
  expect_equal(as.array(result3), c(2, 4, 6))
})

test_that("sum operations work with different precisions", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test sum with different dtypes
  t1 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float16')
  expect_equal(sum(t1), 6)
  
  t2 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float32')
  expect_equal(sum(t2), 6)
  
  t3 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float64')
  expect_equal(sum(t3), 6)
  
  t4 <- gpu_tensor(c(1, 2, 3), c(3), dtype='int64')
  expect_equal(sum(t4), 6)
})

test_that("same dtype tensor operations work", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test addition with same dtypes
  t1 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float32')
  t2 <- gpu_tensor(c(4, 5, 6), c(3), dtype='float32')
  result <- t1 + t2
  expect_equal(as.array(result), c(5, 7, 9))
})

test_that("mixed precision error handling works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test that mixed dtype operations are properly detected
  t1 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float16')
  t2 <- gpu_tensor(c(1, 2, 3), c(3), dtype='float32')
  
  # Should error until type promotion is implemented
  expect_error(t1 + t2, "different dtypes")
})

test_that("precision preservation works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test that operations preserve precision appropriately
  original_data <- c(1.234567, 2.345678, 3.456789)
  
  # Float16 should have lower precision
  t1 <- gpu_tensor(original_data, c(3), dtype='float16')
  result1 <- as.array(t1)
  
  # Float32 should have better precision  
  t2 <- gpu_tensor(original_data, c(3), dtype='float32')
  result2 <- as.array(t2)
  
  # Float64 should have full precision
  t3 <- gpu_tensor(original_data, c(3), dtype='float64')
  result3 <- as.array(t3)
  
  # Float64 should be closest to original
  expect_true(all(abs(result3 - original_data) < abs(result1 - original_data)))
})

test_that("large tensor mixed precision works", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test with larger tensors to ensure memory handling works
  large_data <- runif(1000, 0, 10)
  
  t1 <- gpu_tensor(large_data, c(10, 10, 10), dtype='float16')
  t2 <- gpu_tensor(large_data, c(10, 10, 10), dtype='float32')
  
  expect_equal(length(as.array(t1)), 1000)
  expect_equal(length(as.array(t2)), 1000)
  expect_equal(dim(as.array(t1)), c(10, 10, 10))
  expect_equal(dim(as.array(t2)), c(10, 10, 10))
}) 