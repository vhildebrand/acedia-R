context("gpuTensor slicing")

skip_if_not(gpu_available(), "GPU not available")

test_that("basic 1D slicing works", {
  t <- gpu_tensor(1:10, c(10))
  
  # Single element
  expect_equal(t[5], 5)
  
  # Range
  result <- t[3:6]
  expect_equal(as.vector(result), 3:6)
  expect_equal(dim(result), 4)
})

test_that("2D slicing works", {
  t <- gpu_tensor(1:12, c(3, 4))
  
  # Single row
  result <- t[2, ]
  expect_equal(as.vector(result), c(2, 5, 8, 11))
  expect_equal(dim(result), 4)
  
  # Single column  
  result <- t[, 3]
  expect_equal(as.vector(result), c(7, 8, 9))
  expect_equal(dim(result), 3)
  
  # Single element
  expect_equal(t[2, 3], 8)
  
  # Submatrix
  result <- t[1:2, 2:3]
  expect_equal(dim(result), c(2, 2))
  expect_equal(as.vector(result), c(4, 5, 7, 8))
})

test_that("3D slicing works", {
  t <- gpu_tensor(1:24, c(2, 3, 4))
  
  # Single slice along first dimension
  result <- t[1, , ]
  expect_equal(dim(result), c(3, 4))
  
  # Single element
  expect_equal(t[1, 2, 3], 15)  # Element at row 1, col 2, depth 3
  
  # Partial slice
  result <- t[, 1:2, ]
  expect_equal(dim(result), c(2, 2, 4))
})

test_that("slicing validates indices", {
  t <- gpu_tensor(1:6, c(2, 3))
  
  # Out of bounds
  expect_error(t[5, ], "out of bounds")
  expect_error(t[, 5], "out of bounds")
  
  # Negative indices not supported
  expect_error(t[-1, ], "Only positive indices are supported")
  expect_error(t[, -1], "Only positive indices are supported")
  
  # Zero indices not supported  
  expect_error(t[0, ], "Only positive indices are supported")
})

test_that("slicing handles edge cases", {
  t <- gpu_tensor(1:6, c(2, 3))
  
  # Empty slice (all indices)
  result <- t[]
  expect_equal(as.array(result), as.array(t))
  
  # Too many indices
  expect_error(t[1, 1, 1, 1], "Too many indices")
})

test_that("slicing preserves data types", {
  # Test with different dtypes
  t_float <- gpu_tensor(c(1.5, 2.5, 3.5, 4.5), c(2, 2), dtype="float")
  result <- t_float[1, ]
  expect_equal(as.vector(result), c(1.5, 3.5))
  
  t_double <- gpu_tensor(c(1.5, 2.5, 3.5, 4.5), c(2, 2), dtype="double")  
  result <- t_double[, 1]
  expect_equal(as.vector(result), c(1.5, 2.5))
})

test_that("contiguous ranges work", {
  t <- gpu_tensor(1:10, c(10))
  
  # Contiguous range
  result <- t[3:7]
  expect_equal(as.vector(result), 3:7)
  
  # Non-contiguous should error (for now)
  expect_error(t[c(1, 3, 5)], "Only contiguous ranges are supported")
}) 