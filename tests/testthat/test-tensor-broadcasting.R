context("Tensor broadcasting")

skip_if_not(gpu_available(), "GPU not available")

test_that("2D broadcasting works for addition", {
  # (2,1) + (1,3) -> (2,3)
  a <- gpu_tensor(c(1, 2), c(2, 1))
  b <- gpu_tensor(c(10, 20, 30), c(1, 3))
  
  result <- a + b
  expected <- matrix(c(11, 12, 21, 22, 31, 32), nrow=2, ncol=3)
  
  expect_equal(dim(result), c(2, 3))
  expect_equal(as.array(result), expected, tolerance=1e-10)
})

test_that("2D broadcasting works for multiplication", {
  # (3,1) * (1,2) -> (3,2)  
  a <- gpu_tensor(c(2, 3, 4), c(3, 1))
  b <- gpu_tensor(c(10, 100), c(1, 2))
  
  result <- a * b
  expected <- matrix(c(20, 30, 40, 200, 300, 400), nrow=3, ncol=2)
  
  expect_equal(dim(result), c(3, 2))
  expect_equal(as.array(result), expected, tolerance=1e-10)
})

test_that("3D broadcasting works", {
  # (2,1,3) + (1,2,1) -> (2,2,3)
  a <- gpu_tensor(1:6, c(2, 1, 3))  # reshape [1,2,3,4,5,6] to (2,1,3)
  b <- gpu_tensor(c(10, 20), c(1, 2, 1))
  
  result <- a + b
  expect_equal(dim(result), c(2, 2, 3))
  
  # Verify a few key elements
  result_array <- as.array(result)
  expect_equal(result_array[1,1,1], 1 + 10)  # a[1,1,1] + b[1,1,1]
  expect_equal(result_array[1,2,1], 1 + 20)  # a[1,1,1] + b[1,2,1] 
  expect_equal(result_array[2,1,2], 4 + 10)  # a[2,1,2] + b[1,1,1] (a[2,1,2] = 4, not 5)
})

test_that("scalar tensor broadcasting still works", {
  # Size-1 tensor should use fast scalar path
  a <- gpu_tensor(5, c(1))
  b <- gpu_tensor(1:4, c(2, 2))
  
  result <- a + b
  expected <- matrix(6:9, nrow=2, ncol=2)
  
  expect_equal(as.array(result), expected, tolerance=1e-10)
})

test_that("vector broadcasting works", {
  # (4,) + (1,4) -> (1,4) (but both are effectively vectors)
  a <- gpu_tensor(1:4, c(4))
  b <- gpu_tensor(c(10, 20, 30, 40), c(1, 4))
  
  result <- a + b
  expected <- matrix(c(11, 22, 33, 44), nrow=1, ncol=4)
  
  expect_equal(dim(result), c(1, 4))
  expect_equal(as.array(result), expected, tolerance=1e-10)
})

test_that("incompatible shapes raise errors", {
  a <- gpu_tensor(1:6, c(2, 3))
  b <- gpu_tensor(1:4, c(2, 2))  # (2,3) vs (2,2) - not broadcastable
  
  expect_error(a + b, "not broadcastable")
  expect_error(a * b, "not broadcastable")
})

test_that("same shape tensors use fast path", {
  a <- gpu_tensor(1:6, c(2, 3))
  b <- gpu_tensor(10:15, c(2, 3))
  
  result <- a + b
  expected <- matrix(c(11, 13, 15, 17, 19, 21), nrow=2, ncol=3)
  
  expect_equal(as.array(result), expected, tolerance=1e-10)
})

test_that("broadcasting with different dtypes fails", {
  a <- gpu_tensor(1:4, c(2, 2), dtype="float")
  b <- gpu_tensor(1:2, c(1, 2), dtype="double")
  
  expect_error(a + b, "different dtypes")
  expect_error(a * b, "different dtypes")
})

test_that("complex broadcasting patterns work", {
  # (1,3,1) + (2,1,4) -> (2,3,4)
  a <- gpu_tensor(c(1, 2, 3), c(1, 3, 1))
  b <- gpu_tensor(1:8, c(2, 1, 4))
  
  result <- a + b
  expect_equal(dim(result), c(2, 3, 4))
  
  # Check a few strategic elements
  result_array <- as.array(result)
  expect_equal(result_array[1,1,1], 1 + 1)  # a[1,1,1] + b[1,1,1]
  expect_equal(result_array[1,2,1], 2 + 1)  # a[1,2,1] + b[1,1,1]  
  expect_equal(result_array[2,1,2], 1 + 4)  # a[1,1,1] + b[2,1,2] (b[2,1,2] = 4, not 6)
}) 