context("gpuTensor subtraction and division")

skip_if_not(gpu_available(), "GPU not available")


test_that("tensor - tensor and tensor - scalar work", {
  a <- gpu_tensor(1:6, c(2,3))
  b <- gpu_tensor(rep(1,6), c(2,3))
  # tensor - tensor
  res <- a - b
  expect_equal(as.array(res), matrix(0:5, 2,3), tolerance=1e-10)
  # tensor - scalar
  res2 <- a - 1
  expect_equal(as.array(res2), matrix(0:5, 2,3), tolerance=1e-10)
})


test_that("tensor / tensor and tensor / scalar work", {
  x <- gpu_tensor(c(2,4,6,8), c(2,2))
  y <- gpu_tensor(rep(2,4), c(2,2))
  res <- x / y
  expect_equal(as.array(res), matrix(c(1,2,3,4),2,2), tolerance=1e-10)
  res2 <- x / 2
  expect_equal(as.array(res2), matrix(c(1,2,3,4),2,2), tolerance=1e-10)
}) 