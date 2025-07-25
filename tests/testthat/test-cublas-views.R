context("cuBLAS with contiguous and view tensors")

library(testthat)
library(acediaR)

set.seed(123)

test_that("matmul handles contiguous and transpose views correctly", {
  A <- matrix(rnorm(64), nrow = 8, ncol = 8)
  B <- matrix(rnorm(48), nrow = 8, ncol = 6)

  gA <- gpu_tensor(A, shape = dim(A))
  gB <- gpu_tensor(B, shape = dim(B))

  # contiguous multiplication
  C_gpu <- matmul(gA, gB)
  expect_tensor_equal(C_gpu, A %*% B)

  # transpose views without making them contiguous
  gA_t <- gA$transpose()         # 8×8 -> 8×8 but strides swapped
  gB_t <- gB$transpose()         # 8×6 -> 6×8

  # Now dimensions: (8×8)^T is still 8×8, so use different dims for validity
  D <- matrix(rnorm(40), nrow = 5, ncol = 8)
  gD <- gpu_tensor(D, shape = dim(D))
  gD_t <- gD$transpose()         # 8×5 view

  C2_gpu <- matmul(gA_t, gD_t)    # uses views on both operands
  expect_tensor_equal(C2_gpu, t(A) %*% t(D))
})


test_that("matvec with transpose view produces correct result", {
  A <- matrix(rnorm(30), nrow = 6, ncol = 5)
  v <- rnorm(5)

  gA <- gpu_tensor(A, shape = dim(A))
  gv <- gpu_tensor(v, shape = length(v))

  res_gpu <- matvec(gA, gv)  # contiguous
  expect_tensor_equal(res_gpu, A %*% v)

  # Use transpose view of A and vector treated as (1×N)^T
  gA_t <- gA$transpose()  # 5×6 view
  gv_t <- gv$view(c(1, length(v)))$transpose()  # turn into column vector view

  res_gpu2 <- vecmat(gv, gA_t)
  expect_tensor_equal(res_gpu2, t(v) %*% t(A))
}) 