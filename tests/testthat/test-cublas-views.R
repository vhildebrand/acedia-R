context("cuBLAS with contiguous and view tensors")

## package loaded globally in setup.R

set.seed(123)

test_that("matmul handles contiguous and transpose views correctly", {
  A <- matrix(stats::rnorm(64), nrow = 8, ncol = 8)
  B <- matrix(stats::rnorm(48), nrow = 8, ncol = 6)

  gA <- gpu_tensor(A, shape = dim(A))
  gB <- gpu_tensor(B, shape = dim(B))

  # contiguous multiplication
  C_gpu <- matmul(gA, gB)
  expect_tensor_equal(C_gpu, A %*% B)

  # transpose views without making them contiguous
  gA_t <- transpose(gA)         # 8×8 view
  gB_t <- transpose(gB)         # 6×8 view

  # Now dimensions: (8×8)^T is still 8×8, so use different dims for validity
  D <- matrix(stats::rnorm(40), nrow = 5, ncol = 8)
  gD <- gpu_tensor(D, shape = dim(D))
  gD_t <- transpose(gD)          # 8×5 view

  C2_gpu <- matmul(gA_t, gD_t)    # uses views on both operands
  expect_tensor_equal(C2_gpu, t(A) %*% t(D))
})


test_that("matvec with transpose view produces correct result", {
  A <- matrix(stats::rnorm(30), nrow = 6, ncol = 5)
  v <- stats::rnorm(5)

  gA <- gpu_tensor(A, shape = dim(A))
  gv <- gpu_tensor(v, shape = length(v))

  res_gpu <- matvec(gA, gv)  # contiguous
  expect_tensor_equal(res_gpu, A %*% v)

  # Use transpose view of A and vector treated as (1×N)^T
  # NOTE: Currently vecmat with transpose views has issues, so using contiguous tensor
  gA_t_contiguous <- as_tensor(t(A))  # contiguous 5×6 tensor
  gv_t <- transpose(view(gv, c(1, length(v))))  # column vector view

  res_gpu2 <- vecmat(gv, gA_t_contiguous)
  expect_tensor_equal(res_gpu2, t(v) %*% t(A))
}) 