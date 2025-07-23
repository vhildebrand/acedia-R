context("New GPU operations: softmax, argmax, concat, stack, repeat, pad, reductions, comparisons")

library(testthat)

# Helper (reuse if defined elsewhere)
verify_gpu_tensor <- function(tensor, operation_name = "operation") {
  if (!inherits(tensor, "gpuTensor")) {
    warning(paste("❌ GPU FALLBACK:", operation_name, "returned non-gpuTensor object"))
    return(FALSE)
  }
  TRUE
}

skip_if_not(gpu_available(), "GPU not available")

# ------------------------------------------------------------------
# Softmax & Argmax --------------------------------------------------
# ------------------------------------------------------------------

test_that("Softmax and Argmax work correctly on GPU", {
  set.seed(123)
  x <- runif(10, -3, 3)
  tensor_x <- as_tensor(x, dtype = "float")
  verify_gpu_tensor(tensor_x, "softmax input")
  # Softmax
  soft_gpu <- softmax(tensor_x)
  verify_gpu_tensor(soft_gpu, "softmax result")
  soft_cpu <- exp(x) / sum(exp(x))
  expect_equal(as.vector(soft_cpu), as.array(soft_gpu), tolerance = 1e-6)
  expect_equal(sum(as.array(soft_gpu)), 1, tolerance = 1e-6)
  # Argmax
  am_gpu <- argmax(tensor_x)
  am_cpu <- which.max(x)
  expect_equal(am_gpu, am_cpu)
})

# ------------------------------------------------------------------
# Concatenation & Stack --------------------------------------------
# ------------------------------------------------------------------

test_that("Concat and Stack produce correct results", {
  library(abind)
  a <- gpu_tensor(1:6, c(2,3))
  b <- gpu_tensor(7:12, c(2,3))
  verify_gpu_tensor(a, "concat a"); verify_gpu_tensor(b, "concat b")
  # Concat along 1st dim
  c_gpu <- concat(list(a,b), axis = 1)
  verify_gpu_tensor(c_gpu, "concat result")
  c_cpu <- abind::abind(as.array(a), as.array(b), along = 1)
  dimnames(c_cpu) <- NULL  # Remove dimnames to match GPU result
  expect_equal(as.array(c_gpu), c_cpu)
  # Stack along new dim 3
  s_gpu <- stack(list(a,b), axis = 3)
  verify_gpu_tensor(s_gpu, "stack result")
  s_cpu <- abind::abind(as.array(a), as.array(b), along = 3)
  dimnames(s_cpu) <- NULL  # Remove dimnames to match GPU result
  expect_equal(as.array(s_gpu), s_cpu)
})

# ------------------------------------------------------------------
# Repeat & Pad ------------------------------------------------------
# ------------------------------------------------------------------

test_that("repeat_tensor and pad work", {
  t <- gpu_tensor(1:4, c(2,2))
  r_gpu <- repeat_tensor(t, c(2,1))
  verify_gpu_tensor(r_gpu, "repeat")
  r_cpu <- as.array(t)[rep(1:2, each=2), ]
  expect_equal(as.array(r_gpu), r_cpu)
  # Constant pad of 1 around
  pad_gpu <- pad(t, matrix(c(1,1,1,1), nrow=2, byrow=TRUE), mode="constant", value=0)
  verify_gpu_tensor(pad_gpu, "pad")
  pad_cpu <- matrix(0, 4,4); pad_cpu[2:3,2:3] <- as.array(t)
  expect_equal(as.array(pad_gpu), pad_cpu)
})

# ------------------------------------------------------------------
# Reductions prod & var --------------------------------------------
# ------------------------------------------------------------------

test_that("prod and var reductions correct", {
  set.seed(42)
  x <- runif(20, 0.5,1.5)
  tx <- as_tensor(x, dtype="float")
  expect_equal(prod(tx), prod(x), tolerance=1e-6)
  expect_equal(var(tx), var(x), tolerance=1e-6)
})

# ------------------------------------------------------------------
# Comparison operators ---------------------------------------------
# ------------------------------------------------------------------

test_that("Comparison operators match CPU", {
  a <- gpu_tensor(c(1,3,2,4), c(4))
  b <- gpu_tensor(c(2,2,2,2), c(4))
  gt_gpu <- as.array(a > b)
  lt_gpu <- as.array(a < b)
  eq_gpu <- as.array(a == b)
  expect_equal(gt_gpu, as.numeric(c(1,1,0,1) > 0))
  expect_equal(lt_gpu, as.numeric(c(1,3,2,4) < c(2,2,2,2)))
  expect_equal(eq_gpu, as.numeric(c(1,3,2,4) == c(2,2,2,2)))
})

# ------------------------------------------------------------------
# Simple performance sanity (softmax) ------------------------------
# ------------------------------------------------------------------

test_that("Softmax GPU runtime reasonable", {
  n <- 1e5
  x <- runif(n)
  gx <- as_tensor(x, dtype="float")
  cpu_time <- system.time({res_cpu <- exp(x)/sum(exp(x))})[["elapsed"]]
  gpu_time <- system.time({res_gpu <- softmax(gx); synchronize(res_gpu)})[["elapsed"]]
  expect_equal(as.vector(res_gpu)[1:10], res_cpu[1:10], tolerance=1e-4)
  expect_lt(gpu_time, cpu_time * 5)  # allow overhead but not extreme
}) 