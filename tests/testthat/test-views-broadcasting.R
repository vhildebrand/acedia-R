context("Views, slice mutation, transpose/permute & broadcasting")

library(testthat)

# ----------------------------------------------------------------------------
# GPU slice mutation (in-place) ------------------------------------------------
# ----------------------------------------------------------------------------

test_that("Slice mutation updates parent tensor in-place", {
  skip_if_not(gpu_available(), "GPU not available")

  parent <- gpu_tensor(1:120, shape = c(4, 5, 6), dtype = "float")
  pre_cpu <- as.array(parent)

  scalar_inc <- 7.5
  # In-place add to first two layers (dimension 1)
  parent[1:2, , ] <- scalar_inc
  synchronize(parent)

  post_cpu <- as.array(parent)

  expect_equal(post_cpu[1:2, , ], pre_cpu[1:2, , ] + scalar_inc, tolerance = 1e-6)
  expect_equal(post_cpu[3:4, , ], pre_cpu[3:4, , ],            tolerance = 1e-6)
})

# ----------------------------------------------------------------------------
# Non-contiguous views: transpose --------------------------------------------
# ----------------------------------------------------------------------------

test_that("Operations on transpose view produce correct results", {
  skip_if_not(gpu_available(), "GPU not available")

  set.seed(123)
  mat_data <- matrix(runif(20, -1, 1), nrow = 4, ncol = 5)
  mat_tensor <- as_tensor(mat_data, dtype = "float")

  trans_view <- transpose(mat_tensor)
  result_gpu <- trans_view * 3.0 + 1.0
  result_cpu <- (t(mat_data) * 3.0) + 1.0

  expect_equal(as.array(result_gpu), result_cpu, tolerance = 1e-6)
})

# ----------------------------------------------------------------------------
# Non-contiguous views: permute ----------------------------------------------
# ----------------------------------------------------------------------------

test_that("Permute view arithmetic matches CPU", {
  skip_if_not(gpu_available(), "GPU not available")

  tensor_orig <- gpu_tensor(1:24, shape = c(2, 3, 4), dtype = "float")
  perm_view   <- permute(tensor_orig, c(3, 2, 1))  # new shape 4×3×2

  res_gpu <- perm_view + 2.0

  orig_arr <- array(1:24, dim = c(2, 3, 4))
  perm_arr <- aperm(orig_arr, c(3, 2, 1)) + 2.0

  expect_equal(as.array(res_gpu), perm_arr, tolerance = 1e-6)
})

# ----------------------------------------------------------------------------
# Broadcasting: high-rank success & deliberate failure ------------------------
# ----------------------------------------------------------------------------

test_that("High-rank broadcasting works and mismatch errors are thrown", {
  skip_if_not(gpu_available(), "GPU not available")

  set.seed(456)
  a_data <- runif(2 * 3 * 4)
  b_data <- runif(1 * 3 * 1)

  a_tensor <- gpu_tensor(a_data, shape = c(2, 3, 4), dtype = "float")
  b_tensor <- gpu_tensor(b_data, shape = c(1, 3, 1), dtype = "float")

  broadcast_gpu <- a_tensor + b_tensor

  a_arr <- array(a_data, dim = c(2, 3, 4))
  b_arr <- array(b_data, dim = c(1, 3, 1))
  # Manually broadcast b_arr to 2×3×4
  b_rep <- b_arr[rep(1, 2), , rep(1, 4)]
  expected_cpu <- a_arr + b_rep

  expect_equal(as.array(broadcast_gpu), expected_cpu, tolerance = 1e-6)

  # Incompatible shapes should throw an informative error
  bad_a <- gpu_tensor(runif(2 * 3), shape = c(2, 3), dtype = "float")
  bad_b <- gpu_tensor(runif(2 * 2), shape = c(2, 2), dtype = "float")

  expect_error(bad_a + bad_b, regexp = "broadcast|shape|match", ignore.case = TRUE)
}) 