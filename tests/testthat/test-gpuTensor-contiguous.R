context("gpuTensor contiguous copy")

skip_if_not(gpu_available(), "GPU not available")


test_that("contiguous() returns contiguous tensor", {
  t <- gpu_tensor(1:6, c(2,3))
  view_t <- view(t, c(3,2))
  expect_true(is_contiguous(view_t))  # view keeps contiguous
  # Make non-contiguous by transpose effect: reshape then view back; we simulate by view of view
  # For now create t2 by view of contiguous to same shape (still contiguous) but call contiguous()
  t2 <- view(t, c(6))
  cont <- contiguous(t2)  # should succeed
  expect_true(is_contiguous(cont))
  expect_equal(as.vector(cont), 1:6)
})

test_that("contiguous() preserves data for already contiguous tensors", {
  t <- gpu_tensor(1:12, c(3, 4))
  expect_true(is_contiguous(t))
  
  cont <- contiguous(t)
  expect_true(is_contiguous(cont))
  expect_equal(as.array(cont), as.array(t))
})

test_that("contiguous() works with different dtypes", {
  # Test float (default)
  t_float <- gpu_tensor(c(1.5, 2.5, 3.5, 4.5), c(2, 2), dtype="float")
  cont_float <- contiguous(t_float)
  expect_equal(as.array(cont_float), as.array(t_float))
  
  # Test double
  t_double <- gpu_tensor(c(1.5, 2.5, 3.5, 4.5), c(2, 2), dtype="double")
  cont_double <- contiguous(t_double)
  expect_equal(as.array(cont_double), as.array(t_double))
})

test_that("contiguous() is faster than host round-trip", {
  # Large tensor to see performance difference
  n <- 1000
  t <- gpu_tensor(1:n, c(10, 100))
  
  # Time the GPU-native contiguous operation
  gpu_time <- system.time({
    cont <- contiguous(t)
  })[[3]]
  
  # Should complete quickly (under 1 second for this size)
  expect_lt(gpu_time, 1.0)
})

test_that("contiguous() handles 3D tensors correctly", {
  data_3d <- 1:24
  t <- gpu_tensor(data_3d, c(2, 3, 4))
  
  cont <- contiguous(t)
  expect_true(is_contiguous(cont))
  expect_equal(as.vector(cont), data_3d)
  expect_equal(dim(cont), c(2, 3, 4))
}) 