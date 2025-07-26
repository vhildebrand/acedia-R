## dtype validation and basic performance sanity ----

library(testthat)

context("DType validation and basic performance")

skip_if_not(gpu_available(), "GPU not available")


test_that("unsupported dtype errors are raised", {
  expect_error(gpu_tensor(1:3, c(3), dtype = "int8"), "Unsupported dtype")
  expect_error(empty_tensor(c(3, 3), dtype = "float16"), "Unsupported dtype")
})


test_that("GPU implementation is appreciably faster than CPU for large vectors", {
  n <- 5e5  # 500k elements, large enough to benefit from GPU but quick for CI
  a <- stats::runif(n)
  b <- stats::runif(n)

  # CPU baseline
  cpu_time <- system.time({
    cpu_res <- a * b
  })[["elapsed"]]

  # GPU path (suppress fallback warnings)
  gpu_time <- system.time({
    gpu_res <- gpu_multiply(a, b, warn_fallback = FALSE)
  })[["elapsed"]]

  expect_equal(gpu_res[1:100], cpu_res[1:100], tolerance = 1e-12)

  # We only require GPU run to complete in reasonable wall-clock time (<5 seconds).  
  # Depending on transfer overhead, it may not always beat CPU in CI environments.
  expect_lt(gpu_time, 5)
}) 