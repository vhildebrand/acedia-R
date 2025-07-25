testthat::local_edition(3)

# Helper: convert gpuTensor to host array if needed
as_host <- function(x) if (inherits(x, "gpuTensor")) as.array(x) else x

# Custom expectation that works with gpuTensor/numeric interchangeably
expect_tensor_equal <- function(object, expected, ..., tolerance = 1e-6) {
  testthat::expect_equal(as_host(object), as_host(expected), tolerance = tolerance, ...)
} 