testthat::local_edition(3)

# Helper: convert gpuTensor to host array if needed
as_host <- function(x) if (inherits(x, "gpuTensor")) as.array(x) else x

# Helper functions for tensor testing

# Tensor equality assertion with tolerance
expect_tensor_equal <- function(tensor, expected, tolerance = 1e-6) {
  expect_equal(as.array(tensor), expected, tolerance = tolerance)
}

# GPU tensor verification
verify_gpu_tensor <- function(tensor, operation_name = "operation") {
  if (!inherits(tensor, "gpuTensor")) {
    warning(paste("❌ GPU FALLBACK:", operation_name, "returned non-gpuTensor object"))
    return(FALSE)
  }
  return(TRUE)
}

# Skip tests if GPU not available
skip_on_ci_if_no_gpu <- function() {
  tryCatch({
    test_tensor <- as_tensor(c(1, 2, 3), dtype = "float")
    if (!inherits(test_tensor, "gpuTensor")) {
      skip("GPU not available")
    }
  }, error = function(e) {
    skip("GPU not available")
  })
}

# Verify all tensors in a list are GPU tensors
verify_all_gpu <- function(tensor_list, operation_name = "operation") {
  for (i in seq_along(tensor_list)) {
    if (!verify_gpu_tensor(tensor_list[[i]], paste(operation_name, "result", i))) {
      return(FALSE)
    }
  }
  return(TRUE)
} 