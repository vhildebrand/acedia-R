# Test script for new outer product and matrix-vector operations
library(acediaR)

cat("Testing new linear algebra operations...\n")

# Test outer product
cat("\n=== OUTER PRODUCT TESTS ===\n")
tryCatch({
  
  # Create test vectors
  a_data <- c(1, 2, 3)
  b_data <- c(4, 5)
  
  # Create tensors
  a_tensor <- as_tensor(a_data, dtype = "float32")
  b_tensor <- as_tensor(b_data, dtype = "float32")
  
  cat("Vector a:", a_data, "\n")
  cat("Vector b:", b_data, "\n")
  
  # Compute outer product on GPU
  result_gpu <- outer_product(a_tensor, b_tensor)
  result_cpu <- outer(a_data, b_data)
  
  cat("GPU result shape:", shape(result_gpu), "\n")
  cat("GPU result:\n")
  print(as.array(result_gpu))
  
  cat("CPU result:\n")
  print(result_cpu)
  
  # Check correctness
  match_result <- all.equal(as.array(result_gpu), result_cpu, tolerance = 1e-6)
  if (isTRUE(match_result)) {
    cat("✅ Outer product test PASSED\n")
  } else {
    cat("❌ Outer product test FAILED:", match_result, "\n")
  }
  
}, error = function(e) {
  cat("❌ Outer product test failed:", e$message, "\n")
})

# Test matrix-vector multiplication
cat("\n=== MATRIX-VECTOR TESTS ===\n")
tryCatch({
  
  # Create test matrix and vector
  A_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)  # 2x3 matrix
  v_data <- c(7, 8, 9)  # length 3 vector
  
  # Create tensors
  A_tensor <- as_tensor(A_data, dtype = "float32")
  v_tensor <- as_tensor(v_data, dtype = "float32")
  
  cat("Matrix A (2x3):\n")
  print(A_data)
  cat("Vector v:", v_data, "\n")
  
  # Compute matrix-vector multiplication on GPU
  result_gpu <- matvec(A_tensor, v_tensor)
  result_cpu <- A_data %*% v_data
  
  cat("GPU result:", as.array(result_gpu), "\n")
  cat("CPU result:", as.vector(result_cpu), "\n")
  
  # Check correctness
  match_result <- all.equal(as.array(result_gpu), as.vector(result_cpu), tolerance = 1e-6)
  if (isTRUE(match_result)) {
    cat("✅ Matrix-vector test PASSED\n")
  } else {
    cat("❌ Matrix-vector test FAILED:", match_result, "\n")
  }
  
}, error = function(e) {
  cat("❌ Matrix-vector test failed:", e$message, "\n")
})

# Test vector-matrix multiplication
cat("\n=== VECTOR-MATRIX TESTS ===\n")
tryCatch({
  
  # Create test vector and matrix
  v_data <- c(10, 11)  # length 2 vector
  A_data <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)  # 2x3 matrix
  
  # Create tensors
  v_tensor <- as_tensor(v_data, dtype = "float32")
  A_tensor <- as_tensor(A_data, dtype = "float32")
  
  cat("Vector v:", v_data, "\n")
  cat("Matrix A (2x3):\n")
  print(A_data)
  
  # Compute vector-matrix multiplication on GPU
  result_gpu <- vecmat(v_tensor, A_tensor)
  result_cpu <- v_data %*% A_data
  
  cat("GPU result:", as.array(result_gpu), "\n")
  cat("CPU result:", as.vector(result_cpu), "\n")
  
  # Check correctness
  match_result <- all.equal(as.array(result_gpu), as.vector(result_cpu), tolerance = 1e-6)
  if (isTRUE(match_result)) {
    cat("✅ Vector-matrix test PASSED\n")
  } else {
    cat("❌ Vector-matrix test FAILED:", match_result, "\n")
  }
  
}, error = function(e) {
  cat("❌ Vector-matrix test failed:", e$message, "\n")
})

cat("\n=== PERFORMANCE COMPARISON ===\n")
tryCatch({
  
  # Large outer product performance test
  n1 <- 1000
  n2 <- 1000
  
  a_large <- runif(n1)
  b_large <- runif(n2)
  
  a_tensor_large <- as_tensor(a_large, dtype = "float32")
  b_tensor_large <- as_tensor(b_large, dtype = "float32")
  
  cat("Testing", n1, "x", n2, "outer product...\n")
  
  # Time GPU version
  gpu_time <- system.time({
    result_gpu_large <- outer_product(a_tensor_large, b_tensor_large)
  })
  
  # Time CPU version  
  cpu_time <- system.time({
    result_cpu_large <- outer(a_large, b_large)
  })
  
  cat("GPU time:", gpu_time["elapsed"], "seconds\n")
  cat("CPU time:", cpu_time["elapsed"], "seconds\n")
  cat("Speedup:", cpu_time["elapsed"] / gpu_time["elapsed"], "x\n")
  
  # Verify correctness for a subset
  subset_indices <- 1:min(100, n1)
  subset_j <- 1:min(100, n2)
  
  match_result <- all.equal(as.array(result_gpu_large)[subset_indices, subset_j], 
                result_cpu_large[subset_indices, subset_j], tolerance = 1e-6)
  if (isTRUE(match_result)) {
    cat("✅ Large outer product correctness PASSED\n")
  } else {
    cat("❌ Large outer product correctness FAILED:", match_result, "\n")
  }
  
}, error = function(e) {
  cat("❌ Performance test failed:", e$message, "\n")
})

cat("\nAll tests completed!\n") 