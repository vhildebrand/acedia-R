# Detailed debug for matrix-vector issue
library(acediaR)

cat("=== DETAILED MATRIX-VECTOR DEBUG ===\n")

# Test with even simpler case
A_data <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2)  # Identity matrix
v_data <- c(10, 20)

cat("Identity matrix test:\n")
cat("Matrix A (identity):\n")
print(A_data)
cat("Vector v:", v_data, "\n")

# CPU version
cpu_result <- A_data %*% v_data
cat("CPU result:", as.vector(cpu_result), "\n")

# GPU version  
A_tensor <- as_tensor(A_data, dtype = "float32")
v_tensor <- as_tensor(v_data, dtype = "float32")
gpu_result <- matvec(A_tensor, v_tensor)
cat("GPU result:", as.array(gpu_result), "\n")

cat("Expected for identity: [10, 20]\n")

# Test with different sizes to see if it's a boundary issue
cat("\n--- Testing different matrix sizes ---\n")

for (size in c(2, 3, 4)) {
  cat(sprintf("Testing %dx%d matrix:\n", size, size))
  
  # Create simple test matrix and vector
  A_test <- diag(1:size)  # diagonal matrix with 1,2,3,... on diagonal
  v_test <- rep(1, size)  # vector of ones
  
  cat("Matrix A:\n")
  print(A_test)
  cat("Vector v:", v_test, "\n")
  
  # CPU
  cpu_res <- A_test %*% v_test
  cat("CPU result:", as.vector(cpu_res), "\n")
  
  # GPU
  A_gpu <- as_tensor(A_test, dtype = "float32")
  v_gpu <- as_tensor(v_test, dtype = "float32")
  gpu_res <- matvec(A_gpu, v_gpu)
  cat("GPU result:", as.array(gpu_res), "\n")
  
  # Check if they match
  if (all.equal(as.array(gpu_res), as.vector(cpu_res), tolerance = 1e-6)) {
    cat("✅ MATCH\n")
  } else {
    cat("❌ MISMATCH\n")
  }
  cat("\n")
} 