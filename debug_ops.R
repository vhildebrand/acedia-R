# Debug script for new linear algebra operations
library(acediaR)

cat("=== DEBUGGING NEW OPERATIONS ===\n")

# Simple outer product test
cat("\n--- Outer Product Debug ---\n")
a_data <- c(1, 2)
b_data <- c(3, 4)

cat("Vector a:", a_data, "\n")
cat("Vector b:", b_data, "\n")

# CPU version
cpu_result <- outer(a_data, b_data)
cat("CPU outer product:\n")
print(cpu_result)
cat("CPU result shape:", dim(cpu_result), "\n")

# GPU version
a_tensor <- as_tensor(a_data, dtype = "float32")
b_tensor <- as_tensor(b_data, dtype = "float32")
gpu_result_tensor <- outer_product(a_tensor, b_tensor)
gpu_result <- as.array(gpu_result_tensor)

cat("GPU outer product:\n")
print(gpu_result)
cat("GPU result shape:", shape(gpu_result_tensor), "\n")
cat("Expected: a[i] * b[j] where i=row, j=col\n")
cat("a[1]*b[1]=", a_data[1]*b_data[1], ", a[1]*b[2]=", a_data[1]*b_data[2], "\n")
cat("a[2]*b[1]=", a_data[2]*b_data[1], ", a[2]*b[2]=", a_data[2]*b_data[2], "\n")

# Simple matrix-vector test
cat("\n--- Matrix-Vector Debug ---\n")
A_data <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)  # 2x2 matrix
v_data <- c(5, 6)  # length 2 vector

cat("Matrix A (2x2):\n")
print(A_data)
cat("Vector v:", v_data, "\n")

# CPU version
cpu_matvec <- A_data %*% v_data
cat("CPU matrix-vector result:", as.vector(cpu_matvec), "\n")

# GPU version
A_tensor <- as_tensor(A_data, dtype = "float32")
v_tensor <- as_tensor(v_data, dtype = "float32")
gpu_matvec_tensor <- matvec(A_tensor, v_tensor)
gpu_matvec <- as.array(gpu_matvec_tensor)

cat("GPU matrix-vector result:", gpu_matvec, "\n")
cat("Expected: [A[1,1]*v[1] + A[1,2]*v[2], A[2,1]*v[1] + A[2,2]*v[2]]\n")
cat("Manual calc: [", A_data[1,1]*v_data[1] + A_data[1,2]*v_data[2], ",", 
    A_data[2,1]*v_data[1] + A_data[2,2]*v_data[2], "]\n")

cat("\n--- Tensor Information ---\n")
cat("A_tensor shape:", shape(A_tensor), "\n")
cat("v_tensor shape:", shape(v_tensor), "\n")
cat("Result tensor shape:", shape(gpu_matvec_tensor), "\n")

cat("\n--- Raw Matrix Data Layout Check ---\n")
cat("A_data as vector (R column-major):", as.vector(A_data), "\n")
cat("A_tensor as array:", as.array(A_tensor), "\n") 