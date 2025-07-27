#!/usr/bin/env Rscript
# Linear Algebra Demonstration Script for acediaR
# Shows base R functionality and framework for GPU comparison

library(acediaR)

cat("=== acediaR Linear Algebra Functions Demo ===\n\n")

# Test matrices
set.seed(42)
cat("Creating test matrices...\n")

# Test matrix 1: Small symmetric positive definite
A1 <- matrix(c(4, 1, 2, 1, 3, 1, 2, 1, 5), 3, 3)
cat("Test Matrix A1 (3x3 symmetric positive definite):\n")
print(A1)

# Test matrix 2: Rectangular for QR
A2 <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 4, 3)
cat("\nTest Matrix A2 (4x3 rectangular):\n")
print(A2)

# Test vector for solve
b <- c(1, 2, 3)
cat("\nTest vector b:\n")
print(b)

cat("\n", paste(rep("=", 60), collapse=""), "\n")

# Function availability check
cat("\n=== Function Availability Check ===\n")
linear_funcs <- c("det", "solve", "qr", "chol", "eigen", "lu_decompose")
for (func in linear_funcs) {
  if (exists(func)) {
    cat("✅", func, "- Available\n")
  } else {
    cat("❌", func, "- Not found\n")
  }
}

# S3 method check
cat("\n=== S3 Method Registration Check ===\n")
s3_methods <- c("determinant.gpuTensor", "solve.gpuTensor", "qr.gpuTensor", 
                "chol.gpuTensor")
for (method in s3_methods) {
  if (exists(method, envir = asNamespace('acediaR'))) {
    cat("✅", method, "- Registered\n")
  } else {
    cat("❌", method, "- Not found\n")
  }
}

# Check gpu_eigen separately since it's not an S3 method
if (exists("gpu_eigen")) {
  cat("✅ gpu_eigen - Available\n")
} else {
  cat("❌ gpu_eigen - Not found\n")
}

# Check if methods are in the method tables
cat("\n=== S3 Method Table Check ===\n")
base_funcs <- c("determinant", "solve", "qr", "chol")
for (func in base_funcs) {
  methods_list <- methods(func)
  has_gpu_method <- any(grepl("gpuTensor", methods_list))
  cat(func, "has gpuTensor method:", has_gpu_method, "\n")
}
cat("gpu_eigen is a direct function (not S3 method)\n")

cat("\n", paste(rep("=", 60), collapse=""), "\n")

# Base R demonstrations
cat("\n=== Base R Linear Algebra Demonstrations ===\n")

cat("\n--- Matrix A1 (3x3 symmetric) ---\n")
cat("Determinant:", det(A1), "\n")

# LU-based solve
x1 <- solve(A1, b)
cat("Solve A1 * x = b:", paste(round(x1, 4), collapse=", "), "\n")
cat("Verification A1 %*% x:", paste(round(A1 %*% x1, 4), collapse=", "), "\n")

# QR decomposition
qr1 <- qr(A1)
Q1 <- qr.Q(qr1)
R1 <- qr.R(qr1)
cat("QR decomposition completed\n")
cat("Q matrix:\n")
print(round(Q1, 4))
cat("R matrix:\n") 
print(round(R1, 4))
cat("Q %*% R reconstruction error:", max(abs(Q1 %*% R1 - A1)), "\n")

# Cholesky decomposition
chol1 <- chol(A1)
cat("Cholesky factor (upper triangular):\n")
print(round(chol1, 4))
cat("Cholesky reconstruction error:", max(abs(t(chol1) %*% chol1 - A1)), "\n")

# Eigenvalue decomposition
eigen1 <- eigen(A1, symmetric = TRUE)
cat("Eigenvalues:", paste(round(eigen1$values, 4), collapse=", "), "\n")
cat("First eigenvector:", paste(round(eigen1$vectors[,1], 4), collapse=", "), "\n")
# Verify Av = λv
Av <- A1 %*% eigen1$vectors[,1]
lambdav <- eigen1$values[1] * eigen1$vectors[,1]
cat("Eigenvalue equation error:", max(abs(Av - lambdav)), "\n")

cat("\n--- Matrix A2 (4x3 rectangular) ---\n")
# QR for rectangular matrix
qr2 <- qr(A2)
Q2 <- qr.Q(qr2)
R2 <- qr.R(qr2)
cat("QR decomposition of rectangular matrix:\n")
cat("Q dimensions:", dim(Q2), "\n")
cat("R dimensions:", dim(R2), "\n")
cat("Q %*% R reconstruction error:", max(abs(Q2 %*% R2 - A2)), "\n")

 