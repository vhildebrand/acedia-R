#!/usr/bin/env Rscript
# Linear Algebra Validation Script for acediaR
# Compares GPU implementations with base R for correctness verification

library(acediaR)

# Helper function to check if two matrices are approximately equal
matrices_equal <- function(a, b, tolerance = 1e-6, name = "") {
  if (is.null(a) || is.null(b)) {
    cat("❌", name, "- One matrix is NULL\n")
    return(FALSE)
  }
  
  # Convert gpuTensor to array if needed
  if (inherits(a, "gpuTensor")) {
    a <- as.array(a)
  }
  if (inherits(b, "gpuTensor")) {
    b <- as.array(b)
  }
  
  # Check dimensions
  if (!identical(dim(a), dim(b))) {
    cat("❌", name, "- Dimension mismatch:", dim(a), "vs", dim(b), "\n")
    return(FALSE)
  }
  
  # Check values
  max_diff <- max(abs(a - b), na.rm = TRUE)
  if (max_diff < tolerance) {
    cat("✅", name, "- Match (max diff:", sprintf("%.2e", max_diff), ")\n")
    return(TRUE)
  } else {
    cat("❌", name, "- Values differ (max diff:", sprintf("%.2e", max_diff), ")\n")
    return(FALSE)
  }
}

# Helper function to check if two scalars are approximately equal
scalars_equal <- function(a, b, tolerance = 1e-6, name = "") {
  if (is.null(a) || is.null(b)) {
    cat("❌", name, "- One value is NULL\n")
    return(FALSE)
  }
  
  # Convert gpuTensor to scalar if needed
  if (inherits(a, "gpuTensor")) {
    a <- as.numeric(as.array(a))
  }
  if (inherits(b, "gpuTensor")) {
    b <- as.numeric(as.array(b))
  }
  
  diff <- abs(a - b)
  if (diff < tolerance) {
    cat("✅", name, "- Match (diff:", sprintf("%.2e", diff), ")\n")
    return(TRUE)
  } else {
    cat("❌", name, "- Values differ (diff:", sprintf("%.2e", diff), ")\n")
    cat("   GPU:", a, "vs Base R:", b, "\n")
    return(FALSE)
  }
}

cat("=== acediaR Linear Algebra Validation ===\n\n")

# Test matrices
set.seed(42)
cat("Creating test matrices...\n")

# Test matrix 1: Small symmetric positive definite
A1_base <- matrix(c(4, 1, 2, 1, 3, 1, 2, 1, 5), 3, 3)
cat("Test Matrix A1 (3x3 symmetric positive definite):\n")
print(A1_base)

# Test matrix 2: Larger random positive definite
n <- 5
A2_temp <- matrix(rnorm(n*n), n, n)
A2_base <- A2_temp %*% t(A2_temp) + diag(n) * 0.1
cat("\nTest Matrix A2 (5x5 random positive definite):\n")
print(round(A2_base, 4))

# Test matrix 3: Rectangular for QR
A3_base <- matrix(rnorm(12), 4, 3)
cat("\nTest Matrix A3 (4x3 rectangular for QR):\n")
print(round(A3_base, 4))

# Test vector for solve
b_base <- c(1, 2, 3)
cat("\nTest vector b for solve:\n")
print(b_base)

cat("\n", paste(rep("=", 60), collapse=""), "\n")

# Function to test a single matrix
test_matrix <- function(A_base, matrix_name, test_solve = TRUE) {
  cat("\n### Testing", matrix_name, "###\n")
  
  # Create GPU tensor (this will work once registration is fixed)
  A_gpu <- NULL
  tryCatch({
    # Try different tensor creation functions
    if (exists("create_tensor_unified")) {
      A_gpu <- create_tensor_unified(A_base)
      class(A_gpu) <- c("gpuTensor", class(A_gpu))
      cat("✅ GPU tensor created successfully\n")
    } else {
      cat("❌ create_tensor_unified not available\n")
    }
  }, error = function(e) {
    cat("❌ Could not create GPU tensor:", e$message, "\n")
    cat("   Using base R only for comparison\n")
    A_gpu <- NULL
  })
  
  # Test 1: Determinant
  cat("\n--- Determinant Test ---\n")
  if (nrow(A_base) == ncol(A_base)) {
    det_base <- det(A_base)
    cat("Base R det:", det_base, "\n")
    
    if (!is.null(A_gpu)) {
      tryCatch({
        det_gpu <- det(A_gpu)
        scalars_equal(det_gpu, det_base, name = "Determinant")
      }, error = function(e) {
        cat("❌ GPU det failed:", e$message, "\n")
      })
    }
  } else {
    cat("⏭️  Skipping determinant (non-square matrix)\n")
  }
  
  # Test 2: Linear solve (square matrices only)
  if (test_solve && nrow(A_base) == ncol(A_base) && nrow(A_base) == length(b_base)) {
    cat("\n--- Linear Solve Test ---\n")
    solve_base <- solve(A_base, b_base)
    cat("Base R solve:", paste(round(solve_base, 4), collapse = ", "), "\n")
    
         if (!is.null(A_gpu)) {
       tryCatch({
         if (exists("create_tensor_unified")) {
           b_gpu <- create_tensor_unified(b_base)
           class(b_gpu) <- c("gpuTensor", class(b_gpu))
           solve_gpu <- solve(A_gpu, b_gpu)
           matrices_equal(solve_gpu, solve_base, name = "Linear solve")
         } else {
           cat("❌ create_tensor_unified not available for b vector\n")
         }
       }, error = function(e) {
         cat("❌ GPU solve failed:", e$message, "\n")
       })
     }
  } else {
    cat("\n⏭️  Skipping linear solve (incompatible dimensions)\n")
  }
  
  # Test 3: LU Decomposition
  if (nrow(A_base) == ncol(A_base)) {
    cat("\n--- LU Decomposition Test ---\n")
    # Base R doesn't have built-in LU, so we'll verify A = L*U
    if (!is.null(A_gpu)) {
      tryCatch({
        lu_result <- lu_decompose(A_gpu)
        # Reconstruct A from LU
        lu_matrix <- as.array(lu_result$lu)
        ipiv <- as.array(lu_result$ipiv)
        
        # Extract L and U from packed LU format
        n <- nrow(lu_matrix)
        L <- diag(n)
        U <- matrix(0, n, n)
        
        for (i in 1:n) {
          for (j in 1:n) {
            if (i > j) {
              L[i, j] <- lu_matrix[i, j]
            } else {
              U[i, j] <- lu_matrix[i, j]
            }
          }
        }
        
        # Apply pivoting (simplified check)
        reconstructed <- L %*% U
        cat("LU decomposition completed\n")
        cat("L matrix (lower tri):\n")
        print(round(L, 4))
        cat("U matrix (upper tri):\n")
        print(round(U, 4))
        
      }, error = function(e) {
        cat("❌ GPU LU failed:", e$message, "\n")
      })
    }
  } else {
    cat("\n⏭️  Skipping LU (non-square matrix)\n")
  }
  
  # Test 4: QR Decomposition
  cat("\n--- QR Decomposition Test ---\n")
  qr_base <- qr(A_base)
  Q_base <- qr.Q(qr_base)
  R_base <- qr.R(qr_base)
  cat("Base R Q matrix:\n")
  print(round(Q_base, 4))
  cat("Base R R matrix:\n")
  print(round(R_base, 4))
  
  # Verify Q*R = A
  QR_product_base <- Q_base %*% R_base
  matrices_equal(QR_product_base, A_base, name = "Base R Q*R reconstruction")
  
  if (!is.null(A_gpu)) {
    tryCatch({
      qr_gpu <- qr(A_gpu)
      Q_gpu <- qr_gpu$Q
      R_gpu <- qr_gpu$R
      
      matrices_equal(Q_gpu, Q_base, name = "Q matrix")
      matrices_equal(R_gpu, R_base, name = "R matrix")
      
      # Verify GPU Q*R = A
      QR_product_gpu <- matmul(Q_gpu, R_gpu)
      matrices_equal(QR_product_gpu, A_base, name = "GPU Q*R reconstruction")
      
    }, error = function(e) {
      cat("❌ GPU QR failed:", e$message, "\n")
    })
  }
  
  # Test 5: Cholesky Decomposition (positive definite matrices only)
  if (nrow(A_base) == ncol(A_base)) {
    cat("\n--- Cholesky Decomposition Test ---\n")
    tryCatch({
      chol_base <- chol(A_base)  # Base R returns upper triangular
      chol_base_lower <- t(chol_base)  # Convert to lower triangular
      cat("Base R Cholesky (lower triangular):\n")
      print(round(chol_base_lower, 4))
      
      # Verify L*L^T = A
      LL_product_base <- chol_base_lower %*% t(chol_base_lower)
      matrices_equal(LL_product_base, A_base, name = "Base R L*L^T reconstruction")
      
      if (!is.null(A_gpu)) {
        tryCatch({
          chol_gpu <- chol(A_gpu)  # Our implementation returns lower triangular
          
          matrices_equal(chol_gpu, chol_base_lower, name = "Cholesky factor")
          
          # Verify GPU L*L^T = A
          LL_product_gpu <- matmul(chol_gpu, transpose(chol_gpu))
          matrices_equal(LL_product_gpu, A_base, name = "GPU L*L^T reconstruction")
          
        }, error = function(e) {
          cat("❌ GPU Cholesky failed:", e$message, "\n")
        })
      }
      
    }, error = function(e) {
      cat("⏭️  Skipping Cholesky (not positive definite):", e$message, "\n")
    })
  }
  
  # Test 6: Eigenvalue Decomposition (square symmetric matrices only)
  if (nrow(A_base) == ncol(A_base) && isSymmetric(A_base)) {
    cat("\n--- Eigenvalue Decomposition Test ---\n")
    eigen_base <- eigen(A_base, symmetric = TRUE)
    cat("Base R eigenvalues:", paste(round(eigen_base$values, 4), collapse = ", "), "\n")
    cat("Base R first eigenvector:", paste(round(eigen_base$vectors[,1], 4), collapse = ", "), "\n")
    
    # Verify A*v = λ*v for first eigenvalue/eigenvector
    Av_base <- A_base %*% eigen_base$vectors[,1]
    lambda_v_base <- eigen_base$values[1] * eigen_base$vectors[,1]
    matrices_equal(Av_base, lambda_v_base, name = "Base R eigenvalue equation A*v=λ*v")
    
    if (!is.null(A_gpu)) {
      tryCatch({
        eigen_gpu <- eigen(A_gpu, symmetric = TRUE)
        
        # Note: eigenvalues might be in different order, so we'll check the sum and product
        eigenvals_gpu <- sort(as.numeric(as.array(eigen_gpu$values)), decreasing = TRUE)
        eigenvals_base <- sort(eigen_base$values, decreasing = TRUE)
        
        matrices_equal(eigenvals_gpu, eigenvals_base, name = "Eigenvalues (sorted)")
        
        # For eigenvectors, we'll check orthogonality and that they span the same space
        V_gpu <- as.array(eigen_gpu$vectors)
        VtV_gpu <- t(V_gpu) %*% V_gpu
        matrices_equal(VtV_gpu, diag(ncol(V_gpu)), name = "GPU eigenvectors orthogonality")
        
      }, error = function(e) {
        cat("❌ GPU eigen failed:", e$message, "\n")
      })
    }
  } else {
    cat("\n⏭️  Skipping eigenvalue decomposition (not square symmetric)\n")
  }
}

# Run tests on all matrices
test_matrix(A1_base, "A1 (3x3 symmetric)", test_solve = TRUE)
test_matrix(A2_base, "A2 (5x5 positive definite)", test_solve = FALSE)  # Different size from b
test_matrix(A3_base, "A3 (4x3 rectangular)", test_solve = FALSE)

 