#!/usr/bin/env Rscript
# Performance Benchmark Script for acediaR Linear Algebra Functions
# Compares GPU vs CPU performance across different matrix sizes

library(acediaR)

# Helper function to time operations
time_operation <- function(expr, name = "Operation", n_runs = 5) {
  times <- numeric(n_runs)
  for (i in 1:n_runs) {
    times[i] <- system.time(expr)[["elapsed"]]
  }
  mean_time <- mean(times)
  sd_time <- sd(times)
  cat(sprintf("%-20s: %.4f ± %.4f seconds\n", name, mean_time, sd_time))
  return(mean_time)
}

# Helper function to create test matrices
create_test_matrices <- function(n) {
  set.seed(42)
  
  # Symmetric positive definite matrix
  temp <- matrix(rnorm(n*n), n, n)
  A_spd <- temp %*% t(temp) + diag(n) * 0.1
  
  # General matrix for QR
  A_general <- matrix(rnorm(n*n), n, n)
  
  # Test vector
  b <- rnorm(n)
  
  return(list(spd = A_spd, general = A_general, vector = b))
}

cat("=== acediaR Linear Algebra Performance Benchmark ===\n\n")

# Test different matrix sizes
sizes <- c(100, 500, 1000, 2000)
results <- data.frame(
  size = integer(),
  operation = character(),
  cpu_time = numeric(),
  gpu_time = numeric(),
  speedup = numeric(),
  stringsAsFactors = FALSE
)

for (n in sizes) {
  cat(sprintf("\n=== Matrix Size: %d x %d ===\n", n, n))
  
  # Create test matrices
  matrices <- create_test_matrices(n)
  A_spd <- matrices$spd
  A_general <- matrices$general
  b <- matrices$vector
  
  cat("Created test matrices\n")
  
  # Try to create GPU tensors
  gpu_available <- FALSE
  tryCatch({
    if (exists("create_tensor_unified")) {
      A_spd_gpu <- create_tensor_unified(A_spd)
      class(A_spd_gpu) <- c("gpuTensor", class(A_spd_gpu))
      
      A_general_gpu <- create_tensor_unified(A_general)
      class(A_general_gpu) <- c("gpuTensor", class(A_general_gpu))
      
      b_gpu <- create_tensor_unified(b)
      class(b_gpu) <- c("gpuTensor", class(b_gpu))
      
      gpu_available <- TRUE
      cat("✅ GPU tensors created successfully\n")
    }
  }, error = function(e) {
    cat("❌ GPU tensors not available:", e$message, "\n")
  })
  
  # Benchmark determinant
  cat("\n--- Determinant ---\n")
  cpu_time <- time_operation(det(A_spd), "CPU det", n_runs = 3)
  
  if (gpu_available) {
    tryCatch({
      gpu_time <- time_operation(det(A_spd_gpu), "GPU det", n_runs = 3)
      speedup <- cpu_time / gpu_time
      cat(sprintf("Speedup: %.2fx\n", speedup))
      results <- rbind(results, data.frame(size = n, operation = "det", 
                                          cpu_time = cpu_time, gpu_time = gpu_time, 
                                          speedup = speedup))
    }, error = function(e) {
      cat("❌ GPU det failed:", e$message, "\n")
    })
  } else {
    cat("⏭️  GPU det skipped (not available)\n")
  }
  
  # Benchmark linear solve
  cat("\n--- Linear Solve ---\n")
  cpu_time <- time_operation(solve(A_spd, b), "CPU solve", n_runs = 3)
  
  if (gpu_available) {
    tryCatch({
      gpu_time <- time_operation(solve(A_spd_gpu, b_gpu), "GPU solve", n_runs = 3)
      speedup <- cpu_time / gpu_time
      cat(sprintf("Speedup: %.2fx\n", speedup))
      results <- rbind(results, data.frame(size = n, operation = "solve", 
                                          cpu_time = cpu_time, gpu_time = gpu_time, 
                                          speedup = speedup))
    }, error = function(e) {
      cat("❌ GPU solve failed:", e$message, "\n")
    })
  } else {
    cat("⏭️  GPU solve skipped (not available)\n")
  }
  
  # Benchmark QR decomposition
  cat("\n--- QR Decomposition ---\n")
  cpu_time <- time_operation(qr(A_general), "CPU QR", n_runs = 3)
  
  if (gpu_available) {
    tryCatch({
      gpu_time <- time_operation(qr(A_general_gpu), "GPU QR", n_runs = 3)
      speedup <- cpu_time / gpu_time
      cat(sprintf("Speedup: %.2fx\n", speedup))
      results <- rbind(results, data.frame(size = n, operation = "qr", 
                                          cpu_time = cpu_time, gpu_time = gpu_time, 
                                          speedup = speedup))
    }, error = function(e) {
      cat("❌ GPU QR failed:", e$message, "\n")
    })
  } else {
    cat("⏭️  GPU QR skipped (not available)\n")
  }
  
  # Benchmark Cholesky decomposition
  cat("\n--- Cholesky Decomposition ---\n")
  cpu_time <- time_operation(chol(A_spd), "CPU Cholesky", n_runs = 3)
  
  if (gpu_available) {
    tryCatch({
      gpu_time <- time_operation(chol(A_spd_gpu), "GPU Cholesky", n_runs = 3)
      speedup <- cpu_time / gpu_time
      cat(sprintf("Speedup: %.2fx\n", speedup))
      results <- rbind(results, data.frame(size = n, operation = "chol", 
                                          cpu_time = cpu_time, gpu_time = gpu_time, 
                                          speedup = speedup))
    }, error = function(e) {
      cat("❌ GPU Cholesky failed:", e$message, "\n")
    })
  } else {
    cat("⏭️  GPU Cholesky skipped (not available)\n")
  }
  
  # Benchmark eigenvalue decomposition (only for smaller matrices due to time)
  if (n <= 1000) {
    cat("\n--- Eigenvalue Decomposition ---\n")
    cpu_time <- time_operation(eigen(A_spd, symmetric = TRUE), "CPU eigen", n_runs = 2)
    
    if (gpu_available) {
      tryCatch({
        gpu_time <- time_operation(eigen(A_spd_gpu, symmetric = TRUE), "GPU eigen", n_runs = 2)
        speedup <- cpu_time / gpu_time
        cat(sprintf("Speedup: %.2fx\n", speedup))
        results <- rbind(results, data.frame(size = n, operation = "eigen", 
                                            cpu_time = cpu_time, gpu_time = gpu_time, 
                                            speedup = speedup))
      }, error = function(e) {
        cat("❌ GPU eigen failed:", e$message, "\n")
      })
    } else {
      cat("⏭️  GPU eigen skipped (not available)\n")
    }
  } else {
    cat("\n⏭️  Eigenvalue decomposition skipped (matrix too large)\n")
  }
}

# Print results if any
if (nrow(results) > 0) {
  cat("\nResults:\n")
  print(results)
} 