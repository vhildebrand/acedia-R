#!/usr/bin/env Rscript
#' Matrix Multiplication Performance Investigation
#'
#' This script investigates why our matrix multiplication isn't showing
#' the expected GPU speedups and diagnoses potential issues.

library(acediaR)

cat("=== Matrix Multiplication Performance Investigation ===\n\n")

# Test with much larger matrices and more precise timing
investigate_matrix_performance <- function() {
  
  cat("=== Testing Different Matrix Sizes ===\n")
  
  sizes <- c(1000, 2000, 3000, 4000)
  
  for (size in sizes) {
    cat(sprintf("\n--- Matrix Size: %dx%d (%d elements, %.1f MB) ---\n", 
                size, size, size^2, size^2 * 4 / 1024^2))
    
    # Generate large test matrices
    set.seed(42)
    A <- matrix(runif(size * size), nrow = size, ncol = size)
    B <- matrix(runif(size * size), nrow = size, ncol = size)
    
    cat("Generating matrices... done\n")
    
    # Time CPU matrix multiplication (multiple runs for accuracy)
    cat("CPU matrix multiplication: ")
    cpu_times <- numeric(3)
    for (i in 1:3) {
      gc()  # Force garbage collection
      cpu_times[i] <- system.time({
        C_cpu <- A %*% B
      })['elapsed']
    }
    cpu_time <- median(cpu_times)
    cat(sprintf("%.3f seconds\n", cpu_time))
    
    # Time GPU matrix multiplication setup
    cat("GPU tensor creation: ")
    setup_time <- system.time({
      A_gpu <- gpu_tensor(as.vector(A), shape = c(size, size))
      B_gpu <- gpu_tensor(as.vector(B), shape = c(size, size))
    })['elapsed']
    cat(sprintf("%.3f seconds\n", setup_time))
    
    # Time GPU matrix multiplication (resident)
    cat("GPU matrix multiplication (resident): ")
    gpu_times <- numeric(3)
    for (i in 1:3) {
      gpu_times[i] <- system.time({
        C_gpu <- matmul(A_gpu, B_gpu)
        synchronize(C_gpu)
      })['elapsed']
    }
    gpu_time <- median(gpu_times)
    cat(sprintf("%.3f seconds\n", gpu_time))
    
    # Time full GPU pipeline
    cat("GPU full pipeline (with transfers): ")
    full_gpu_time <- system.time({
      A_gpu_full <- gpu_tensor(as.vector(A), shape = c(size, size))
      B_gpu_full <- gpu_tensor(as.vector(B), shape = c(size, size))
      C_gpu_full <- matmul(A_gpu_full, B_gpu_full)
      C_result <- as.array(C_gpu_full)
    })['elapsed']
    cat(sprintf("%.3f seconds\n", full_gpu_time))
    
    # Calculate metrics
    speedup_resident <- cpu_time / gpu_time
    speedup_full <- cpu_time / full_gpu_time
    
    # Calculate GFLOPS
    total_ops <- 2.0 * size^3
    cpu_gflops <- total_ops / (cpu_time * 1e9)
    gpu_gflops <- total_ops / (gpu_time * 1e9)
    
    cat(sprintf("Results:\n"))
    cat(sprintf("  CPU GFLOPS: %.1f\n", cpu_gflops))
    cat(sprintf("  GPU GFLOPS: %.1f\n", gpu_gflops))
    cat(sprintf("  Speedup (resident): %.2fx\n", speedup_resident))
    cat(sprintf("  Speedup (full pipeline): %.2fx\n", speedup_full))
    cat(sprintf("  Transfer overhead: %.3f seconds\n", setup_time))
    
    # Verify correctness
    C_verify <- as.array(C_gpu)
    max_diff <- max(abs(C_cpu - C_verify))
    cat(sprintf("  Max difference: %.2e (should be small)\n", max_diff))
    
    if (speedup_resident < 1.5 && size >= 2000) {
      cat("  ⚠️  WARNING: Low GPU speedup for large matrix!\n")
      cat("     Possible issues:\n")
      cat("     - Not using optimized cuBLAS\n")
      cat("     - Memory bandwidth bottleneck\n")
      cat("     - Kernel launch overhead\n")
      cat("     - Suboptimal tensor layout\n")
    }
  }
}

# Run the investigation
investigate_matrix_performance()

cat("\n=== Comparison with R's Optimized BLAS ===\n")
cat("R uses optimized BLAS libraries (OpenBLAS, MKL, etc.) for matrix operations.\n")
cat("These are highly optimized CPU implementations that can be very fast.\n")
cat("GPU advantage becomes apparent when:\n")
cat("1. Matrix sizes are very large (>5000x5000)\n")
cat("2. Multiple operations are chained\n")
cat("3. Batch operations are performed\n\n")

# Test if R is using optimized BLAS
cat("=== R BLAS Information ===\n")
tryCatch({
  blas_info <- sessionInfo()
  cat("BLAS library in use:\n")
  print(blas_info$BLAS)
  cat("\nLAPACK library in use:\n")
  print(blas_info$LAPACK)
}, error = function(e) {
  cat("Could not determine BLAS/LAPACK information\n")
})

cat("\n=== GPU vs Optimized CPU BLAS Analysis ===\n")
cat("Modern CPUs with optimized BLAS can achieve:\n")
cat("- 100-500 GFLOPS for matrix multiplication\n")
cat("- Highly optimized memory access patterns\n")
cat("- Vectorized instructions (AVX, etc.)\n")
cat("- Multi-core parallelization\n\n")

cat("GPUs typically show advantage when:\n")
cat("- Matrix sizes are very large (memory bandwidth advantage)\n")
cat("- Batch processing multiple matrices\n")
cat("- Mixed precision operations (FP16)\n")
cat("- Part of larger computational pipelines\n\n")

cat("=== Recommendations ===\n")
cat("1. Test with even larger matrices (5000x5000+)\n")
cat("2. Implement batch matrix multiplication\n")
cat("3. Consider mixed precision (FP16) operations\n")
cat("4. Profile GPU utilization during operations\n")
cat("5. Verify cuBLAS is being used optimally\n\n")

cat("=== Key Insight ===\n")
cat("The lack of dramatic GPU speedup for matrix multiplication suggests:\n")
cat("1. R's CPU BLAS is highly optimized (which is good!)\n")
cat("2. Our GPU implementation may need optimization\n")
cat("3. Matrix sizes tested may not be large enough\n")
cat("4. GPU shows strength in different scenarios than single GEMM\n\n")

cat("This is actually common - modern CPUs with optimized BLAS\n")
cat("can be very competitive with GPUs for single matrix operations.\n")
cat("GPU advantage appears in sustained computational workloads.\n") 