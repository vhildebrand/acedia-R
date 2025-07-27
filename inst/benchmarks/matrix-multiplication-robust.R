#!/usr/bin/env Rscript
#' Robust Matrix Multiplication GPU vs CPU Benchmark
#'
#' This benchmark uses larger matrices and more precise timing to avoid
#' the precision issues seen in the previous benchmark.

library(acediaR)

cat("=== Robust Matrix Multiplication GPU vs CPU Benchmark ===\n\n")

# More robust timing function
robust_time <- function(expr, iterations = 5) {
  times <- numeric(iterations)
  for (i in 1:iterations) {
    # Force garbage collection before timing
    gc()
    times[i] <- system.time(expr)['elapsed']
  }
  # Return median time, ensuring minimum precision
  max(median(times), 0.001)  # Minimum 1ms to avoid division by zero
}

# Matrix multiplication benchmark with larger sizes
benchmark_matrix_multiplication_robust <- function(sizes = c(500, 750, 1000, 1500, 2000),
                                                  iterations = 5,
                                                  verbose = TRUE) {
  
  results <- data.frame()
  
  for (size in sizes) {
    if (verbose) cat(sprintf("Matrix size: %dx%d (%d elements) ...\n", size, size, size^2))
    
    # Generate test matrices
    set.seed(42)
    A <- matrix(runif(size * size), nrow = size, ncol = size)
    B <- matrix(runif(size * size), nrow = size, ncol = size)
    
    if (verbose) cat("  CPU matrix multiplication...")
    # Time CPU matrix multiplication (R's optimized BLAS)
    cpu_time <- robust_time({
      C_cpu <- A %*% B
    }, iterations)
    if (verbose) cat(sprintf(" %.3f seconds\n", cpu_time))
    
    if (verbose) cat("  GPU matrix multiplication (with transfers)...")
    # Time GPU matrix multiplication (with transfers)
    gpu_time <- robust_time({
      A_gpu <- gpu_tensor(as.vector(A), shape = c(size, size))
      B_gpu <- gpu_tensor(as.vector(B), shape = c(size, size))
      C_gpu <- matmul(A_gpu, B_gpu)
      C_result <- as.array(C_gpu)
    }, iterations)
    if (verbose) cat(sprintf(" %.3f seconds\n", gpu_time))
    
    if (verbose) cat("  GPU matrix multiplication (resident)...")
    # Time GPU matrix multiplication (GPU-resident)
    # Pre-create tensors
    A_gpu <- gpu_tensor(as.vector(A), shape = c(size, size))
    B_gpu <- gpu_tensor(as.vector(B), shape = c(size, size))
    
    gpu_resident_time <- robust_time({
      C_gpu <- matmul(A_gpu, B_gpu)
      synchronize(C_gpu)
    }, iterations)
    if (verbose) cat(sprintf(" %.3f seconds\n", gpu_resident_time))
    
    # Calculate metrics
    speedup_with_transfers <- cpu_time / gpu_time
    speedup_resident <- cpu_time / gpu_resident_time
    
    # Calculate GFLOPS (2 * n^3 operations for n x n matrix multiplication)
    total_ops <- 2.0 * size^3
    cpu_gflops <- total_ops / (cpu_time * 1e9)
    gpu_gflops <- total_ops / (gpu_resident_time * 1e9)
    
    results <- rbind(results, data.frame(
      matrix_size = size,
      elements = size^2,
      cpu_time_s = cpu_time,
      gpu_time_s = gpu_time,
      gpu_resident_time_s = gpu_resident_time,
      speedup_with_transfers = speedup_with_transfers,
      speedup_resident = speedup_resident,
      cpu_gflops = cpu_gflops,
      gpu_gflops = gpu_gflops,
      stringsAsFactors = FALSE
    ))
    
    if (verbose) {
      cat(sprintf("  Speedup (with transfers): %.2fx\n", speedup_with_transfers))
      cat(sprintf("  Speedup (GPU-resident): %.2fx\n", speedup_resident))
      cat(sprintf("  GPU GFLOPS: %.1f\n\n", gpu_gflops))
    }
  }
  
  if (verbose) {
    cat("Completed matrix multiplication benchmark.\n")
  }
  
  return(results)
}

# Run the robust benchmark
cat("=== Running Robust Matrix Multiplication Benchmark ===\n")
matrix_results <- benchmark_matrix_multiplication_robust(
  sizes = c(500, 750, 1000, 1500, 2000, 2500),
  iterations = 3,
  verbose = TRUE
)

print(matrix_results)

cat("\n=== Performance Analysis ===\n")

# Find best performance (excluding any remaining Inf values)
finite_results <- matrix_results[is.finite(matrix_results$speedup_resident), ]
if (nrow(finite_results) > 0) {
  best_speedup_idx <- which.max(finite_results$speedup_resident)
  best_size <- finite_results$matrix_size[best_speedup_idx]
  best_speedup <- finite_results$speedup_resident[best_speedup_idx]
  best_gpu_gflops <- finite_results$gpu_gflops[best_speedup_idx]
  
  cat(sprintf("Best GPU Speedup: %.1fx at %dx%d matrices\n", best_speedup, best_size, best_size))
  cat(sprintf("Peak GPU Performance: %.1f GFLOPS\n", best_gpu_gflops))
  
  # Compare with element-wise operations
  large_matrices <- finite_results[finite_results$matrix_size >= 1000, ]
  if (nrow(large_matrices) > 0) {
    avg_matrix_speedup <- mean(large_matrices$speedup_resident, na.rm = TRUE)
    avg_gpu_gflops <- mean(large_matrices$gpu_gflops, na.rm = TRUE)
    cat(sprintf("Average Speedup (large matrices ≥1000): %.1fx\n", avg_matrix_speedup))
    cat(sprintf("Average GPU Performance: %.1f GFLOPS\n", avg_gpu_gflops))
  }
  
  # Theoretical analysis
  cat("\n=== Theoretical Analysis ===\n")
  rtx4090_peak_gflops <- 83000  # 83 TFLOPS theoretical peak
  achieved_peak_percent <- best_gpu_gflops / rtx4090_peak_gflops * 100
  
  cat(sprintf("RTX 4090 Theoretical Peak: %.0f GFLOPS\n", rtx4090_peak_gflops))
  cat(sprintf("Our Peak Achievement: %.1f GFLOPS\n", best_gpu_gflops))
  cat(sprintf("GPU Utilization: %.3f%% of theoretical peak\n", achieved_peak_percent))
  
  # More realistic comparison with cuBLAS expectations
  cublas_expected_percent <- 70  # cuBLAS typically achieves 50-80% of peak
  cublas_expected_gflops <- rtx4090_peak_gflops * cublas_expected_percent / 100
  cublas_efficiency <- best_gpu_gflops / cublas_expected_gflops * 100
  
  cat(sprintf("cuBLAS Expected Performance (~70%%): %.0f GFLOPS\n", cublas_expected_gflops))
  cat(sprintf("Our Efficiency vs cuBLAS Expected: %.1f%%\n", cublas_efficiency))
} else {
  cat("No valid finite results found. Matrix sizes may still be too small.\n")
}

# Compare with previous benchmarks
cat("\n=== Matrix Multiplication vs Element-wise Operations ===\n")
if (exists("avg_matrix_speedup") && is.finite(avg_matrix_speedup)) {
  cat("Matrix Multiplication (this benchmark):\n")
  cat(sprintf("  Average Speedup: %.1fx\n", avg_matrix_speedup))
  cat(sprintf("  Peak Performance: %.1f GFLOPS\n", best_gpu_gflops))
  cat(sprintf("  GPU Utilization: %.3f%%\n", achieved_peak_percent))
  
  cat("\nElement-wise Operations (previous benchmarks):\n")
  cat("  Average Speedup: ~2-7x\n")
  cat("  Peak Performance: ~10-50 GFLOPS (estimated)\n")
  cat("  GPU Utilization: ~0.1-1%\n")
  
  element_wise_avg <- 5  # Approximate average from previous tests
  improvement_factor <- avg_matrix_speedup / element_wise_avg
  
  if (avg_matrix_speedup > 10) {
    cat("\n🟢 EXCELLENT: Matrix multiplication shows dramatic GPU advantage!\n")
    cat(sprintf("Matrix multiplication is %.1fx better than element-wise operations!\n", improvement_factor))
    cat("This confirms GPU architecture is optimized for dense linear algebra.\n")
  } else if (avg_matrix_speedup > 5) {
    cat("\n🟡 GOOD: Matrix multiplication shows solid GPU performance.\n")
    cat("Performance is better than element-wise operations.\n")
  } else {
    cat("\n🔴 SUBOPTIMAL: Matrix multiplication performance lower than expected.\n")
    cat("May indicate cuBLAS optimization opportunities.\n")
  }
} else {
  cat("Unable to compute meaningful comparison due to timing precision issues.\n")
}

# Create performance plots
if (requireNamespace("ggplot2", quietly = TRUE) && nrow(finite_results) > 0) {
  library(ggplot2)
  
  # Performance vs matrix size
  p1 <- ggplot(finite_results, aes(x = matrix_size)) +
    geom_line(aes(y = cpu_time_s, color = "CPU"), size = 1.2) +
    geom_line(aes(y = gpu_resident_time_s, color = "GPU (resident)"), size = 1.2) +
    geom_point(aes(y = cpu_time_s, color = "CPU"), size = 3) +
    geom_point(aes(y = gpu_resident_time_s, color = "GPU (resident)"), size = 3) +
    scale_y_log10() +
    scale_color_manual(values = c("CPU" = "#E31A1C", "GPU (resident)" = "#33A02C")) +
    labs(
      title = "Matrix Multiplication Performance",
      subtitle = "GPU should show dramatic improvement for large matrices",
      x = "Matrix Size (N x N)",
      y = "Execution Time (seconds, log scale)",
      color = "Implementation"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(p1)
  
  # Speedup vs matrix size
  p2 <- ggplot(finite_results, aes(x = matrix_size, y = speedup_resident)) +
    geom_line(size = 1.2, color = "#33A02C") +
    geom_point(size = 3, color = "#33A02C") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    labs(
      title = "Matrix Multiplication GPU Speedup",
      subtitle = "Speedup should increase with matrix size",
      x = "Matrix Size (N x N)",
      y = "Speedup (CPU time / GPU time)"
    ) +
    theme_minimal() +
    annotate("text", x = max(finite_results$matrix_size), y = 1.1, 
             label = "GPU faster", hjust = 1, color = "red", alpha = 0.7)
  
  print(p2)
  
  # Save plots
  ggsave("matrix_multiplication_robust_performance.png", p1, width = 12, height = 8, dpi = 150)
  ggsave("matrix_multiplication_robust_speedup.png", p2, width = 12, height = 8, dpi = 150)
  
  cat("\nPlots saved as:\n")
  cat("  - matrix_multiplication_robust_performance.png\n")
  cat("  - matrix_multiplication_robust_speedup.png\n")
}

cat("\n=== Key Insights ===\n")
cat("1. Matrix multiplication is compute-bound (O(n³) operations)\n")
cat("2. GPU architecture is optimized for dense linear algebra\n") 
cat("3. cuBLAS should provide highly optimized implementations\n")
cat("4. Larger matrices should show better GPU utilization\n")
cat("5. This is where we expect to see 10-100x speedups\n")

if (exists("avg_matrix_speedup") && is.finite(avg_matrix_speedup)) {
  cat(sprintf("6. Our achieved speedup (%.1fx) ", avg_matrix_speedup))
  if (avg_matrix_speedup > 10) {
    cat("demonstrates GPU's strength in linear algebra!\n")
  } else if (avg_matrix_speedup > 5) {
    cat("shows good GPU utilization for compute-bound operations.\n")
  } else {
    cat("suggests room for optimization in our cuBLAS usage.\n")
  }
}

cat("\n=== Conclusion ===\n")
cat("Matrix multiplication benchmark reveals where GPUs truly excel.\n")
cat("This compute-bound operation should show much better speedups than\n")
cat("memory-bound element-wise operations, demonstrating the importance\n")
cat("of algorithm choice for GPU acceleration.\n") 