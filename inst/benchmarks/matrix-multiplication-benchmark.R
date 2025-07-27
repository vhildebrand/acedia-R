#!/usr/bin/env Rscript
#' Matrix Multiplication GPU vs CPU Benchmark
#'
#' This benchmark tests matrix multiplication performance, where GPUs should
#' show their true strength with much higher speedups than element-wise operations.

library(acediaR)

cat("=== Matrix Multiplication GPU vs CPU Benchmark ===\n\n")

cat("GPU Status:\n")
cat("  Available:", gpu_available(), "\n")
cat("  Info:", gpu_info(), "\n\n")

# Matrix multiplication benchmark function
benchmark_matrix_multiplication <- function(sizes = c(100, 250, 500, 1000, 1500),
                                          iterations = 3,
                                          verbose = TRUE) {
  
  results <- data.frame()
  
  for (size in sizes) {
    if (verbose) cat(sprintf("Matrix size: %dx%d ...\n", size, size))
    
    # Generate test matrices
    set.seed(42)
    A <- matrix(runif(size * size), nrow = size, ncol = size)
    B <- matrix(runif(size * size), nrow = size, ncol = size)
    
    # Time CPU matrix multiplication
    cpu_times <- numeric(iterations)
    for (i in 1:iterations) {
      cpu_times[i] <- system.time({
        C_cpu <- A %*% B
      })['elapsed']
    }
    cpu_time <- median(cpu_times)
    
    # Time GPU matrix multiplication (with transfers)
    gpu_times <- numeric(iterations)
    for (i in 1:iterations) {
      gpu_times[i] <- system.time({
        # Convert to tensors and multiply
        A_gpu <- gpu_tensor(as.vector(A), shape = c(size, size))
        B_gpu <- gpu_tensor(as.vector(B), shape = c(size, size))
        C_gpu <- matmul(A_gpu, B_gpu)
        C_result <- as.array(C_gpu)
      })['elapsed']
    }
    gpu_time <- median(gpu_times)
    
    # Time GPU matrix multiplication (GPU-resident)
    gpu_resident_times <- numeric(iterations)
    for (i in 1:iterations) {
      # Pre-create tensors (setup time not counted)
      A_gpu <- gpu_tensor(as.vector(A), shape = c(size, size))
      B_gpu <- gpu_tensor(as.vector(B), shape = c(size, size))
      
      gpu_resident_times[i] <- system.time({
        # Only time the matrix multiplication
        C_gpu <- matmul(A_gpu, B_gpu)
        synchronize(C_gpu)
      })['elapsed']
    }
    gpu_resident_time <- median(gpu_resident_times)
    
    # Calculate metrics
    speedup_with_transfers <- cpu_time / gpu_time
    speedup_resident <- cpu_time / gpu_resident_time
    
    # Calculate GFLOPS (2 * n^3 operations for n x n matrix multiplication)
    total_ops <- 2 * size^3
    cpu_gflops <- total_ops / (cpu_time * 1e9)
    gpu_gflops <- total_ops / (gpu_resident_time * 1e9)
    
    results <- rbind(results, data.frame(
      matrix_size = size,
      elements = size^2,
      cpu_time_ms = cpu_time * 1000,
      gpu_time_ms = gpu_time * 1000,
      gpu_resident_time_ms = gpu_resident_time * 1000,
      speedup_with_transfers = speedup_with_transfers,
      speedup_resident = speedup_resident,
      cpu_gflops = cpu_gflops,
      gpu_gflops = gpu_gflops,
      stringsAsFactors = FALSE
    ))
  }
  
  if (verbose) {
    cat("\nCompleted matrix multiplication benchmark.\n")
  }
  
  return(results)
}

# Run the benchmark
cat("=== Running Matrix Multiplication Benchmark ===\n")
matrix_results <- benchmark_matrix_multiplication(
  sizes = c(100, 250, 500, 750, 1000, 1500),
  iterations = 3,
  verbose = TRUE
)

print(matrix_results)

cat("\n=== Performance Analysis ===\n")

# Find best performance
best_speedup_idx <- which.max(matrix_results$speedup_resident)
best_size <- matrix_results$matrix_size[best_speedup_idx]
best_speedup <- matrix_results$speedup_resident[best_speedup_idx]
best_gpu_gflops <- matrix_results$gpu_gflops[best_speedup_idx]

cat(sprintf("Best GPU Speedup: %.1fx at %dx%d matrices\n", best_speedup, best_size, best_size))
cat(sprintf("Peak GPU Performance: %.1f GFLOPS\n", best_gpu_gflops))

# Compare with element-wise operations
avg_matrix_speedup <- mean(matrix_results$speedup_resident[matrix_results$matrix_size >= 500], na.rm = TRUE)
cat(sprintf("Average Speedup (large matrices): %.1fx\n", avg_matrix_speedup))

# Theoretical analysis
cat("\n=== Theoretical Analysis ===\n")
rtx4090_peak_gflops <- 83000  # 83 TFLOPS theoretical peak
achieved_peak_percent <- best_gpu_gflops / rtx4090_peak_gflops * 100

cat(sprintf("RTX 4090 Theoretical Peak: %.0f GFLOPS\n", rtx4090_peak_gflops))
cat(sprintf("Our Peak Achievement: %.1f GFLOPS\n", best_gpu_gflops))
cat(sprintf("GPU Utilization: %.2f%% of theoretical peak\n", achieved_peak_percent))

# Compare matrix multiplication efficiency vs element-wise
cat("\n=== Matrix Multiplication vs Element-wise Operations ===\n")
cat("Matrix Multiplication (this benchmark):\n")
cat(sprintf("  Average Speedup: %.1fx\n", avg_matrix_speedup))
cat(sprintf("  Peak Performance: %.1f GFLOPS\n", best_gpu_gflops))
cat(sprintf("  GPU Utilization: %.2f%%\n", achieved_peak_percent))

cat("\nElement-wise Operations (previous benchmarks):\n")
cat("  Average Speedup: ~2-7x\n")
cat("  Peak Performance: ~10-50 GFLOPS (estimated)\n")
cat("  GPU Utilization: ~0.1-1%\n")

if (avg_matrix_speedup > 10) {
  cat("\n🟢 EXCELLENT: Matrix multiplication shows dramatic GPU advantage!\n")
  cat("This confirms GPU architecture is optimized for dense linear algebra.\n")
} else if (avg_matrix_speedup > 5) {
  cat("\n🟡 GOOD: Matrix multiplication shows solid GPU performance.\n")
  cat("Performance is better than element-wise operations.\n")
} else {
  cat("\n🔴 SUBOPTIMAL: Matrix multiplication performance lower than expected.\n")
  cat("May indicate cuBLAS optimization opportunities.\n")
}

# Create performance comparison plot
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  
  # Performance vs matrix size
  p1 <- ggplot(matrix_results, aes(x = matrix_size)) +
    geom_line(aes(y = cpu_time_ms, color = "CPU"), size = 1.2) +
    geom_line(aes(y = gpu_resident_time_ms, color = "GPU (resident)"), size = 1.2) +
    geom_point(aes(y = cpu_time_ms, color = "CPU"), size = 3) +
    geom_point(aes(y = gpu_resident_time_ms, color = "GPU (resident)"), size = 3) +
    scale_y_log10() +
    scale_color_manual(values = c("CPU" = "#E31A1C", "GPU (resident)" = "#33A02C")) +
    labs(
      title = "Matrix Multiplication Performance",
      subtitle = "GPU should show dramatic improvement for large matrices",
      x = "Matrix Size (N x N)",
      y = "Execution Time (ms, log scale)",
      color = "Implementation"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(p1)
  
  # Speedup vs matrix size
  p2 <- ggplot(matrix_results, aes(x = matrix_size, y = speedup_resident)) +
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
    annotate("text", x = max(matrix_results$matrix_size), y = 1.1, 
             label = "GPU faster", hjust = 1, color = "red", alpha = 0.7)
  
  print(p2)
  
  # GFLOPS comparison
  p3 <- ggplot(matrix_results, aes(x = matrix_size)) +
    geom_line(aes(y = cpu_gflops, color = "CPU"), size = 1.2) +
    geom_line(aes(y = gpu_gflops, color = "GPU"), size = 1.2) +
    geom_point(aes(y = cpu_gflops, color = "CPU"), size = 3) +
    geom_point(aes(y = gpu_gflops, color = "GPU"), size = 3) +
    scale_y_log10() +
    scale_color_manual(values = c("CPU" = "#E31A1C", "GPU" = "#33A02C")) +
    labs(
      title = "Matrix Multiplication GFLOPS Performance",
      subtitle = "Higher is better - GPU should dominate for large matrices",
      x = "Matrix Size (N x N)",
      y = "Performance (GFLOPS, log scale)",
      color = "Implementation"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(p3)
  
  # Save plots
  ggsave("matrix_multiplication_performance.png", p1, width = 12, height = 8, dpi = 150)
  ggsave("matrix_multiplication_speedup.png", p2, width = 12, height = 8, dpi = 150)
  ggsave("matrix_multiplication_gflops.png", p3, width = 12, height = 8, dpi = 150)
  
  cat("\nPlots saved as:\n")
  cat("  - matrix_multiplication_performance.png\n")
  cat("  - matrix_multiplication_speedup.png\n")
  cat("  - matrix_multiplication_gflops.png\n")
}

cat("\n=== Key Insights ===\n")
cat("1. Matrix multiplication is compute-bound (O(n³) operations)\n")
cat("2. GPU architecture is optimized for dense linear algebra\n") 
cat("3. cuBLAS should provide highly optimized implementations\n")
cat("4. Larger matrices should show better GPU utilization\n")
cat("5. This is where we expect to see 10-100x speedups\n\n")

cat("=== Comparison with Previous Benchmarks ===\n")
cat("Element-wise operations: 2-7x speedup (memory-bound)\n")
cat(sprintf("Matrix multiplication: %.1fx speedup (compute-bound)\n", avg_matrix_speedup))

if (avg_matrix_speedup > 7) {
  improvement_factor <- avg_matrix_speedup / 5  # Compare to ~5x element-wise average
  cat(sprintf("Matrix multiplication is %.1fx better than element-wise operations!\n", improvement_factor))
  cat("This confirms that GPU performance depends heavily on algorithm type.\n")
} else {
  cat("Matrix multiplication performance is similar to element-wise operations.\n")
  cat("This may indicate optimization opportunities in our cuBLAS usage.\n")
}

cat("\n=== Conclusion ===\n")
cat("Matrix multiplication benchmark reveals GPU's true computational strength.\n")
cat("This is the type of operation where GPUs should dominate CPUs significantly.\n") 