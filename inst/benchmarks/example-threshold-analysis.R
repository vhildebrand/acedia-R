#!/usr/bin/env Rscript
#' Example: GPU vs CPU Threshold Analysis
#'
#' This script demonstrates how to use the benchmark_gpu_threshold() and
#' plot_gpu_threshold() functions to analyze when GPU operations become
#' faster than CPU operations.

library(acediaR)

# Check if GPU is available
if (!gpu_available()) {
  stop("This example requires a CUDA-capable GPU")
}

cat("=== GPU vs CPU Threshold Analysis ===\n\n")

# Test different operations
operations <- c("multiply", "add", "dot")
sizes <- c(1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7)

for (op in operations) {
  cat("Benchmarking", op, "operation...\n")
  
  # Run benchmark
  results <- benchmark_gpu_threshold(
    sizes = sizes,
    op = op,
    iterations = 5,  # Fewer iterations for demo
    verbose = FALSE
  )
  
  # Print summary
  cat("\nResults for", op, ":\n")
  print(results[, c("size", "cpu_time_ms", "gpu_time_ms", "speedup")])
  
  # Find threshold
  threshold_idx <- which(results$speedup > 1)[1]
  if (!is.na(threshold_idx)) {
    threshold_size <- results$size[threshold_idx]
    cat("GPU becomes faster at size:", threshold_size, "\n")
  } else {
    cat("GPU never becomes faster in tested range\n")
  }
  
  # Plot if ggplot2 is available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    cat("Generating plot...\n")
    plot_gpu_threshold(results, title = paste("GPU Speedup -", op))
    
    # Save plot
    plot_file <- paste0("gpu_threshold_", op, ".png")
    ggplot2::ggsave(plot_file, width = 10, height = 6, dpi = 150)
    cat("Plot saved as:", plot_file, "\n")
  }
  
  cat("\n", strrep("=", 50), "\n\n")
}

cat("Analysis complete!\n") 