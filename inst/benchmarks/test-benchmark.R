#!/usr/bin/env Rscript
#' Test GPU vs CPU Benchmarking (Workaround Version)
#'
#' This script demonstrates GPU vs CPU performance comparison
#' by working around the gpu_available() export issue

library(acediaR)

cat("=== GPU vs CPU Performance Test ===\n\n")

# Create a simple gpu_available workaround
gpu_available <- function() {
  tryCatch({
    .Call('_acediaR_gpu_available')
  }, error = function(e) {
    cat("Warning: Could not check GPU availability, assuming TRUE\n")
    TRUE
  })
}

# Test different sizes
sizes <- c(1e3, 5e3, 1e4, 5e4, 1e5)
results <- data.frame()

cat("Testing multiplication across different input sizes...\n")

for (size in sizes) {
  cat(sprintf("Size: %g elements\n", size))
  
  # Generate test data
  set.seed(42)
  a <- runif(size)
  b <- runif(size)
  
  # Time CPU operation
  cpu_time <- system.time({
    cpu_result <- a * b
  })['elapsed']
  
  # Time GPU operation (with direct tensor creation to avoid gpu_available call)
  gpu_time <- system.time({
    # Create tensors directly
    a_gpu <- gpu_tensor(a, shape = length(a))
    b_gpu <- gpu_tensor(b, shape = length(b))
    gpu_result <- a_gpu * b_gpu
    # Convert back to R to ensure computation is complete
    gpu_result_r <- as.array(gpu_result)
  })['elapsed']
  
  speedup <- cpu_time / gpu_time
  
  cat(sprintf("  CPU: %.3f ms, GPU: %.3f ms, Speedup: %.2fx\n", 
              cpu_time * 1000, gpu_time * 1000, speedup))
  
  # Store results
  results <- rbind(results, data.frame(
    size = size,
    cpu_time_ms = cpu_time * 1000,
    gpu_time_ms = gpu_time * 1000,
    speedup = speedup
  ))
}

cat("\n=== Summary Results ===\n")
print(results)

# Find crossover point (where GPU becomes faster than CPU)
gpu_faster <- results[results$speedup > 1, ]
if (nrow(gpu_faster) > 0) {
  threshold_size <- min(gpu_faster$size)
  cat(sprintf("\nGPU becomes faster than CPU at size: %g elements\n", threshold_size))
} else {
  cat("\nGPU was not faster than CPU for any tested size\n")
}

# Create a simple plot if ggplot2 is available
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  
  p <- ggplot(results, aes(x = size)) +
    geom_line(aes(y = cpu_time_ms, color = "CPU"), size = 1) +
    geom_line(aes(y = gpu_time_ms, color = "GPU"), size = 1) +
    geom_point(aes(y = cpu_time_ms, color = "CPU"), size = 2) +
    geom_point(aes(y = gpu_time_ms, color = "GPU"), size = 2) +
    scale_x_log10(labels = function(x) format(x, scientific = TRUE)) +
    scale_y_log10() +
    labs(
      title = "GPU vs CPU Performance Comparison",
      subtitle = "Element-wise multiplication timing",
      x = "Input Size (elements)",
      y = "Execution Time (ms)",
      color = "Implementation"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  ggsave("gpu_cpu_benchmark.png", p, width = 10, height = 6, dpi = 150)
  cat("\nPlot saved as 'gpu_cpu_benchmark.png'\n")
  
  # Also create speedup plot
  p2 <- ggplot(results, aes(x = size, y = speedup)) +
    geom_line(size = 1, color = "blue") +
    geom_point(size = 2, color = "blue") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_x_log10(labels = function(x) format(x, scientific = TRUE)) +
    labs(
      title = "GPU Speedup Over CPU",
      subtitle = "Values > 1 indicate GPU is faster",
      x = "Input Size (elements)",
      y = "Speedup (CPU time / GPU time)"
    ) +
    theme_minimal() +
    annotate("text", x = max(results$size), y = 1.1, 
             label = "GPU faster", hjust = 1, color = "red", alpha = 0.7)
  
  ggsave("gpu_speedup_benchmark.png", p2, width = 10, height = 6, dpi = 150)
  cat("Speedup plot saved as 'gpu_speedup_benchmark.png'\n")
} else {
  cat("\nTo generate plots, install ggplot2: install.packages('ggplot2')\n")
}

cat("\n=== Test Complete ===\n") 