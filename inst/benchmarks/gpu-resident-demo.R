#!/usr/bin/env Rscript
#' GPU-Resident Operations Demo
#'
#' This script demonstrates the dramatic performance difference between:
#' 1. Single GPU operations (with transfer overhead)
#' 2. Chained GPU-resident operations (minimal transfers)
#'
#' This addresses the key insight that GPU performance is much better
#' when multiple operations are performed without leaving the GPU.

library(acediaR)

cat("=== GPU-Resident Operations Performance Demo ===\n\n")

cat("GPU Status:\n")
cat("  Available:", gpu_available(), "\n")
cat("  Info:", gpu_info(), "\n\n")

# Compare single operations vs chained operations
sizes <- c(1e4, 5e4, 1e5, 5e5, 1e6)

cat("=== Part 1: Single Operations (High Transfer Overhead) ===\n")
single_results <- benchmark_gpu_threshold(
  op = "multiply", 
  sizes = sizes, 
  iterations = 3, 
  verbose = FALSE
)

cat("Single multiply operation results:\n")
print(single_results[, c("size", "speedup")])
cat("\nObservation: GPU is consistently slower due to transfer overhead\n\n")

cat("=== Part 2: Chained Operations (GPU-Resident) ===\n")
chain_results <- benchmark_gpu_chains(
  sizes = sizes,
  chain_length = 10,
  iterations = 3,
  verbose = FALSE
)

cat("Chained operations results (10 operations per chain):\n")
print(chain_results[, c("size", "speedup", "resident_speedup")])
cat("\nObservation: GPU-resident operations show much better performance!\n\n")

# Create comparison table
cat("=== Performance Comparison Summary ===\n")
comparison <- data.frame(
  Size = sizes,
  Single_Op_Speedup = single_results$speedup,
  Chained_With_Transfers = chain_results$speedup,
  GPU_Resident_Speedup = chain_results$resident_speedup,
  Improvement_Factor = chain_results$resident_speedup / single_results$speedup
)
comparison$Improvement_Factor[is.infinite(comparison$Improvement_Factor)] <- NA

print(comparison)

cat("\n=== Key Insights ===\n")
cat("1. Single operations: GPU consistently slower due to transfer overhead\n")
cat("2. Chained with transfers: Still slower, but gap narrows with more operations\n")
cat("3. GPU-resident chains: Significant speedups achieved!\n")

# Find where GPU-resident becomes faster
gpu_faster <- chain_results[chain_results$resident_speedup > 1 & !is.na(chain_results$resident_speedup), ]
if (nrow(gpu_faster) > 0) {
  threshold_size <- min(gpu_faster$size)
  best_speedup <- max(gpu_faster$resident_speedup, na.rm = TRUE)
  best_size <- gpu_faster[which.max(gpu_faster$resident_speedup), "size"]
  
  cat(sprintf("4. GPU-resident operations become faster at size: %g elements\n", threshold_size))
  cat(sprintf("5. Best GPU-resident speedup: %.2fx at size %g elements\n", best_speedup, best_size))
} else {
  cat("4. GPU-resident operations need larger sizes or longer chains for speedup\n")
}

cat("\n=== Recommendations ===\n")
cat("• For single operations: Use CPU for small to medium data\n")
cat("• For workflows with multiple operations: Keep data on GPU between operations\n")
cat("• GPU shows its strength in computational pipelines, not single operations\n")
cat("• Transfer overhead is the main bottleneck for GPU performance\n\n")

# Generate visualizations
cat("Generating performance visualizations...\n")
plot_gpu_chains(chain_results, save_plots = TRUE)

# Also create a combined comparison plot
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  
  # Prepare data for combined plot
  combined_data <- data.frame(
    Size = rep(sizes, 3),
    Speedup = c(single_results$speedup, 
                chain_results$speedup, 
                chain_results$resident_speedup),
    Type = rep(c("Single Operation", "Chained (with transfers)", "GPU-Resident"), each = length(sizes))
  )
  
  # Remove infinite/NaN values
  combined_data <- combined_data[is.finite(combined_data$Speedup), ]
  
  p_combined <- ggplot(combined_data, aes(x = Size, y = Speedup, color = Type)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_x_log10(labels = function(x) format(x, scientific = TRUE)) +
    scale_color_manual(values = c(
      "Single Operation" = "#E31A1C",
      "Chained (with transfers)" = "#1F78B4",
      "GPU-Resident" = "#33A02C"
    )) +
    labs(
      title = "GPU Performance: Single vs Chained Operations",
      subtitle = "GPU-resident operations eliminate transfer overhead",
      x = "Input Size (elements)",
      y = "Speedup (CPU time / GPU time)",
      color = "Operation Type"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    annotate("text", x = max(combined_data$Size), y = 1.1, 
             label = "GPU faster", hjust = 1, color = "red", alpha = 0.7)
  
  print(p_combined)
  ggsave("gpu_single_vs_chained_comparison.png", p_combined, width = 12, height = 8, dpi = 150)
  cat("Combined comparison plot saved as 'gpu_single_vs_chained_comparison.png'\n")
}

cat("\n=== Demo Complete ===\n")
cat("This demonstrates why GPU performance depends heavily on workflow design!\n") 