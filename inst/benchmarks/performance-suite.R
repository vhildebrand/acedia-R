#' Comprehensive Performance Benchmarking Suite
#'
#' This script provides a complete performance analysis of acediaR operations,
#' measuring GPU vs CPU performance across different operations and sizes,
#' and generating detailed reports and visualizations.

library(acediaR)

#' Run comprehensive performance benchmark
#'
#' @param operations Character vector of operations to benchmark
#' @param sizes Numeric vector of input sizes to test
#' @param iterations Number of timing iterations per test
#' @param output_dir Directory to save results and plots
#' @return List containing all benchmark results
#' @export
run_performance_suite <- function(operations = c("multiply", "add", "dot"),
                                  sizes = c(1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7),
                                  iterations = 10,
                                  output_dir = "benchmark_results") {
  
  if (!gpu_available()) {
    stop("GPU not available for benchmarking")
  }
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  cat("=== acediaR Performance Benchmarking Suite ===\n")
  cat("GPU Info:", gpu_info(), "\n")
  cat("Available GPU Memory:", format(gpu_memory_available(), scientific = TRUE), "bytes\n\n")
  
  all_results <- list()
  summary_stats <- data.frame()
  
  for (op in operations) {
    cat("Benchmarking", op, "operation...\n")
    
    # Run benchmark
    results <- benchmark_gpu_threshold(
      sizes = sizes,
      op = op,
      iterations = iterations,
      verbose = TRUE
    )
    
    all_results[[op]] <- results
    
    # Calculate summary statistics
    avg_speedup <- mean(results$speedup)
    max_speedup <- max(results$speedup)
    threshold_idx <- which(results$speedup > 1)[1]
    threshold_size <- if (!is.na(threshold_idx)) results$size[threshold_idx] else NA
    
    summary_stats <- rbind(summary_stats, data.frame(
      operation = op,
      avg_speedup = avg_speedup,
      max_speedup = max_speedup,
      threshold_size = threshold_size,
      stringsAsFactors = FALSE
    ))
    
    # Save individual results
    write.csv(results, file.path(output_dir, paste0(op, "_results.csv")), row.names = FALSE)
    
    # Generate and save plot
    if (requireNamespace("ggplot2", quietly = TRUE)) {
      plot_file <- file.path(output_dir, paste0(op, "_speedup.png"))
      png(plot_file, width = 1200, height = 800, res = 150)
      plot_gpu_threshold(results, title = paste("GPU Speedup -", toupper(op), "Operation"))
      dev.off()
      cat("Plot saved:", plot_file, "\n")
    }
    
    cat("\n")
  }
  
  # Generate summary report
  cat("=== PERFORMANCE SUMMARY ===\n")
  print(summary_stats)
  
  # Save summary
  write.csv(summary_stats, file.path(output_dir, "performance_summary.csv"), row.names = FALSE)
  
  # Generate combined plot if ggplot2 available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    generate_combined_plot(all_results, output_dir)
  }
  
  # Generate text report
  generate_text_report(all_results, summary_stats, output_dir)
  
  cat("\nBenchmarking complete! Results saved in:", output_dir, "\n")
  
  invisible(list(results = all_results, summary = summary_stats))
}

#' Generate combined speedup plot for all operations
generate_combined_plot <- function(all_results, output_dir) {
  library(ggplot2)
  
  # Combine all results
  combined_df <- do.call(rbind, all_results)
  
  # Create combined plot
  p <- ggplot(combined_df, aes(x = size, y = speedup, colour = operation)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    scale_x_log10(breaks = unique(combined_df$size)) +
    scale_colour_brewer(type = "qual", palette = "Set1") +
    geom_hline(yintercept = 1, linetype = "dashed", colour = "red", alpha = 0.7) +
    labs(
      title = "GPU vs CPU Performance Comparison",
      subtitle = "Speedup ratio across different operations and input sizes",
      x = "Input Size (log10 scale)",
      y = "Speedup (CPU time / GPU time)",
      colour = "Operation"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  # Save plot
  plot_file <- file.path(output_dir, "combined_speedup_comparison.png")
  ggsave(plot_file, p, width = 12, height = 8, dpi = 150)
  cat("Combined plot saved:", plot_file, "\n")
}

#' Generate detailed text report
generate_text_report <- function(all_results, summary_stats, output_dir) {
  report_file <- file.path(output_dir, "performance_report.txt")
  
  sink(report_file)
  
  cat("acediaR Performance Benchmarking Report\n")
  cat("========================================\n\n")
  cat("Generated on:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
  cat("GPU Info:", gpu_info(), "\n")
  cat("Available GPU Memory:", format(gpu_memory_available(), scientific = TRUE), "bytes\n\n")
  
  cat("EXECUTIVE SUMMARY\n")
  cat("-----------------\n")
  for (i in 1:nrow(summary_stats)) {
    row <- summary_stats[i, ]
    cat(sprintf("• %s: Average speedup %.2fx, Max speedup %.2fx\n", 
                toupper(row$operation), row$avg_speedup, row$max_speedup))
    if (!is.na(row$threshold_size)) {
      cat(sprintf("  GPU becomes faster at size: %s\n", format(row$threshold_size, scientific = TRUE)))
    } else {
      cat("  GPU never becomes faster in tested range\n")
    }
  }
  cat("\n")
  
  cat("DETAILED RESULTS\n")
  cat("----------------\n\n")
  
  for (op in names(all_results)) {
    results <- all_results[[op]]
    cat(toupper(op), "OPERATION:\n")
    print(results)
    cat("\n")
  }
  
  cat("RECOMMENDATIONS\n")
  cat("---------------\n")
  
  # Find best performing operation
  best_op <- summary_stats[which.max(summary_stats$avg_speedup), "operation"]
  cat("• Best overall performance:", toupper(best_op), "\n")
  
  # Find operations that benefit from GPU
  gpu_beneficial <- summary_stats[!is.na(summary_stats$threshold_size), ]
  if (nrow(gpu_beneficial) > 0) {
    cat("• Operations that benefit from GPU acceleration:\n")
    for (i in 1:nrow(gpu_beneficial)) {
      row <- gpu_beneficial[i, ]
      cat(sprintf("  - %s (threshold: %s elements)\n", 
                  toupper(row$operation), format(row$threshold_size, scientific = TRUE)))
    }
  }
  
  # Memory transfer overhead analysis
  small_sizes <- summary_stats[summary_stats$operation %in% names(all_results), ]
  cat("• For small operations (< 10^5 elements), consider CPU implementation due to transfer overhead\n")
  cat("• For large operations (> 10^6 elements), GPU acceleration provides significant benefits\n")
  
  sink()
  
  cat("Text report saved:", report_file, "\n")
}

# Example usage (uncomment to run):
# results <- run_performance_suite() 