#!/usr/bin/env Rscript
# Generate Linear Algebra Performance Plots for acediaR

library(ggplot2)
library(dplyr)
library(gridExtra)

# Create the benchmark results data frame from our test
results <- data.frame(
  size = c(100, 100, 100, 100, 100, 500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000),
  operation = c("det", "solve", "qr", "chol", "eigen", "det", "solve", "qr", "chol", "eigen", "det", "solve", "qr", "chol", "eigen", "det", "solve", "qr", "chol"),
  cpu_time = c(0.0006666667, 0.0003333333, 0.0000000000, 0.0000000000, 0.0010000000, 0.0006666667, 0.0010000000, 0.0030000000, 0.0006666667, 0.0095000000, 0.0053333333, 0.0040000000, 0.0260000000, 0.0010000000, 0.0435000000, 0.0153333333, 0.0486666667, 0.2620000000, 0.0033333333),
  gpu_time = c(0.0053333333, 0.0003333333, 0.0063333333, 0.0066666667, 0.0005000000, 0.0013333333, 0.0006666667, 0.0140000000, 0.1506666667, 0.0005000000, 0.0043333333, 0.0016666667, 0.0470000000, 0.6030000000, 0.0045000000, 0.0080000000, 0.0050000000, 0.1853333333, 2.4080000000),
  speedup = c(0.125000000, 1.000000000, 0.000000000, 0.000000000, 2.000000000, 0.500000000, 1.500000000, 0.214285714, 0.004424779, 19.000000000, 1.230769231, 2.400000000, 0.553191489, 0.001658375, 9.666666667, 1.916666667, 9.733333333, 1.413669065, 0.001384275),
  stringsAsFactors = FALSE
)

# Filter out operations where CPU time is 0 (measurement issues)
results_clean <- results %>% 
  filter(cpu_time > 0) %>%
  mutate(
    operation = factor(operation, levels = c("det", "solve", "qr", "chol", "eigen"),
                      labels = c("Determinant", "Linear Solve", "QR Decomp", "Cholesky", "Eigenvalues"))
  )

# Create performance comparison plots
cat("Creating performance visualization plots...\n")

# 1. Speedup vs Matrix Size Plot
p1 <- ggplot(results_clean, aes(x = size, y = speedup, color = operation)) +
  geom_line(size = 1.2, alpha = 0.8) +
  geom_point(size = 3, alpha = 0.9) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", alpha = 0.7) +
  scale_x_continuous(trans = "log10", breaks = c(100, 500, 1000, 2000),
                     labels = c("100", "500", "1K", "2K")) +
  scale_y_continuous(trans = "log10", 
                     breaks = c(0.001, 0.01, 0.1, 1, 10, 20),
                     labels = c("0.001x", "0.01x", "0.1x", "1x", "10x", "20x")) +
  labs(title = "GPU vs CPU Performance: Linear Algebra Operations",
       subtitle = "Speedup Factor by Matrix Size (Higher = GPU Faster)",
       x = "Matrix Size (N × N)",
       y = "GPU Speedup Factor (GPU time / CPU time)",
       color = "Operation") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    axis.title = element_text(size = 12),
    legend.title = element_text(size = 12, face = "bold"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  ) +
  annotation_logticks(sides = "bl") +
  scale_color_brewer(type = "qual", palette = "Set1")

# 2. Execution Time Comparison (CPU vs GPU)
results_long <- results_clean %>%
  tidyr::pivot_longer(cols = c(cpu_time, gpu_time), 
                      names_to = "platform", 
                      values_to = "time") %>%
  mutate(platform = factor(platform, levels = c("cpu_time", "gpu_time"),
                          labels = c("CPU", "GPU")))

p2 <- ggplot(results_long, aes(x = size, y = time, color = platform, linetype = operation)) +
  geom_line(size = 1.1, alpha = 0.8) +
  geom_point(size = 2.5, alpha = 0.9) +
  scale_x_continuous(trans = "log10", breaks = c(100, 500, 1000, 2000),
                     labels = c("100", "500", "1K", "2K")) +
  scale_y_continuous(trans = "log10") +
  labs(title = "Execution Time: CPU vs GPU Linear Algebra",
       subtitle = "Wall-clock Time by Matrix Size (Lower = Better)",
       x = "Matrix Size (N × N)",
       y = "Execution Time (seconds)",
       color = "Platform",
       linetype = "Operation") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    axis.title = element_text(size = 12),
    legend.title = element_text(size = 12, face = "bold"),
    panel.grid.minor = element_blank()
  ) +
  annotation_logticks(sides = "bl") +
  scale_color_manual(values = c("CPU" = "#E31A1C", "GPU" = "#1F78B4"))

# 3. Heatmap of Speedup by Operation and Size
p3 <- ggplot(results_clean, aes(x = factor(size), y = operation, fill = log10(speedup))) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.2fx", speedup)), 
            color = "white", fontface = "bold", size = 3.5) +
  scale_fill_gradient2(low = "#D73027", mid = "#FFFFBF", high = "#1A9850",
                       midpoint = 0, name = "Speedup\n(log10)",
                       labels = function(x) sprintf("10^%.1f", x)) +
  labs(title = "GPU Speedup Heatmap: Linear Algebra Operations",
       subtitle = "Red = CPU Faster, Green = GPU Faster",
       x = "Matrix Size (N × N)",
       y = "Operation") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    axis.title = element_text(size = 12),
    legend.title = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )

# Save plots
ggsave("linear_algebra_speedup.png", p1, width = 12, height = 8, dpi = 300)
ggsave("linear_algebra_timing.png", p2, width = 12, height = 8, dpi = 300)
ggsave("linear_algebra_heatmap.png", p3, width = 10, height = 6, dpi = 300)

# Create combined plot
combined_plot <- grid.arrange(p1, p2, p3, ncol = 1, heights = c(1, 1, 0.8))
ggsave("linear_algebra_performance_combined.png", combined_plot, width = 12, height = 16, dpi = 300)

cat("Performance plots saved:\n")
cat("  • linear_algebra_speedup.png - Speedup comparison\n")
cat("  • linear_algebra_timing.png - Execution time comparison\n") 
cat("  • linear_algebra_heatmap.png - Speedup heatmap\n")
cat("  • linear_algebra_performance_combined.png - All plots combined\n")

# Print summary statistics
cat("\n=== Performance Summary ===\n")
cat("Best GPU Performance (highest speedup):\n")
best_gpu <- results_clean %>% 
  arrange(desc(speedup)) %>% 
  head(5)
for (i in 1:nrow(best_gpu)) {
  cat(sprintf("  %s (%dx%d): %.2fx speedup\n", 
              best_gpu$operation[i], best_gpu$size[i], best_gpu$size[i], best_gpu$speedup[i]))
}

cat("\nWorst GPU Performance (CPU faster):\n")
worst_gpu <- results_clean %>% 
  arrange(speedup) %>% 
  head(3)
for (i in 1:nrow(worst_gpu)) {
  cat(sprintf("  %s (%dx%d): %.3fx speedup (CPU %.2fx faster)\n", 
              worst_gpu$operation[i], worst_gpu$size[i], worst_gpu$size[i], 
              worst_gpu$speedup[i], 1/worst_gpu$speedup[i]))
}

cat("\n🚀 Linear algebra performance analysis complete!\n") 