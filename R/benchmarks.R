#' Benchmark GPU vs CPU Threshold for Common Operations
#'
#' Measures execution time for CPU and GPU implementations of a chosen
#' operation across a range of input sizes, then produces a data frame that
#' can be plotted to visualise the crossover point where GPU becomes faster
#' than CPU (speedup > 1).
#'
#' @param sizes Numeric vector of input sizes to benchmark. Defaults to
#'   powers of 10 from 1e3 to 1e7.
#' @param op Character string selecting the operation. One of
#'   "multiply", "add", "dot". Defaults to "multiply".
#' @param iterations Number of timing iterations per size (default 10).
#' @param verbose Logical; print progress to console (default TRUE).
#'
#' @return A data.frame with columns: size, operation, cpu_time_ms,
#'   gpu_time_ms, speedup.
#'
#' @examples
#' \dontrun{
#' # Benchmark element-wise multiplication
#' results <- benchmark_gpu_threshold(op = "multiply")
#' plot_gpu_threshold(results)
#' }
#'
#' @export
benchmark_gpu_threshold <- function(sizes = c(1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6),
                                   op = "multiply", 
                                   iterations = 10,
                                   verbose = TRUE) {
  
  if (!gpu_available()) {
    stop("GPU not available for benchmarking")
  }
  
  if (!op %in% c("multiply", "add", "dot")) {
    stop("Operation must be one of: 'multiply', 'add', 'dot'")
  }
  
  results <- data.frame()
  
  for (size in sizes) {
    if (verbose) cat(sprintf("Size %g ...\n", size))
    
    # Generate test data
    set.seed(42)
    if (op == "dot") {
      a <- runif(size)
      b <- runif(size)
    } else {
      a <- runif(size)
      b <- runif(size)
    }
    
    # Time CPU operation
    cpu_times <- numeric(iterations)
    for (i in 1:iterations) {
      cpu_times[i] <- system.time({
        if (op == "multiply") {
          result_cpu <- a * b
        } else if (op == "add") {
          result_cpu <- a + b
        } else if (op == "dot") {
          result_cpu <- sum(a * b)
        }
      })['elapsed']
    }
    cpu_time <- median(cpu_times)
    
    # Time GPU operation (includes transfer overhead)
    gpu_times <- numeric(iterations)
    for (i in 1:iterations) {
      gpu_times[i] <- system.time({
        if (op == "multiply") {
          result_gpu <- gpu_multiply(a, b, warn_fallback = FALSE)
        } else if (op == "add") {
          result_gpu <- gpu_add(a, b, warn_fallback = FALSE)
        } else if (op == "dot") {
          result_gpu <- gpu_dot(a, b, warn_fallback = FALSE)
        }
      })['elapsed']
    }
    gpu_time <- median(gpu_times)
    
    speedup <- cpu_time / gpu_time
    
    results <- rbind(results, data.frame(
      operation = op,
      size = size,
      cpu_time_ms = cpu_time * 1000,
      gpu_time_ms = gpu_time * 1000,
      speedup = speedup
    ))
  }
  
  if (verbose) {
    cat(sprintf("\nCompleted benchmark for %s operation.\n", op))
  }
  
  return(results)
}

#' Benchmark Chained GPU Operations vs CPU
#'
#' Benchmarks sequences of operations that stay on the GPU vs equivalent
#' CPU operations. This should show much better GPU performance since
#' data doesn't need to transfer back and forth for each operation.
#'
#' @param sizes Numeric vector of input sizes to benchmark
#' @param chain_length Number of operations to chain together (default 5)
#' @param iterations Number of timing iterations per size (default 10)
#' @param verbose Logical; print progress to console (default TRUE)
#'
#' @return A data.frame with columns: size, chain_length, cpu_time_ms,
#'   gpu_time_ms, speedup, gpu_resident_time_ms, resident_speedup
#'
#' @examples
#' \dontrun{
#' # Benchmark chained operations
#' results <- benchmark_gpu_chains(sizes = c(1e4, 1e5, 1e6), chain_length = 10)
#' plot_gpu_chains(results)
#' }
#'
#' @export
benchmark_gpu_chains <- function(sizes = c(1e4, 5e4, 1e5, 5e5, 1e6),
                                 chain_length = 5,
                                 iterations = 10,
                                 verbose = TRUE) {
  
  if (!gpu_available()) {
    stop("GPU not available for benchmarking")
  }
  
  results <- data.frame()
  
  for (size in sizes) {
    if (verbose) cat(sprintf("Size %g, chain length %d ...\n", size, chain_length))
    
    # Generate test data
    set.seed(42)
    a <- runif(size)
    b <- runif(size)
    c <- runif(size)
    
    # Time CPU chained operations
    cpu_times <- numeric(iterations)
    for (i in 1:iterations) {
      cpu_times[i] <- system.time({
        result <- a
        for (j in 1:chain_length) {
          if (j %% 3 == 1) {
            result <- result * b  # multiply
          } else if (j %% 3 == 2) {
            result <- result + c  # add
          } else {
            result <- sqrt(result)  # unary operation
          }
        }
      })['elapsed']
    }
    cpu_time <- median(cpu_times)
    
    # Time GPU chained operations (with transfer overhead)
    gpu_times <- numeric(iterations)
    for (i in 1:iterations) {
      gpu_times[i] <- system.time({
        # Create tensors (transfer to GPU)
        a_gpu <- gpu_tensor(a, shape = length(a))
        b_gpu <- gpu_tensor(b, shape = length(b))
        c_gpu <- gpu_tensor(c, shape = length(c))
        
        # Chain operations on GPU
        result_gpu <- a_gpu
        for (j in 1:chain_length) {
          if (j %% 3 == 1) {
            result_gpu <- result_gpu * b_gpu  # multiply
          } else if (j %% 3 == 2) {
            result_gpu <- result_gpu + c_gpu  # add
          } else {
            result_gpu <- sqrt(result_gpu)    # unary operation
          }
        }
        
        # Transfer back to CPU
        final_result <- as.array(result_gpu)
      })['elapsed']
    }
    gpu_time <- median(gpu_times)
    
    # Time GPU-resident operations (minimal transfer)
    gpu_resident_times <- numeric(iterations)
    for (i in 1:iterations) {
      # Pre-create tensors (setup time not counted)
      a_gpu <- gpu_tensor(a, shape = length(a))
      b_gpu <- gpu_tensor(b, shape = length(b))
      c_gpu <- gpu_tensor(c, shape = length(c))
      
      gpu_resident_times[i] <- system.time({
        # Only time the chained operations, not transfers
        result_gpu <- a_gpu
        for (j in 1:chain_length) {
          if (j %% 3 == 1) {
            result_gpu <- result_gpu * b_gpu
          } else if (j %% 3 == 2) {
            result_gpu <- result_gpu + c_gpu
          } else {
            result_gpu <- sqrt(result_gpu)
          }
                 }
         # Synchronize to ensure operations complete
         synchronize(result_gpu)
       })['elapsed']
    }
    gpu_resident_time <- median(gpu_resident_times)
    
    speedup <- cpu_time / gpu_time
    resident_speedup <- cpu_time / gpu_resident_time
    
    results <- rbind(results, data.frame(
      size = size,
      chain_length = chain_length,
      cpu_time_ms = cpu_time * 1000,
      gpu_time_ms = gpu_time * 1000,
      gpu_resident_time_ms = gpu_resident_time * 1000,
      speedup = speedup,
      resident_speedup = resident_speedup
    ))
  }
  
  if (verbose) {
    cat(sprintf("\nCompleted chained benchmark (chain length: %d).\n", chain_length))
  }
  
  return(results)
}

#' Plot GPU Chain Benchmark Results
#'
#' Creates visualizations showing the performance of chained GPU operations
#' vs CPU, highlighting the benefit of keeping data GPU-resident.
#'
#' @param results Data frame from benchmark_gpu_chains()
#' @param save_plots Logical; save plots to files (default TRUE)
#'
#' @return Invisible list of ggplot objects
#'
#' @examples
#' \dontrun{
#' results <- benchmark_gpu_chains()
#' plot_gpu_chains(results)
#' }
#'
#' @export
plot_gpu_chains <- function(results, save_plots = TRUE) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 is required for plotting. Please install it via install.packages('ggplot2').")
  }
  
  library(ggplot2)
  
  # Plot 1: Execution time comparison
  p1 <- ggplot(results, aes(x = size)) +
    geom_line(aes(y = cpu_time_ms, color = "CPU"), size = 1.2) +
    geom_line(aes(y = gpu_time_ms, color = "GPU (with transfers)"), size = 1.2) +
    geom_line(aes(y = gpu_resident_time_ms, color = "GPU (resident)"), size = 1.2) +
    geom_point(aes(y = cpu_time_ms, color = "CPU"), size = 3) +
    geom_point(aes(y = gpu_time_ms, color = "GPU (with transfers)"), size = 3) +
    geom_point(aes(y = gpu_resident_time_ms, color = "GPU (resident)"), size = 3) +
    scale_x_log10(labels = function(x) format(x, scientific = TRUE)) +
    scale_y_log10() +
    scale_color_manual(values = c(
      "CPU" = "#E31A1C",
      "GPU (with transfers)" = "#1F78B4", 
      "GPU (resident)" = "#33A02C"
    )) +
    labs(
      title = sprintf("Chained Operations Performance (Chain Length: %d)", results$chain_length[1]),
      subtitle = "GPU-resident operations should show significant speedup",
      x = "Input Size (elements)",
      y = "Execution Time (ms)",
      color = "Implementation"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(p1)
  
  # Plot 2: Speedup comparison
  p2 <- ggplot(results, aes(x = size)) +
    geom_line(aes(y = speedup, color = "GPU (with transfers)"), size = 1.2) +
    geom_line(aes(y = resident_speedup, color = "GPU (resident)"), size = 1.2) +
    geom_point(aes(y = speedup, color = "GPU (with transfers)"), size = 3) +
    geom_point(aes(y = resident_speedup, color = "GPU (resident)"), size = 3) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_x_log10(labels = function(x) format(x, scientific = TRUE)) +
    scale_color_manual(values = c(
      "GPU (with transfers)" = "#1F78B4",
      "GPU (resident)" = "#33A02C"
    )) +
    labs(
      title = "GPU Speedup for Chained Operations",
      subtitle = "Values > 1 indicate GPU is faster than CPU",
      x = "Input Size (elements)",
      y = "Speedup (CPU time / GPU time)",
      color = "GPU Mode"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    annotate("text", x = max(results$size), y = 1.1, 
             label = "GPU faster", hjust = 1, color = "red", alpha = 0.7)
  
  print(p2)
  
  if (save_plots) {
    ggsave("gpu_chains_performance.png", p1, width = 12, height = 8, dpi = 150)
    ggsave("gpu_chains_speedup.png", p2, width = 12, height = 8, dpi = 150)
    cat("Plots saved as 'gpu_chains_performance.png' and 'gpu_chains_speedup.png'\n")
  }
  
  invisible(list(performance = p1, speedup = p2))
}

#' Plot GPU Speedup and Threshold
#'
#' Generates a ggplot showing speedup (CPU time / GPU time) versus input size on
#' a log10 x-axis.  A horizontal line at y = 1 indicates parity; the first point
#' where the curve crosses above this line is highlighted and returned.
#'
#' @param df Data frame produced by `benchmark_gpu_threshold()`.
#' @param title Plot title (optional).
#' @return The ggplot object, invisibly.
#' @examples
#' \dontrun{
#' res <- benchmark_gpu_threshold()
#' plot_gpu_threshold(res)
#' }
#' @export
plot_gpu_threshold <- function(df, title = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 is required for plotting. Please install it via install.packages('ggplot2').")
  }
  library(ggplot2)

  if (is.null(title)) {
    title <- paste("GPU Speedup –", unique(df$operation))
  }

  # Identify threshold index (first size where speedup > 1)
  thresh_idx <- which(df$speedup > 1)[1]
  threshold_size <- if (!is.na(thresh_idx)) df$size[thresh_idx] else NA

  p <- ggplot(df, aes(x = size, y = speedup)) +
    geom_line(colour = "steelblue", size = 1) +
    geom_point(size = 2, colour = "steelblue") +
    scale_x_log10(breaks = df$size) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = "red") +
    labs(title = title, x = "Input size (log10 scale)", y = "Speedup (CPU / GPU)") +
    theme_minimal()

  if (!is.na(threshold_size)) {
    p <- p + geom_vline(xintercept = threshold_size, colour = "darkgreen", linetype = "dotted") +
      annotate("text", x = threshold_size, y = max(df$speedup), label = paste0("Threshold = ", threshold_size),
               angle = 90, vjust = -0.5, hjust = 1, colour = "darkgreen")
  }

  print(p)
  invisible(p)
} 