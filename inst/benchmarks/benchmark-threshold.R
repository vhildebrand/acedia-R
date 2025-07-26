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
#' # Benchmark element-wise multiplication threshold
#' res <- benchmark_gpu_threshold(op = "multiply")
#' plot_gpu_threshold(res)
#' }
#' @export
benchmark_gpu_threshold <- function(sizes = c(1e3, 1e4, 1e5, 1e6, 1e7),
                                    op = c("multiply", "add", "dot"),
                                    iterations = 10,
                                    verbose = TRUE) {
  op <- match.arg(op)

  if (!gpu_available()) {
    stop("GPU not available for benchmarking")
  }

  # We prefer microbenchmark if installed
  use_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

  # Select R and GPU functions based on op
  fn_cpu <- fn_gpu <- NULL
  if (op == "multiply") {
    fn_cpu <- function(a, b) a * b
    fn_gpu <- function(a, b) gpu_multiply(a, b, warn_fallback = FALSE)
  } else if (op == "add") {
    fn_cpu <- function(a, b) a + b
    fn_gpu <- function(a, b) gpu_add(a, b, warn_fallback = FALSE)
  } else if (op == "dot") {
    fn_cpu <- function(a, b) sum(a * b)
    fn_gpu <- function(a, b) gpu_dot(a, b, warn_fallback = FALSE)
  }

  results <- data.frame()

  for (size in sizes) {
    if (verbose) cat("Size", format(size, scientific = TRUE), "...\n")

    # Prepare random vectors
    set.seed(42)
    a <- runif(size)
    b <- runif(size)

    # CPU timing
    cpu_time_ms <- if (use_microbenchmark) {
      mb <- microbenchmark::microbenchmark(fn_cpu(a, b), times = iterations, unit = "ms")
      median(mb$time) / 1e6
    } else {
      median(replicate(iterations, system.time(fn_cpu(a, b))["elapsed"])) * 1000
    }

    # GPU timing (includes transfer because helper converts to gpuTensor)
    gpu_time_ms <- if (use_microbenchmark) {
      mb <- microbenchmark::microbenchmark(fn_gpu(a, b), times = iterations, unit = "ms")
      median(mb$time) / 1e6
    } else {
      median(replicate(iterations, system.time(fn_gpu(a, b))["elapsed"])) * 1000
    }

    speedup <- cpu_time_ms / gpu_time_ms

    results <- rbind(results, data.frame(
      operation = op,
      size = size,
      cpu_time_ms = cpu_time_ms,
      gpu_time_ms = gpu_time_ms,
      speedup = speedup,
      stringsAsFactors = FALSE
    ))
  }

  if (verbose) {
    cat("\nCompleted benchmark for", op, "operation.\n")
  }

  results
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