#' Benchmark GPU vs CPU Performance for Multiplication Operations
#'
#' This function compares the performance of GPU-accelerated multiplication
#' operations against their CPU counterparts across different vector sizes.
#'
#' @param sizes Vector of sizes to test (default: powers of 10 from 1e3 to 1e7)
#' @param iterations Number of iterations for each benchmark (default: 10)
#' @param include_transfer If TRUE, includes data transfer time in GPU benchmarks
#' @param verbose If TRUE, prints detailed progress information
#' 
#' @return A data frame containing benchmark results
#' 
#' @details
#' This function benchmarks the following operations:
#' - Element-wise vector multiplication (gpu_multiply vs *)
#' - Scalar multiplication (gpu_scale vs *)  
#' - Dot product (gpu_dot vs sum(a * b))
#' 
#' The benchmark measures execution time and computes speedup ratios.
#' Results include both raw timing data and summary statistics.
#' 
#' @examples
#' \dontrun{
#' # Basic benchmark
#' results <- benchmark_gpu_multiply()
#' print(results)
#' 
#' # Custom benchmark with specific sizes
#' results <- benchmark_gpu_multiply(
#'   sizes = c(1e4, 1e5, 1e6),
#'   iterations = 5
#' )
#' 
#' # Plot results
#' library(ggplot2)
#' ggplot(results, aes(x = size, y = speedup, color = operation)) +
#'   geom_line() +
#'   scale_x_log10() +
#'   labs(title = "GPU Speedup vs Vector Size",
#'        x = "Vector Size", y = "Speedup Ratio")
#' }
#' 
#' @export
benchmark_gpu_multiply <- function(sizes = c(1e3, 1e4, 1e5, 1e6, 1e7),
                                   iterations = 10,
                                   include_transfer = FALSE,
                                   verbose = TRUE) {
  
  # Check if GPU is available
  if (!gpu_available()) {
    stop("GPU not available for benchmarking")
  }
  
  # Check if required packages are available
  if (!requireNamespace("microbenchmark", quietly = TRUE)) {
    warning("microbenchmark package not available, using system.time() instead")
    use_microbenchmark <- FALSE
  } else {
    use_microbenchmark <- TRUE
  }
  
  results <- data.frame()
  
  for (size in sizes) {
    if (verbose) {
      cat("Benchmarking size:", format(size, scientific = TRUE), "\n")
    }
    
    # Generate test data
    set.seed(42)  # For reproducible results
    a <- runif(size)
    b <- runif(size)
    scalar <- 2.5
    
    # Benchmark element-wise multiplication
    if (verbose) cat("  Element-wise multiplication...")
    
    if (use_microbenchmark) {
      mb_mult <- microbenchmark::microbenchmark(
        CPU = a * b,
        GPU = gpu_multiply(a, b, warn_fallback = FALSE),
        times = iterations,
        unit = "ms"
      )
      cpu_mult_time <- median(mb_mult$time[mb_mult$expr == "CPU"]) / 1e6
      gpu_mult_time <- median(mb_mult$time[mb_mult$expr == "GPU"]) / 1e6
    } else {
      cpu_mult_times <- replicate(iterations, system.time(a * b)[["elapsed"]])
      gpu_mult_times <- replicate(iterations, system.time(gpu_multiply(a, b, warn_fallback = FALSE))[["elapsed"]])
      cpu_mult_time <- median(cpu_mult_times) * 1000  # Convert to ms
      gpu_mult_time <- median(gpu_mult_times) * 1000  # Convert to ms
    }
    
    mult_speedup <- cpu_mult_time / gpu_mult_time
    
    results <- rbind(results, data.frame(
      operation = "element_wise_multiply",
      size = size,
      cpu_time_ms = cpu_mult_time,
      gpu_time_ms = gpu_mult_time,
      speedup = mult_speedup,
      stringsAsFactors = FALSE
    ))
    
    if (verbose) cat(" done\n")
    
    # Benchmark scalar multiplication
    if (verbose) cat("  Scalar multiplication...")
    
    if (use_microbenchmark) {
      mb_scale <- microbenchmark::microbenchmark(
        CPU = a * scalar,
        GPU = gpu_scale(a, scalar, warn_fallback = FALSE),
        times = iterations,
        unit = "ms"
      )
      cpu_scale_time <- median(mb_scale$time[mb_scale$expr == "CPU"]) / 1e6
      gpu_scale_time <- median(mb_scale$time[mb_scale$expr == "GPU"]) / 1e6
    } else {
      cpu_scale_times <- replicate(iterations, system.time(a * scalar)[["elapsed"]])
      gpu_scale_times <- replicate(iterations, system.time(gpu_scale(a, scalar, warn_fallback = FALSE))[["elapsed"]])
      cpu_scale_time <- median(cpu_scale_times) * 1000  # Convert to ms
      gpu_scale_time <- median(gpu_scale_times) * 1000  # Convert to ms
    }
    
    scale_speedup <- cpu_scale_time / gpu_scale_time
    
    results <- rbind(results, data.frame(
      operation = "scalar_multiply",
      size = size,
      cpu_time_ms = cpu_scale_time,
      gpu_time_ms = gpu_scale_time,
      speedup = scale_speedup,
      stringsAsFactors = FALSE
    ))
    
    if (verbose) cat(" done\n")
    
    # Benchmark dot product
    if (verbose) cat("  Dot product...")
    
    if (use_microbenchmark) {
      mb_dot <- microbenchmark::microbenchmark(
        CPU = sum(a * b),
        GPU = gpu_dot(a, b, warn_fallback = FALSE),
        times = iterations,
        unit = "ms"
      )
      cpu_dot_time <- median(mb_dot$time[mb_dot$expr == "CPU"]) / 1e6
      gpu_dot_time <- median(mb_dot$time[mb_dot$expr == "GPU"]) / 1e6
    } else {
      cpu_dot_times <- replicate(iterations, system.time(sum(a * b))[["elapsed"]])
      gpu_dot_times <- replicate(iterations, system.time(gpu_dot(a, b, warn_fallback = FALSE))[["elapsed"]])
      cpu_dot_time <- median(cpu_dot_times) * 1000  # Convert to ms
      gpu_dot_time <- median(gpu_dot_times) * 1000  # Convert to ms
    }
    
    dot_speedup <- cpu_dot_time / gpu_dot_time
    
    results <- rbind(results, data.frame(
      operation = "dot_product",
      size = size,
      cpu_time_ms = cpu_dot_time,
      gpu_time_ms = gpu_dot_time,
      speedup = dot_speedup,
      stringsAsFactors = FALSE
    ))
    
    if (verbose) cat(" done\n\n")
  }
  
  # Add summary statistics
  if (verbose) {
    cat("Benchmark Summary:\n")
    cat("==================\n\n")
    
    for (op in unique(results$operation)) {
      op_results <- results[results$operation == op, ]
      avg_speedup <- mean(op_results$speedup)
      max_speedup <- max(op_results$speedup)
      min_speedup <- min(op_results$speedup)
      
      cat(sprintf("%s:\n", gsub("_", " ", op)))
      cat(sprintf("  Average speedup: %.2fx\n", avg_speedup))
      cat(sprintf("  Maximum speedup: %.2fx (size: %s)\n", 
                  max_speedup, format(op_results$size[which.max(op_results$speedup)], scientific = TRUE)))
      cat(sprintf("  Minimum speedup: %.2fx (size: %s)\n", 
                  min_speedup, format(op_results$size[which.min(op_results$speedup)], scientific = TRUE)))
      cat("\n")
    }
  }
  
  return(results)
}

#' @export
# Note: benchmark_gpuVector_multiply function has been removed
# Use benchmark_gpu_multiply with tensor-based operations instead 