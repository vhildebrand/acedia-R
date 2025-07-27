#!/usr/bin/env Rscript
#' Compute-Intensive GPU Operations Test
#'
#' Tests GPU performance with more compute-heavy operations that should
#' show higher GPU utilization and better speedups.

library(acediaR)

cat("=== Compute-Intensive GPU Operations Test ===\n\n")

# Test with more compute-intensive chained operations
benchmark_compute_intensive <- function(sizes = c(1e5, 5e5, 1e6), 
                                       chain_length = 20, 
                                       iterations = 3,
                                       verbose = TRUE) {
  
  results <- data.frame()
  
  for (size in sizes) {
    if (verbose) cat(sprintf("Size %g, compute-intensive chain length %d ...\n", size, chain_length))
    
    # Generate test data
    set.seed(42)
    a <- runif(size, min = 0.1, max = 2.0)  # Avoid zeros for division
    b <- runif(size, min = 0.1, max = 2.0)
    c <- runif(size, min = 0.1, max = 2.0)
    
    # Time CPU compute-intensive operations
    cpu_times <- numeric(iterations)
    for (i in 1:iterations) {
      cpu_times[i] <- system.time({
        result <- a
        for (j in 1:chain_length) {
          if (j %% 6 == 1) {
            result <- result * b           # multiply
          } else if (j %% 6 == 2) {
            result <- result + c           # add
          } else if (j %% 6 == 3) {
            result <- sqrt(result)         # square root
          } else if (j %% 6 == 4) {
            result <- exp(log(result))     # exp(log(x)) = x, but more compute
          } else if (j %% 6 == 5) {
            result <- sin(result) + cos(result)  # trigonometric
          } else {
            result <- result^1.5           # power operation
          }
        }
      })['elapsed']
    }
    cpu_time <- median(cpu_times)
    
    # Time GPU compute-intensive operations (GPU-resident)
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
          if (j %% 6 == 1) {
            result_gpu <- result_gpu * b_gpu        # multiply
          } else if (j %% 6 == 2) {
            result_gpu <- result_gpu + c_gpu        # add
          } else if (j %% 6 == 3) {
            result_gpu <- sqrt(result_gpu)          # square root
          } else if (j %% 6 == 4) {
            result_gpu <- exp(log(result_gpu))      # exp(log(x))
          } else if (j %% 6 == 5) {
            result_gpu <- sin(result_gpu) + cos(result_gpu)  # trigonometric
          } else {
            result_gpu <- result_gpu^1.5            # power operation
          }
        }
        # Synchronize to ensure operations complete
        synchronize(result_gpu)
      })['elapsed']
    }
    gpu_resident_time <- median(gpu_resident_times)
    
    resident_speedup <- cpu_time / gpu_resident_time
    
    results <- rbind(results, data.frame(
      size = size,
      chain_length = chain_length,
      cpu_time_ms = cpu_time * 1000,
      gpu_resident_time_ms = gpu_resident_time * 1000,
      resident_speedup = resident_speedup
    ))
  }
  
  if (verbose) {
    cat(sprintf("\nCompleted compute-intensive benchmark (chain length: %d).\n", chain_length))
  }
  
  return(results)
}

# Test simple operations vs compute-intensive operations
cat("=== Comparison: Simple vs Compute-Intensive Operations ===\n\n")

sizes <- c(1e5, 5e5, 1e6)

cat("1. Simple Operations (multiply, add, sqrt):\n")
simple_results <- benchmark_gpu_chains(
  sizes = sizes,
  chain_length = 10,
  iterations = 3,
  verbose = FALSE
)
print(simple_results[, c("size", "resident_speedup")])

cat("\n2. Compute-Intensive Operations (exp, log, sin, cos, pow):\n")
intensive_results <- benchmark_compute_intensive(
  sizes = sizes,
  chain_length = 20,  # Longer chain to amortize overhead
  iterations = 3,
  verbose = FALSE
)
print(intensive_results[, c("size", "resident_speedup")])

cat("\n=== Performance Comparison ===\n")
comparison <- data.frame(
  Size = sizes,
  Simple_Speedup = simple_results$resident_speedup,
  Intensive_Speedup = intensive_results$resident_speedup,
  Improvement = intensive_results$resident_speedup / simple_results$resident_speedup
)
print(comparison)

cat("\n=== Analysis ===\n")
avg_simple <- mean(simple_results$resident_speedup, na.rm = TRUE)
avg_intensive <- mean(intensive_results$resident_speedup, na.rm = TRUE)

cat(sprintf("Average Speedup - Simple Operations: %.2fx\n", avg_simple))
cat(sprintf("Average Speedup - Compute-Intensive: %.2fx\n", avg_intensive))
cat(sprintf("Improvement Factor: %.2fx\n", avg_intensive / avg_simple))

cat("\n=== Theoretical Analysis for Compute-Intensive Operations ===\n")
# For compute-intensive operations, we should be more compute-bound
size <- 1e6
ops_per_element <- 20  # More complex operations
arithmetic_intensity <- 10  # Rough estimate of ops per memory access

total_ops <- size * ops_per_element
total_gflops <- total_ops / 1e9

# GPU compute capacity (higher utilization for complex ops)
gpu_utilization <- 0.3  # Higher for compute-intensive operations
theoretical_gpu_gflops <- 83000 * gpu_utilization  # 24.9 TFLOPS effective
theoretical_gpu_time_ms <- total_gflops / theoretical_gpu_gflops * 1000

# CPU performance for complex operations
cpu_gflops_complex <- 0.05  # Much slower for complex operations
theoretical_cpu_time_ms <- total_gflops / cpu_gflops_complex * 1000

theoretical_speedup <- theoretical_cpu_time_ms / theoretical_gpu_time_ms

cat(sprintf("For 1M elements with compute-intensive operations:\n"))
cat(sprintf("  Total Operations: %.3f GFLOPS\n", total_gflops))
cat(sprintf("  Theoretical CPU Time: %.1f ms\n", theoretical_cpu_time_ms))
cat(sprintf("  Theoretical GPU Time: %.1f ms\n", theoretical_gpu_time_ms))
cat(sprintf("  Theoretical Speedup: %.1fx\n", theoretical_speedup))

actual_speedup <- intensive_results[intensive_results$size == 1e6, "resident_speedup"]
efficiency <- actual_speedup / theoretical_speedup * 100

cat(sprintf("  Actual Speedup: %.1fx\n", actual_speedup))
cat(sprintf("  GPU Efficiency: %.1f%%\n", efficiency))

cat("\n=== Key Findings ===\n")
if (avg_intensive > avg_simple * 2) {
  cat("🟢 SIGNIFICANT IMPROVEMENT with compute-intensive operations\n")
  cat("• GPU utilization is much better for complex operations\n")
  cat("• Confirms that GPU excels at compute-heavy workloads\n")
} else if (avg_intensive > avg_simple * 1.5) {
  cat("🟡 MODERATE IMPROVEMENT with compute-intensive operations\n")
  cat("• Some benefit from increased arithmetic intensity\n")
} else {
  cat("🔴 LIMITED IMPROVEMENT\n")
  cat("• May still be memory-bound or have other bottlenecks\n")
}

cat("\n=== Summary ===\n")
cat("Element-wise operations (even chained) are inherently limited by:\n")
cat("1. Memory bandwidth (need to read/write data)\n")
cat("2. Low arithmetic intensity (few operations per memory access)\n")
cat("3. Kernel launch overhead\n\n")

cat("GPU shows its true power with:\n")
cat("• High arithmetic intensity operations\n")
cat("• Matrix operations (BLAS)\n")
cat("• Convolutions\n")
cat("• Iterative algorithms\n")
cat("• Parallel reductions\n\n")

cat(sprintf("Our achieved speedups (%.1fx for simple, %.1fx for intensive) are\n", avg_simple, avg_intensive))
cat("actually quite good for this class of operations!\n") 