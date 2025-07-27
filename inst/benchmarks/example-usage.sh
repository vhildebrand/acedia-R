#!/bin/bash
#' Example Usage for Matrix Multiplication Benchmark CLI
#'
#' This script demonstrates different ways to use the matrix benchmark CLI

echo "=== Matrix Multiplication Benchmark CLI Examples ==="
echo

# Check if we're in the right directory
if [ ! -f "inst/benchmarks/run-matrix-benchmark.R" ]; then
    echo "Error: Please run this script from the acediaR package root directory"
    exit 1
fi

echo "1. Basic usage (default settings):"
echo "   Rscript inst/benchmarks/run-matrix-benchmark.R"
echo

echo "2. Quick test with small matrices:"
echo "   Rscript inst/benchmarks/run-matrix-benchmark.R --sizes 500,1000,1500"
echo

echo "3. High-precision benchmark with more iterations:"
echo "   Rscript inst/benchmarks/run-matrix-benchmark.R --sizes 1000,2000,3000,4000 --iterations 5"
echo

echo "4. Save plots to specific directory:"
echo "   Rscript inst/benchmarks/run-matrix-benchmark.R --output-dir ./benchmark_results"
echo

echo "5. Quiet mode (minimal output):"
echo "   Rscript inst/benchmarks/run-matrix-benchmark.R --quiet"
echo

echo "6. Skip plot generation (faster):"
echo "   Rscript inst/benchmarks/run-matrix-benchmark.R --no-plots"
echo

echo "7. Large matrix benchmark (may take time):"
echo "   Rscript inst/benchmarks/run-matrix-benchmark.R --sizes 2000,3000,4000,5000"
echo

echo "8. Help information:"
echo "   Rscript inst/benchmarks/run-matrix-benchmark.R --help"
echo

echo "=== Running a Quick Demo ==="
echo "Running: Rscript inst/benchmarks/run-matrix-benchmark.R --sizes 1000,2000 --output-dir ./demo_plots"
echo

# Run a quick demo
Rscript inst/benchmarks/run-matrix-benchmark.R --sizes 1000,2000 --output-dir ./demo_plots

echo
echo "=== Demo Complete ==="
echo "Check the ./demo_plots directory for generated plots!"
echo
echo "Tip: For best results showing GPU advantage, use larger matrices (≥2000×2000)"
echo "     and ensure your system has sufficient GPU memory." 