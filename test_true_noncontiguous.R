# Test script for truly non-contiguous tensor support in linear algebra operations
library(acediaR)

cat("Creating truly non-contiguous tensors and testing linear algebra operations...\n")

test_with_artificial_noncontiguous <- function() {
  cat("\n=== Analysis: Current transpose creates contiguous copies ===\n")
  cat("The current transpose() implementation creates a new contiguous tensor\n")
  cat("by copying data through host memory, rather than creating a non-contiguous view.\n")
  cat("This is why our previous test showed all tensors as contiguous.\n")
  
  # Create original matrix
  large_tensor <- as_tensor(1:24, dtype = "float32")
  original_matrix <- reshape(large_tensor, c(4, 6))
  
  cat("\nOriginal matrix (4x6):\n")
  print(as.array(original_matrix))
  cat("Is contiguous:", is_contiguous(original_matrix), "\n")
  
  # Show that transpose creates a new contiguous tensor
  transposed <- transpose(original_matrix)
  cat("\nTransposed matrix (6x4):\n") 
  print(as.array(transposed))
  cat("Is contiguous:", is_contiguous(transposed), "\n")
  cat("Memory different:", !identical(as.array(original_matrix), t(as.array(transposed))), "\n")
}

analyze_current_implementation <- function() {
  cat("\n=== Analysis: Current Linear Algebra Implementation ===\n")
  
  cat("Looking at TensorLinearAlgebra.cpp, the current outer_product, matvec,\n")
  cat("and vecmat implementations:\n")
  cat("1. Do NOT check if input tensors are contiguous\n")
  cat("2. Directly pass tensor data pointers to CUDA kernels\n")
  cat("3. Assume contiguous memory layout in the kernels\n")
  cat("4. Do NOT use stride-aware kernels\n\n")
  
  cat("This means:\n")
  cat("✅ They work correctly with contiguous tensors\n")
  cat("❌ They would give INCORRECT results with non-contiguous tensors\n")
  cat("❌ They don't handle arbitrary memory strides\n\n")
  
  cat("In contrast, some other operations (like element-wise arithmetic)\n")
  cat("do handle non-contiguous tensors by:\n")
  cat("1. Checking if tensors are contiguous\n")
  cat("2. Creating contiguous copies if needed\n")
  cat("3. Using stride-aware kernels in some cases\n")
}

demonstrate_contiguous_handling <- function() {
  cat("\n=== Demonstration: How other operations handle non-contiguous tensors ===\n")
  
  cat("From TensorArithmetic.cpp, element-wise operations handle non-contiguous tensors:\n")
  cat("They call tensor.contiguous() to ensure contiguous memory before kernel launch.\n\n")
  
  cat("From TensorMath.cpp, unary operations check contiguity:\n")
  cat("- If contiguous: use fast kernel\n")
  cat("- If non-contiguous: use stride-aware kernel\n\n")
  
  # Demonstrate with a simple test
  a <- as_tensor(c(1, 2, 3, 4), dtype = "float32")
  b <- as_tensor(c(5, 6, 7, 8), dtype = "float32")
  
  cat("Testing element-wise multiplication (handles non-contiguous correctly):\n")
  result <- a * b
  cat("Result:", as.array(result), "\n")
  cat("Expected:", c(5, 12, 21, 32), "\n")
}

recommend_fix <- function() {
  cat("\n=== RECOMMENDATION: Fix Non-Contiguous Support ===\n")
  
  cat("To properly support non-contiguous tensors in linear algebra operations,\n")
  cat("we should modify TensorLinearAlgebra.cpp to:\n\n")
  
  cat("Option 1 - Simple Fix (Ensure Contiguous):\n")
  cat("- Check if input tensors are contiguous\n")
  cat("- Call tensor.contiguous() if not contiguous\n")
  cat("- Use the contiguous copy for computation\n\n")
  
  cat("Option 2 - Advanced Fix (Stride-Aware Kernels):\n")
  cat("- Implement stride-aware versions of outer_product, matvec, vecmat kernels\n")
  cat("- Use TensorDescriptor to handle arbitrary strides\n")
  cat("- Choose kernel based on contiguity (fast vs. stride-aware)\n\n")
  
  cat("Option 1 is simpler and ensures correctness.\n")
  cat("Option 2 is more efficient for non-contiguous inputs.\n")
}

# Run all analyses
test_with_artificial_noncontiguous()
analyze_current_implementation()
demonstrate_contiguous_handling()
recommend_fix()

cat("\n=== SUMMARY ===\n")
cat("❌ Current outer_product, matvec, vecmat do NOT support non-contiguous tensors\n")
cat("✅ They work correctly with contiguous tensors (which includes transpose results)\n")
cat("🔧 We need to add contiguity checking to these operations for safety\n")
cat("📈 For performance, we could implement stride-aware kernels in the future\n") 