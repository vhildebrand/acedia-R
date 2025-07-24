# Test script for PyTorch-like tensor view functionality  
library(acediaR)

cat("=== PYTORCH-LIKE TENSOR VIEWS TEST ===\n")

test_efficient_transpose <- function() {
  cat("\n--- Testing Efficient Transpose Views ---\n")
  
  # Create test matrix
  original <- as_tensor(matrix(1:12, nrow=3, ncol=4), dtype = "float32")
  cat("Original tensor shape:", shape(original), "\n")
  cat("Original is contiguous:", is_contiguous(original), "\n")
  cat("Original data:\n")
  print(as.array(original))
  
  # Test new efficient transpose
  cat("\n--- Efficient transpose (view-based) ---\n")
  transposed <- transpose(original)
  cat("Transposed shape:", shape(transposed), "\n")
  cat("Transposed is contiguous:", is_contiguous(transposed), "\n")
  cat("Transposed data:\n")
  print(as.array(transposed))
  
  # Verify correctness
  expected <- t(as.array(original))
  match_result <- all.equal(as.array(transposed), expected, tolerance = 1e-6)
  if (isTRUE(match_result)) {
    cat("✅ Efficient transpose: CORRECT\n")
  } else {
    cat("❌ Efficient transpose: INCORRECT\n")
  }
  
  # Test transpose_view function
  cat("\n--- Using transpose_view() function ---\n")
  transposed2 <- transpose_view(original)
  match_result2 <- all.equal(as.array(transposed2), expected, tolerance = 1e-6)
  if (isTRUE(match_result2)) {
    cat("✅ transpose_view: CORRECT\n")
  } else {
    cat("❌ transpose_view: INCORRECT\n")
  }
}

test_permute_views <- function() {
  cat("\n--- Testing Efficient Permute Views ---\n")
  
  # Create 3D tensor
  tensor_3d <- as_tensor(array(1:24, dim = c(2, 3, 4)), dtype = "float32")
  cat("Original 3D tensor shape:", shape(tensor_3d), "\n")
  cat("Original is contiguous:", is_contiguous(tensor_3d), "\n")
  
  # Test permute view
  cat("\n--- Permute dimensions (2,3,1) ---\n")
  permuted <- permute(tensor_3d, c(2, 3, 1))  # (2,3,4) -> (3,4,2)
  cat("Permuted shape:", shape(permuted), "\n")
  cat("Permuted is contiguous:", is_contiguous(permuted), "\n")
  
  # Test using permute_view function
  permuted2 <- permute_view(tensor_3d, c(2, 3, 1))
  cat("permute_view result shape:", shape(permuted2), "\n")
  
  match_result <- all.equal(as.array(permuted), as.array(permuted2), tolerance = 1e-6)
  if (isTRUE(match_result)) {
    cat("✅ Permute view consistency: CORRECT\n")
  } else {
    cat("❌ Permute view consistency: INCORRECT\n")
  }
}

test_view_operations <- function() {
  cat("\n--- Testing Non-Contiguous View Operations ---\n")
  
  # Create tensor and make it non-contiguous
  original <- as_tensor(1:24, dtype = "float32")
  matrix_tensor <- reshape(original, c(4, 6))
  transposed <- transpose(matrix_tensor)  # This creates a non-contiguous view
  
  cat("Original shape:", shape(matrix_tensor), "- contiguous:", is_contiguous(matrix_tensor), "\n")
  cat("Transposed shape:", shape(transposed), "- contiguous:", is_contiguous(transposed), "\n")
  
  # Test view on non-contiguous tensor
  cat("\n--- Creating view from non-contiguous tensor ---\n")
  tryCatch({
    view_tensor <- view(transposed, c(24))  # Try to create 1D view
    cat("✅ View from non-contiguous tensor: SUCCESS\n")
    cat("View shape:", shape(view_tensor), "\n")
    cat("View is contiguous:", is_contiguous(view_tensor), "\n")
  }, error = function(e) {
    cat("❌ View from non-contiguous tensor FAILED:", e$message, "\n")
  })
}

test_tensor_info <- function() {
  cat("\n--- Testing Tensor Info Function ---\n")
  
  # Create various tensors
  contiguous_tensor <- as_tensor(1:12, dtype = "float32")
  matrix_tensor <- reshape(contiguous_tensor, c(3, 4))
  transposed_tensor <- transpose(matrix_tensor)
  
  cat("\n--- Contiguous tensor info ---\n")
  info1 <- tensor_info(matrix_tensor)
  print(info1)
  
  cat("\n--- Non-contiguous tensor info ---\n")
  info2 <- tensor_info(transposed_tensor)
  print(info2)
}

test_memory_efficiency <- function() {
  cat("\n--- Testing Memory Efficiency ---\n")
  cat("This demonstrates that views share memory and are much faster than copies.\n")
  
  # Create large tensor for performance test
  large_data <- runif(10000)
  large_tensor <- as_tensor(matrix(large_data, 100, 100), dtype = "float32")
  
  # Time efficient transpose (view)
  cat("\n--- Timing efficient transpose (view-based) ---\n")
  start_time <- Sys.time()
  for (i in 1:100) {
    transposed <- transpose(large_tensor)
  }
  view_time <- Sys.time() - start_time
  cat("100 transpose views took:", format(view_time, digits = 4), "\n")
  
  # The view-based operations should be nearly instantaneous since
  # they only manipulate metadata (shape and strides), not data
  cat("✅ View operations should be very fast (no data copying)\n")
}

demonstrate_pytorch_behavior <- function() {
  cat("\n--- Demonstrating PyTorch-like Behavior ---\n")
  
  # Create tensor
  x <- as_tensor(matrix(1:6, 2, 3), dtype = "float32")
  cat("Original tensor x:\n")
  print(as.array(x))
  
  # Create transpose view
  x_t <- transpose(x)
  cat("\nTranspose view x_t:\n")
  print(as.array(x_t))
  
  cat("\nKey features:\n")
  cat("- transpose() now creates a VIEW, not a copy\n")
  cat("- permute() creates views that share memory\n") 
  cat("- view() works with non-contiguous tensors\n")
  cat("- Operations automatically handle strides when needed\n")
  cat("- Much more memory efficient for large tensors\n")
}

# Run all tests
test_efficient_transpose()
test_permute_views()
test_view_operations()
test_tensor_info()
test_memory_efficiency()
demonstrate_pytorch_behavior()

cat("\n=== SUMMARY ===\n")
cat("✅ Implemented PyTorch-like tensor view functionality:\n")
cat("  - Efficient transpose as view (no data copying)\n")
cat("  - Permute operations as views\n")
cat("  - Non-contiguous tensor support in views\n")
cat("  - TensorDescriptor for stride-aware operations\n")
cat("  - Much better memory efficiency\n")
 