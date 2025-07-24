# Test script for non-contiguous tensor support in linear algebra operations
library(acediaR)

cat("Testing non-contiguous tensor support in linear algebra operations...\n")

# Helper function to create non-contiguous tensors
create_noncontiguous_test <- function() {
  cat("\n=== Creating non-contiguous tensors ===\n")
  
  # Create a larger tensor and extract non-contiguous slices
  large_tensor <- as_tensor(1:24, dtype = "float32")
  large_matrix <- reshape(large_tensor, c(4, 6))
  
  cat("Original matrix (4x6):\n")
  print(as.array(large_matrix))
  cat("Is contiguous:", is_contiguous(large_matrix), "\n")
  
  # Create non-contiguous by transposing
  transposed <- transpose(large_matrix)  # Should be 6x4
  cat("\nTransposed matrix (6x4):\n") 
  print(as.array(transposed))
  cat("Is contiguous:", is_contiguous(transposed), "\n")
  
  return(list(
    contiguous_matrix = large_matrix,
    noncontiguous_matrix = transposed
  ))
}

test_outer_product_noncontiguous <- function() {
  cat("\n=== Testing outer product with non-contiguous tensors ===\n")
  
  # Create test vectors - vectors are typically contiguous
  a_data <- c(1, 2, 3)
  b_data <- c(4, 5)
  
  a_tensor <- as_tensor(a_data, dtype = "float32")
  b_tensor <- as_tensor(b_data, dtype = "float32")
  
  cat("Vector a:", a_data, "- contiguous:", is_contiguous(a_tensor), "\n")
  cat("Vector b:", b_data, "- contiguous:", is_contiguous(b_tensor), "\n")
  
  # Test with contiguous vectors
  result_gpu <- outer_product(a_tensor, b_tensor)
  result_cpu <- outer(a_data, b_data)
  
  cat("GPU outer product result:\n")
  print(as.array(result_gpu))
  cat("CPU outer product result:\n")
  print(result_cpu)
  
  match_result <- all.equal(as.array(result_gpu), result_cpu, tolerance = 1e-6)
  if (isTRUE(match_result)) {
    cat("✅ Outer product with contiguous vectors: PASSED\n")
  } else {
    cat("❌ Outer product with contiguous vectors: FAILED\n")
  }
  
  # Note: Vectors are inherently 1D and typically contiguous, 
  # so non-contiguous vector tests are not very meaningful
}

test_matvec_noncontiguous <- function() {
  cat("\n=== Testing matrix-vector operations with non-contiguous tensors ===\n")
  
  test_data <- create_noncontiguous_test()
  contiguous_matrix <- test_data$contiguous_matrix
  noncontiguous_matrix <- test_data$noncontiguous_matrix
  
  # Create test vector
  v_data <- c(1, 2, 3, 4, 5, 6)
  v_tensor <- as_tensor(v_data, dtype = "float32")
  cat("Vector v:", v_data, "- contiguous:", is_contiguous(v_tensor), "\n")
  
  # Test 1: Contiguous matrix × vector
  cat("\n--- Test 1: Contiguous matrix-vector multiplication ---\n")
  cat("Matrix shape:", shape(contiguous_matrix), "- contiguous:", is_contiguous(contiguous_matrix), "\n")
  
  tryCatch({
    result_contiguous <- matvec(contiguous_matrix, v_tensor)
    cat("GPU result (contiguous matrix):", as.array(result_contiguous), "\n")
    
    # CPU verification
    cpu_result <- as.array(contiguous_matrix) %*% v_data
    cat("CPU result:", as.vector(cpu_result), "\n")
    
    match_result <- all.equal(as.array(result_contiguous), as.vector(cpu_result), tolerance = 1e-6)
    if (isTRUE(match_result)) {
      cat("✅ Matrix-vector with contiguous matrix: PASSED\n")
    } else {
      cat("❌ Matrix-vector with contiguous matrix: FAILED\n")
    }
  }, error = function(e) {
    cat("❌ Matrix-vector with contiguous matrix FAILED:", e$message, "\n")
  })
  
  # Test 2: Non-contiguous matrix × vector
  cat("\n--- Test 2: Non-contiguous matrix-vector multiplication ---\n")
  cat("Matrix shape:", shape(noncontiguous_matrix), "- contiguous:", is_contiguous(noncontiguous_matrix), "\n")
  
  # Adjust vector size for transposed matrix (6x4 needs 4-element vector)
  v_small_data <- c(1, 2, 3, 4)
  v_small_tensor <- as_tensor(v_small_data, dtype = "float32")
  
  tryCatch({
    result_noncontiguous <- matvec(noncontiguous_matrix, v_small_tensor)
    cat("GPU result (non-contiguous matrix):", as.array(result_noncontiguous), "\n")
    
    # CPU verification
    cpu_result <- as.array(noncontiguous_matrix) %*% v_small_data
    cat("CPU result:", as.vector(cpu_result), "\n")
    
    match_result <- all.equal(as.array(result_noncontiguous), as.vector(cpu_result), tolerance = 1e-6)
    if (isTRUE(match_result)) {
      cat("✅ Matrix-vector with non-contiguous matrix: PASSED\n")
    } else {
      cat("❌ Matrix-vector with non-contiguous matrix: FAILED\n")
    }
  }, error = function(e) {
    cat("❌ Matrix-vector with non-contiguous matrix FAILED:", e$message, "\n")
  })
}

test_vecmat_noncontiguous <- function() {
  cat("\n=== Testing vector-matrix operations with non-contiguous tensors ===\n")
  
  test_data <- create_noncontiguous_test()
  contiguous_matrix <- test_data$contiguous_matrix     # 4x6
  noncontiguous_matrix <- test_data$noncontiguous_matrix  # 6x4
  
  # Test 1: Vector × contiguous matrix
  cat("\n--- Test 1: Vector × contiguous matrix ---\n")
  v_data <- c(1, 2, 3, 4)  # Length must match matrix rows
  v_tensor <- as_tensor(v_data, dtype = "float32")
  
  tryCatch({
    result_contiguous <- vecmat(v_tensor, contiguous_matrix)
    cat("GPU result (contiguous matrix):", as.array(result_contiguous), "\n")
    
    # CPU verification
    cpu_result <- v_data %*% as.array(contiguous_matrix)
    cat("CPU result:", as.vector(cpu_result), "\n")
    
    match_result <- all.equal(as.array(result_contiguous), as.vector(cpu_result), tolerance = 1e-6)
    if (isTRUE(match_result)) {
      cat("✅ Vector-matrix with contiguous matrix: PASSED\n")
    } else {
      cat("❌ Vector-matrix with contiguous matrix: FAILED\n")
    }
  }, error = function(e) {
    cat("❌ Vector-matrix with contiguous matrix FAILED:", e$message, "\n")
  })
  
  # Test 2: Vector × non-contiguous matrix
  cat("\n--- Test 2: Vector × non-contiguous matrix ---\n")
  v_data_2 <- c(1, 2, 3, 4, 5, 6)  # Length must match non-contiguous matrix rows
  v_tensor_2 <- as_tensor(v_data_2, dtype = "float32")
  
  tryCatch({
    result_noncontiguous <- vecmat(v_tensor_2, noncontiguous_matrix)
    cat("GPU result (non-contiguous matrix):", as.array(result_noncontiguous), "\n")
    
    # CPU verification
    cpu_result <- v_data_2 %*% as.array(noncontiguous_matrix)
    cat("CPU result:", as.vector(cpu_result), "\n")
    
    match_result <- all.equal(as.array(result_noncontiguous), as.vector(cpu_result), tolerance = 1e-6)
    if (isTRUE(match_result)) {
      cat("✅ Vector-matrix with non-contiguous matrix: PASSED\n")
    } else {
      cat("❌ Vector-matrix with non-contiguous matrix: FAILED\n")
    }
  }, error = function(e) {
    cat("❌ Vector-matrix with non-contiguous matrix FAILED:", e$message, "\n")
  })
}

# Run all tests
test_outer_product_noncontiguous()
test_matvec_noncontiguous()
test_vecmat_noncontiguous()

cat("\n=== Summary ===\n")
cat("This test verifies whether linear algebra operations work correctly\n")
cat("with both contiguous and non-contiguous tensors.\n")
cat("Non-contiguous tensors typically arise from operations like transpose,\n")
cat("permute, or slicing that change memory access patterns.\n") 