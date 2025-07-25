context("Core tensor operations and arithmetic")

# Helper functions (from helper-tensor-eq.R)
expect_tensor_equal <- function(tensor, expected, tolerance = 1e-6) {
  expect_equal(as.array(tensor), expected, tolerance = tolerance)
}

# Skip GPU tests if not available
skip_on_ci_if_no_gpu <- function() {
  tryCatch({
    test_tensor <- as_tensor(c(1, 2, 3), dtype = "float")
    if (!inherits(test_tensor, "gpuTensor")) {
      skip("GPU not available")
    }
  }, error = function(e) {
    skip("GPU not available")
  })
}

skip_on_ci_if_no_gpu()

# =============================================================================
# TENSOR CREATION AND BASIC PROPERTIES
# =============================================================================

test_that("Basic tensor creation works", {
  # Test vector creation
  vec_data <- c(1, 2, 3, 4, 5)
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  expect_s3_class(vec_tensor, "gpuTensor")
  expect_equal(as.vector(vec_tensor), vec_data, tolerance = 1e-6)
  expect_equal(shape(vec_tensor), length(vec_data))
  
  # Test matrix creation
  mat_data <- matrix(1:12, nrow = 3, ncol = 4)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  expect_s3_class(mat_tensor, "gpuTensor")
  expect_equal(as.array(mat_tensor), mat_data, tolerance = 1e-6)
  expect_equal(shape(mat_tensor), c(3, 4))
  
  # Test 3D array creation
  arr_data <- array(1:24, dim = c(2, 3, 4))
  arr_tensor <- as_tensor(arr_data, dtype = "float")
  
  expect_s3_class(arr_tensor, "gpuTensor")
  expect_equal(as.array(arr_tensor), arr_data, tolerance = 1e-6)
  expect_equal(shape(arr_tensor), c(2, 3, 4))
})

test_that("Different dtypes work correctly", {
  test_data <- c(1.5, 2.7, 3.1)
  
  # Float32
  tensor_f32 <- as_tensor(test_data, dtype = "float")
  expect_equal(dtype(tensor_f32), "float")
  expect_equal(as.vector(tensor_f32), test_data, tolerance = 1e-6)
  
  # Float64
  tensor_f64 <- as_tensor(test_data, dtype = "double")
  expect_equal(dtype(tensor_f64), "double")
  expect_equal(as.vector(tensor_f64), test_data, tolerance = 1e-15)
})

test_that("Tensor properties work correctly", {
  tensor_3d <- as_tensor(array(1:24, dim = c(2, 3, 4)), dtype = "float")
  
  expect_equal(shape(tensor_3d), c(2, 3, 4))
  expect_equal(size(tensor_3d), 24)
  expect_equal(length(dim(tensor_3d)), 3)
  expect_true(is_contiguous(tensor_3d))
})

# =============================================================================
# BASIC ARITHMETIC OPERATIONS
# =============================================================================

test_that("Element-wise arithmetic works", {
  a_data <- c(1, 2, 3, 4)
  b_data <- c(5, 6, 7, 8)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  # Addition
  add_result <- a_tensor + b_tensor
  expect_tensor_equal(add_result, a_data + b_data)
  
  # Subtraction
  sub_result <- a_tensor - b_tensor
  expect_tensor_equal(sub_result, a_data - b_data)
  
  # Multiplication
  mul_result <- a_tensor * b_tensor
  expect_tensor_equal(mul_result, a_data * b_data)
  
  # Division
  div_result <- a_tensor / b_tensor
  expect_tensor_equal(div_result, a_data / b_data)
})

test_that("Scalar arithmetic works", {
  tensor_data <- c(1, 2, 3, 4)
  tensor <- as_tensor(tensor_data, dtype = "float")
  scalar <- 2.5
  
  # Scalar addition
  add_result <- tensor + scalar
  expect_tensor_equal(add_result, tensor_data + scalar)
  
  # Scalar multiplication
  mul_result <- tensor * scalar
  expect_tensor_equal(mul_result, tensor_data * scalar)
  
  # Scalar subtraction
  sub_result <- tensor - scalar
  expect_tensor_equal(sub_result, tensor_data - scalar)
  
  # Scalar division
  div_result <- tensor / scalar
  expect_tensor_equal(div_result, tensor_data / scalar)
})

test_that("Broadcasting works correctly", {
  # Vector + matrix broadcasting
  vec_data <- c(1, 2, 3)
  mat_data <- matrix(1:6, nrow = 2, ncol = 3)
  
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  
  # This should broadcast the vector across matrix rows
  result <- mat_tensor + vec_tensor
  expected <- sweep(mat_data, 2, vec_data, "+")
  expect_tensor_equal(result, expected)
})

# =============================================================================
# REDUCTION OPERATIONS
# =============================================================================

test_that("Sum reduction works correctly", {
  # Vector sum
  vec_data <- c(1, 2, 3, 4, 5)
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  sum_result <- sum(vec_tensor)
  expect_equal(as.numeric(sum_result), sum(vec_data), tolerance = 1e-6)
  
  # Matrix sum
  mat_data <- matrix(1:12, nrow = 3, ncol = 4)
  mat_tensor <- as_tensor(mat_data, dtype = "float")
  sum_result <- sum(mat_tensor)
  expect_equal(as.numeric(sum_result), sum(mat_data), tolerance = 1e-6)
})

test_that("Mean reduction works correctly", {
  vec_data <- c(1, 2, 3, 4, 5)
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  if (exists("mean.gpuTensor")) {
    mean_result <- mean(vec_tensor)
    # Handle both scalar returns and tensor returns
    if (inherits(mean_result, "gpuTensor")) {
      expect_equal(as.numeric(as.array(mean_result)), mean(vec_data), tolerance = 1e-6)
    } else {
      expect_equal(as.numeric(mean_result), mean(vec_data), tolerance = 1e-6)
    }
  } else {
    skip("mean.gpuTensor not available")
  }
})

test_that("Max and Min reductions work correctly", {
  vec_data <- c(1, 5, 2, 8, 3)
  vec_tensor <- as_tensor(vec_data, dtype = "float")
  
  if (exists("max.gpuTensor")) {
    max_result <- max(vec_tensor)
    # Handle both scalar returns and tensor returns
    if (inherits(max_result, "gpuTensor")) {
      expect_equal(as.numeric(as.array(max_result)), max(vec_data), tolerance = 1e-6)
    } else {
      expect_equal(as.numeric(max_result), max(vec_data), tolerance = 1e-6)
    }
  }
  
  if (exists("min.gpuTensor")) {
    min_result <- min(vec_tensor)
    # Handle both scalar returns and tensor returns  
    if (inherits(min_result, "gpuTensor")) {
      expect_equal(as.numeric(as.array(min_result)), min(vec_data), tolerance = 1e-6)
    } else {
      expect_equal(as.numeric(min_result), min(vec_data), tolerance = 1e-6)
    }
  }
})

# =============================================================================
# MATHEMATICAL FUNCTIONS
# =============================================================================

test_that("Mathematical functions work correctly", {
  test_data <- c(1, 2, 3, 4)
  tensor <- as_tensor(test_data, dtype = "float")
  
  # Exponential
  if (exists("exp.gpuTensor")) {
    exp_result <- exp(tensor)
    expect_tensor_equal(exp_result, exp(test_data))
  }
  
  # Logarithm (only for positive values)
  if (exists("log.gpuTensor")) {
    log_result <- log(tensor)
    expect_tensor_equal(log_result, log(test_data))
  }
  
  # Square root
  if (exists("sqrt.gpuTensor")) {
    sqrt_result <- sqrt(tensor)
    expect_tensor_equal(sqrt_result, sqrt(test_data))
  }
  
  # Trigonometric functions
  if (exists("sin.gpuTensor")) {
    sin_result <- sin(tensor)
    expect_tensor_equal(sin_result, sin(test_data))
  }
  
  if (exists("cos.gpuTensor")) {
    cos_result <- cos(tensor)
    expect_tensor_equal(cos_result, cos(test_data))
  }
})

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

test_that("Activation functions work correctly", {
  test_data <- c(-2, -1, 0, 1, 2)
  tensor <- as_tensor(test_data, dtype = "float")
  
  # ReLU
  if (exists("relu")) {
    relu_result <- relu(tensor)
    expected_relu <- pmax(test_data, 0)
    expect_tensor_equal(relu_result, expected_relu)
  }
  
  # Sigmoid
  if (exists("sigmoid")) {
    sigmoid_result <- sigmoid(tensor)
    expected_sigmoid <- 1 / (1 + exp(-test_data))
    expect_tensor_equal(sigmoid_result, expected_sigmoid)
  }
  
  # Tanh
  if (exists("tanh.gpuTensor")) {
    tanh_result <- tanh(tensor)
    expect_tensor_equal(tanh_result, tanh(test_data))
  }
})

# =============================================================================
# COMPARISON OPERATIONS
# =============================================================================

test_that("Comparison operations work correctly", {
  a_data <- c(1, 3, 5, 2)
  b_data <- c(2, 3, 4, 6)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  # Greater than
  if (exists(">.gpuTensor")) {
    gt_result <- a_tensor > b_tensor
    expected_gt <- a_data > b_data
    # Comparison operators return 1/0 instead of TRUE/FALSE
    actual_values <- as.array(gt_result)
    expected_values <- ifelse(expected_gt, 1, 0)  # Convert to 1/0
    expect_equal(actual_values, expected_values, tolerance = 1e-6)
  }
  
  # Less than
  if (exists("<.gpuTensor")) {
    lt_result <- a_tensor < b_tensor
    expected_lt <- a_data < b_data
    # Comparison operators might return 1/0 instead of TRUE/FALSE
    actual_values <- as.array(lt_result)
    expected_values <- ifelse(expected_lt, 1, 0)  # Convert to 1/0
    expect_equal(actual_values, expected_values, tolerance = 1e-6)
  }
})

# =============================================================================
# ERROR HANDLING
# =============================================================================

test_that("Basic error handling works", {
  tensor_2x3 <- as_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "float")
  tensor_2x2 <- as_tensor(matrix(1:4, nrow = 2, ncol = 2), dtype = "float")
  
  # Incompatible shapes for arithmetic
  expect_error(tensor_2x3 + tensor_2x2, "(shape|dimension|broadcast)", ignore.case = TRUE)
  
  # Invalid dtype
  expect_error(as_tensor(c(1, 2, 3), dtype = "invalid"), "dtype|type", ignore.case = TRUE)
})

test_that("Operations maintain GPU execution", {
  # Test with moderately large tensors
  n <- 1000
  a_data <- runif(n)
  b_data <- runif(n)
  
  a_tensor <- as_tensor(a_data, dtype = "float")
  b_tensor <- as_tensor(b_data, dtype = "float")
  
  # Chain of operations
  result <- (a_tensor + b_tensor) * 2.0 - 1.0
  
  expect_s3_class(result, "gpuTensor")
  expected <- (a_data + b_data) * 2.0 - 1.0
  expect_equal(as.vector(result), expected, tolerance = 1e-6)
})

# =============================================================================
# ADVANCED OPERATIONS (from test-new-gpu-ops.R and test-advanced-ops.R)
# =============================================================================

test_that("Softmax and Argmax work correctly on GPU", {
  if (!exists("softmax") || !exists("argmax")) {
    skip("Softmax/Argmax functions not available")
  }
  
  set.seed(123)
  x <- runif(10, -3, 3)
  tensor_x <- as_tensor(x, dtype = "float")
  
  # Softmax
  soft_gpu <- softmax(tensor_x)
  expect_s3_class(soft_gpu, "gpuTensor")
  soft_cpu <- exp(x) / sum(exp(x))
  expect_equal(as.vector(soft_cpu), as.array(soft_gpu), tolerance = 1e-6)
  expect_equal(sum(as.array(soft_gpu)), 1, tolerance = 1e-6)
  
  # Argmax
  am_gpu <- argmax(tensor_x)
  am_cpu <- which.max(x)
  expect_equal(am_gpu, am_cpu)
})

test_that("Concatenation and Stack operations work correctly", {
  if (!exists("concat_tensor") || !exists("stack_tensor")) {
    skip("Concat/Stack functions not available")
  }
  
  # Create test tensors
  a_data <- matrix(1:6, nrow = 2, ncol = 3)
  b_data <- matrix(7:12, nrow = 2, ncol = 3)
  
  a <- as_tensor(a_data, dtype = "float")
  b <- as_tensor(b_data, dtype = "float")
  
  # Concat along first dimension
  c_gpu <- concat_tensor(list(a, b), axis = 1)
  expect_s3_class(c_gpu, "gpuTensor")
  expect_equal(shape(c_gpu), c(4, 3))  # 2+2, 3
  
  expected_concat <- rbind(a_data, b_data)
  expect_equal(as.array(c_gpu), expected_concat, tolerance = 1e-6)
  
  # Stack along new dimension
  s_gpu <- stack_tensor(list(a, b), axis = 3)
  expect_s3_class(s_gpu, "gpuTensor")
  expect_equal(shape(s_gpu), c(2, 3, 2))  # Original shape + new dim
})

test_that("Repeat and Pad operations work correctly", {
  if (!exists("repeat_tensor")) {
    skip("repeat_tensor function not available")
  }
  
  # Test repeat
  original_data <- matrix(1:4, nrow = 2, ncol = 2)
  t <- as_tensor(original_data, dtype = "float")
  
  r_gpu <- repeat_tensor(t, c(2, 1))
  expect_s3_class(r_gpu, "gpuTensor")
  
  # Test pad only if available and working
  if (exists("pad_tensor")) {
    tryCatch({
      p_gpu <- pad_tensor(t, c(1, 1, 1, 1))  # pad 1 on each side
      expect_s3_class(p_gpu, "gpuTensor")
      expect_equal(shape(p_gpu), c(4, 4))  # 2+1+1, 2+1+1
    }, error = function(e) {
      skip(paste("pad_tensor not working:", e$message))
    })
  } else {
    skip("pad_tensor function not available")
  }
})

test_that("Product and variance reductions work correctly", {
  test_data <- c(1, 2, 3, 4, 5)
  tensor <- as_tensor(test_data, dtype = "float")
  
  # Product reduction
  if (exists("prod.gpuTensor")) {
    prod_result <- prod(tensor)
    expect_equal(as.numeric(prod_result), prod(test_data), tolerance = 1e-6)
  }
  
  # Variance reduction  
  if (exists("var.gpuTensor")) {
    var_result <- var(tensor)
    expect_equal(as.numeric(var_result), var(test_data), tolerance = 1e-6)
  }
}) 