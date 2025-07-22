# Comprehensive integration tests for views, broadcasting, operations combinations

test_that("views work with all binary operations", {
  skip_if_not(gpu_available(), "GPU not available")
  
  t1 <- gpu_tensor(c(1, 2, 3, 4, 5, 6), c(2, 3))
  scalar <- 2
  
  # Test transpose + binary ops
  t1_transpose <- transpose(t1)
  expect_no_error(t1_transpose + scalar)
  expect_no_error(t1_transpose - scalar)
  expect_no_error(t1_transpose * scalar)
  expect_no_error(t1_transpose / scalar)
  
  # Test reshape + binary ops
  t1_reshape <- reshape(t1, c(3, 2))
  expect_no_error(t1_reshape + scalar)
  expect_no_error(t1_reshape - scalar)
  expect_no_error(t1_reshape * scalar)
  expect_no_error(t1_reshape / scalar)
  
  # Test view + binary ops
  t1_view <- view(t1, c(6))
  expect_no_error(t1_view + scalar)
  expect_no_error(t1_view - scalar)
  expect_no_error(t1_view * scalar)
  expect_no_error(t1_view / scalar)
})

test_that("views work with all unary operations", {
  skip_if_not(gpu_available(), "GPU not available")
  
  t1 <- gpu_tensor(c(1, 4, 9, 16, 25, 36), c(2, 3))
  
  # Test transpose + unary ops
  t1_transpose <- transpose(t1)
  expect_no_error(exp(t1_transpose))
  expect_no_error(log(t1_transpose))
  expect_no_error(sqrt(t1_transpose))
  
  # Test reshape + unary ops
  t1_reshape <- reshape(t1, c(3, 2))
  expect_no_error(exp(t1_reshape))
  expect_no_error(log(t1_reshape))
  expect_no_error(sqrt(t1_reshape))
  
  # Test view + unary ops
  t1_view <- view(t1, c(6))
  expect_no_error(exp(t1_view))
  expect_no_error(log(t1_view))
  expect_no_error(sqrt(t1_view))
})

test_that("views work with all reduction operations", {
  skip_if_not(gpu_available(), "GPU not available")
  
  t1 <- gpu_tensor(c(1, 2, 3, 4, 5, 6), c(2, 3))
  
  # Test transpose + reductions
  t1_transpose <- transpose(t1)
  expect_no_error(sum(t1_transpose))
  expect_no_error(mean(t1_transpose))
  expect_no_error(max(t1_transpose))
  expect_no_error(min(t1_transpose))
  
  # Test reshape + reductions
  t1_reshape <- reshape(t1, c(3, 2))
  expect_no_error(sum(t1_reshape))
  expect_no_error(mean(t1_reshape))
  expect_no_error(max(t1_reshape))
  expect_no_error(min(t1_reshape))
})

test_that("broadcasting works with unary operations", {
  skip_if_not(gpu_available(), "GPU not available")
  
  t1 <- gpu_tensor(c(1, 4, 9, 16), c(2, 2))
  t2 <- gpu_tensor(c(1, 2), c(2, 1))  # Broadcasting compatible
  
  # Test broadcast + unary ops
  broadcasted_add <- t1 + t2
  expect_no_error(exp(broadcasted_add))
  expect_no_error(log(broadcasted_add))
  expect_no_error(sqrt(broadcasted_add))
  
  broadcasted_mul <- t1 * t2
  expect_no_error(exp(broadcasted_mul))
  expect_no_error(sqrt(broadcasted_mul))
})

test_that("complex operation chains work correctly", {
  skip_if_not(gpu_available(), "GPU not available")
  
  t1 <- gpu_tensor(c(1, 4, 9, 16, 25, 36), c(2, 3))
  
  # Complex chain: transpose -> add scalar -> sqrt -> mean
  expect_no_error({
    result <- mean(sqrt(transpose(t1) + 1))
  })
  expect_true(is.numeric(result))
  expect_length(result, 1)
  
  # Another complex chain: reshape -> multiply -> log -> max
  expect_no_error({
    result2 <- max(log(reshape(t1, c(6)) * 2))
  })
  expect_true(is.numeric(result2))
  
  # Chain with broadcasting: broadcast -> exp -> sum
  t2 <- gpu_tensor(c(1, 2), c(2, 1))
  expect_no_error({
    result3 <- sum(exp(t1 + t2))
  })
  expect_true(is.numeric(result3))
})

test_that("tensor-tensor operations work with views", {
  skip_if_not(gpu_available(), "GPU not available")
  
  t1 <- gpu_tensor(c(1, 2, 3, 4), c(2, 2))
  t2 <- gpu_tensor(c(2, 3, 4, 5), c(2, 2))
  
  # Test tensor-tensor operations with views
  t1_transpose <- transpose(t1)
  t2_transpose <- transpose(t2)
  
  expect_no_error(t1_transpose + t2_transpose)
  expect_no_error(t1_transpose - t2_transpose)
  expect_no_error(t1_transpose * t2_transpose)
  expect_no_error(t1_transpose / t2_transpose)
  
  # Verify results are correct
  result_add <- t1_transpose + t2_transpose
  expected_add <- as.vector(t1_transpose) + as.vector(t2_transpose)
  expect_equal(as.vector(result_add), expected_add, tolerance = 1e-6)
})

test_that("all operations preserve correct dtypes", {
  skip_if_not(gpu_available(), "GPU not available")
  
  # Test with double precision
  t1_double <- gpu_tensor(c(1, 2, 3, 4), c(2, 2), dtype = "double")
  
  # All operations should preserve dtype
  expect_equal(attr(exp(t1_double), "dtype"), "double")
  expect_equal(attr(sqrt(t1_double), "dtype"), "double")
  expect_equal(attr(transpose(t1_double), "dtype"), "double")
  expect_equal(attr(t1_double + 1, "dtype"), "double")
  
  # Test with float precision (if supported)
  # Note: Currently our R interface defaults to double, but this tests the concept
  t1_float <- gpu_tensor(c(1, 2, 3, 4), c(2, 2), dtype = "float")
  expect_equal(attr(t1_float * 2, "dtype"), "float")
})

test_that("error handling works across all operation combinations", {
  skip_if_not(gpu_available(), "GPU not available")
  
  t1 <- gpu_tensor(c(1, 2, 3, 4), c(2, 2))
  
  # Test invalid reshapes
  expect_error(reshape(t1, c(3, 3)))  # Wrong size
  expect_error(view(t1, c(5)))        # Wrong size
  
  # Test invalid operations
  expect_error(log(gpu_tensor(c(-1, -2), c(2))))  # Log of negative
  expect_error(sqrt(gpu_tensor(c(-1, -2), c(2)))) # Sqrt of negative
  
  # Test tensor size mismatches (non-broadcasting)
  t2 <- gpu_tensor(c(1, 2, 3), c(3))
  expect_error(t1 + t2)  # Incompatible shapes
}) 