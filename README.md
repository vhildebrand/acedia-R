# acediaR: High-Performance GPU Computing in R

**`acediaR`** is an R package that brings the massive parallel processing power of NVIDIA GPUs to your R environment. It provides a core `gpuTensor` object and a comprehensive suite of functions for tensor creation, manipulation, and computation, all executed directly on the GPU. This allows for significant acceleration of numerical and linear algebra workloads.

This guide provides a complete walkthrough of the library's features, from initial setup to advanced operations.

## Table of Contents

- [Installation](#installation)
- [Core Concept: The `gpuTensor`](#core-concept-the-gputensor)
- [1. Getting Started: Checking GPU Status](#1-getting-started-checking-gpu-status)
- [2. Tensor Creation and Data Types](#2-tensor-creation-and-data-types)
  - [Supported Data Types (dtypes)](#supported-data-types-dtypes)
  - [Creation Functions](#creation-functions)
- [3. Data Transfer: GPU <=> CPU](#3-data-transfer-gpu--cpu)
- [4. Tensor Operations](#4-tensor-operations)
  - [Element-wise Arithmetic](#element-wise-arithmetic)
  - [Mathematical Functions](#mathematical-functions)
  - [Comparison Operators](#comparison-operators)
- [5. Linear Algebra with cuBLAS](#5-linear-algebra-with-cublas)
- [6. Reduction Operations](#6-reduction-operations)
- [7. Shape Manipulation and Views](#7-shape-manipulation-and-views)
- [8. Joining Tensors](#8-joining-tensors)
- [9. Advanced Utilities](#9-advanced-utilities)
  - [Reproducibility](#reproducibility)
  - [Synchronization](#synchronization)
  - [Performance Benchmarking](#performance-benchmarking)
- [Full Example Workflow](#full-example-workflow)

---

## Installation

```R
# Coming soon to CRAN
# For now, install from source or using devtools:
# devtools::install_github("path/to/acediaR")
```

## Core Concept: The `gpuTensor`

The `gpuTensor` is the central object in `acediaR`. It is a multi-dimensional array (tensor) that resides in the GPU's high-speed memory. All operations on `gpuTensor` objects are performed by custom CUDA kernels, bypassing the CPU to minimize data transfer overhead and maximize performance.

## 1. Getting Started: Checking GPU Status

Before you begin, load the library and check the status of your available NVIDIA GPUs.

```R
library(acediaR)

# Display information about available GPUs, CUDA driver, and runtime
gpu_status()
```

## 2. Tensor Creation and Data Types

You can create tensors by moving existing R data to the GPU or by generating them directly on the device.

### Supported Data Types (dtypes)

`acediaR` supports the following data types. Specifying the correct `dtype` is crucial for performance and memory management.

| `dtype` String | C++/CUDA Type         | Description                       |
| :------------- | :-------------------- | :-------------------------------- |
| `"float64"`    | `double`              | Double-precision floating point   |
| `"float32"`    | `float`               | Single-precision floating point   |
| `"int32"`      | `int`                 | 32-bit signed integer             |
| `"bool"`       | `bool`                | Boolean type (for logical masks)  |


### Creation Functions

```R
# 1. From an existing R object (vector, matrix, array)
r_matrix <- matrix(1:12, nrow = 3, ncol = 4)
gpu_matrix <- as_tensor(r_matrix, dtype = "float32")

# 2. Directly on the GPU (more efficient)
# Create a 4x4 tensor with random uniform values between 0 and 1
rand_t <- rand_tensor(c(4, 4), dtype = "float32")

# Create a 1000-element tensor with normally distributed values
norm_t <- rand_tensor_normal(1000, mean = 0, sd = 1, dtype = "float64")

# Create an uninitialized (empty) tensor
empty_t <- empty_tensor(c(10, 5), dtype = "int32")

# Create a tensor of ones with the same shape as another tensor
ones_t <- create_ones_like(rand_t)
```

**Key Creation Functions:**

| Function                 | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| `as_tensor()`            | Converts an R object to a `gpuTensor`.                |
| `gpu_tensor()`           | Low-level `gpuTensor` constructor.                    |
| `empty_tensor()`         | Creates an uninitialized tensor.                      |
| `rand_tensor()`          | Creates a tensor with uniformly distributed values.   |
| `rand_tensor_normal()`   | Creates a tensor with normally distributed values.    |
| `create_ones_like()`     | Creates a tensor of ones with the shape of an input.  |

## 3. Data Transfer: GPU <=> CPU

To work with your results in R, you must explicitly transfer data from the GPU back to the CPU's main memory.

```R
# Transfer a gpuTensor back to an R array
cpu_array <- as.array(gpu_matrix)

# For a tensor with a single element, retrieve it as an R scalar
scalar_tensor <- tensor_sum(as_tensor(c(10, 20))) # A gpuTensor containing 30
scalar_value <- tensor_to_scalar(scalar_tensor)   # R numeric: 30
```

## 4. Tensor Operations

`acediaR` overloads many standard R operators and functions for seamless use with `gpuTensor` objects.

### Element-wise Arithmetic
Operations are applied to each element in parallel on the GPU.

```R
a <- as_tensor(c(1, 2, 3), dtype = "float32")
b <- as_tensor(c(4, 5, 6), dtype = "float32")

# Arithmetic operations
c_add <- a + b  # Result: [5, 7, 9]
c_mul <- a * b  # Result: [4, 10, 18]
c_pow <- a ^ 2  # Result: [1, 4, 9]
```
**Operators**: `+`, `-`, `*`, `/`, `^` (`pow`).

### Mathematical Functions
A rich set of mathematical functions are available.

```R
t <- as_tensor(c(1, 4, 9), dtype = "float32")

log_t <- log(t)
sqrt_t <- sqrt(t)

# Activation functions
relu_t <- relu(t)
sigmoid_t <- sigmoid(t)
```
**Functions**: `log()`, `exp()`, `sqrt()`, `sin()`, `cos()`, `tanh()`, `relu()`, `sigmoid()`, `round()`, `floor()`, `ceiling()`, `abs()`.

### Comparison Operators
These return a logical `gpuTensor` (containing 0s and 1s).

```R
t1 <- as_tensor(c(1, 5, 2))
t2 <- as_tensor(c(1, 3, 4))

t1 == t2  # Returns gpuTensor: [1, 0, 0]
t1 > t2   # Returns gpuTensor: [0, 1, 0]
```
**Operators**: `==`, `!=`, `>`, `<`, `>=`, `<=`.

## 5. Linear Algebra with cuBLAS

Leverage NVIDIA's highly optimized cuBLAS library for linear algebra.

```R
# Create two matrices on the GPU
mat_A <- rand_tensor(c(256, 512), dtype = "float32")
mat_B <- rand_tensor(c(512, 128), dtype = "float32")

# Matrix-matrix multiplication
mat_C <- matmul(mat_A, mat_B)

# Other key operations
vec <- rand_tensor(512)
mat_vec_prod <- matvec(mat_A, vec)      # Matrix-vector product
outer_p <- outer_product(vec, vec)      # Outer product
```

## 6. Reduction Operations

Aggregate tensor values across the entire tensor or along a specific axis.

```R
t <- as_tensor(matrix(1:6, nrow = 2, ncol = 3))
# 1 3 5
# 2 4 6

# Reduce the entire tensor
tensor_sum(t)   # Result: 21
tensor_mean(t)  # Result: 3.5
tensor_max(t)   # Result: 6

# Axis-aware reductions (sum along columns, axis=1)
col_sums <- tensor_sum(t, axis = 1) # Result: [3, 7, 11]

# Find the index of the minimum/maximum value
argmin(t) # Returns 0 (index of '1')
argmax(t) # Returns 5 (index of '6')
```
**Functions**: `tensor_sum()`, `tensor_prod()`, `tensor_mean()`, `tensor_var()`, `tensor_min()`, `tensor_max()`, `all()`, `any()`, `argmin()`, `argmax()`.

## 7. Shape Manipulation and Views

`acediaR` offers memory-efficient tools for reshaping, transposing, and slicing tensors. **Views** are particularly powerful, as they create a new tensor header for the same underlying data, avoiding costly memory copies.

```R
t <- rand_tensor(c(2, 3, 4)) # Shape: 2x3x4

# Reshape (data layout remains the same)
reshaped_t <- reshape(t, c(6, 4)) # Shape: 6x4

# Permute dimensions (reorders data access logic)
permuted_t <- permute(t, c(3, 1, 2)) # New shape: 4x2x3

# Create a view of a slice (no data copy)
# The exact S3 dispatch for `[` is defined
sub_view <- t[1, , ] # Selects the first slice, shape: 3x4

# Check if two tensors share the same memory block
shares_memory(t, sub_view) # Returns TRUE
shares_memory(t, reshaped_t) # Returns TRUE
shares_memory(t, permute(t, c(1,2,3))) # Also true if not permuted

# Use contiguous() to create a dense copy if needed for performance
contiguous_t <- contiguous(permuted_t)
```

## 8. Joining Tensors

Combine multiple tensors into one.

```R
t1 <- as_tensor(c(1, 2))
t2 <- as_tensor(c(3, 4))

# Concatenate along an existing axis (axis=1)
concatenated <- concat_tensor(list(t1, t2), axis = 1) # Shape: [4], Data: [1,2,3,4]

# Stack along a new axis (axis=1)
stacked <- stack_tensor(list(t1, t2), axis = 1) # Shape: [2, 2]
```

## 9. Advanced Utilities

### Reproducibility
Set the seed for the GPU's random number generator to ensure your results are deterministic.
```R
set_random_seed(42)
t1 <- rand_tensor(c(2, 2))

set_random_seed(42)
t2 <- rand_tensor(c(2, 2))
# t1 and t2 are identical
```

### Synchronization
GPU computations are asynchronous. To accurately measure performance, you must explicitly wait for all pending GPU work to finish.
```R
start_time <- Sys.time()
result <- matmul(mat_A, mat_B)
synchronize() # Block R until the matmul is complete
end_time <- Sys.time()
print(end_time - start_time)
```

### Performance Benchmarking
`acediaR` provides tools to help you understand the performance trade-offs between CPU and GPU execution.
```R
# Find the data size where the GPU becomes faster for a given operation
threshold <- benchmark_gpu_threshold(
  op = "add",
  min_size = 1e3,
  max_size = 1e7,
  dtype = "float32"
)

# Plot the crossover point
plot_gpu_threshold(threshold)
```

## Full Example Workflow

Here is a script demonstrating a typical `acediaR` workflow.

```R
# 1. Setup
library(acediaR)
set_random_seed(123)

# 2. Create tensors on the GPU
mat_a <- rand_tensor(c(1024, 512), dtype = "float32")
mat_b <- rand_tensor(c(512, 2048), dtype = "float32")

# 3. Perform linear algebra
# Time the operation accurately with synchronize()
start_time <- Sys.time()
mat_c <- matmul(mat_a, mat_b)
synchronize()
end_time <- Sys.time()

cat("Matrix multiplication took:", end_time - start_time, "seconds\n")

# 4. Perform element-wise and reduction operations
mat_c_activated <- sigmoid(mat_c)
col_means <- tensor_mean(mat_c_activated, axis = 1)

# 5. Inspect results
print("Shape of result:")
print(dim(mat_c_activated))

print("Shape of column means:")
print(dim(col_means))

# 6. Transfer a small part of the result back to R for validation/plotting
result_subset <- as.array(col_means[1:10])
print("First 10 column means:")
print(result_subset)
``` 