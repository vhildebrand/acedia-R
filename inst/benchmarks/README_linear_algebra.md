# Linear Algebra Validation and Benchmarking Scripts

This directory contains comprehensive validation and benchmarking scripts for acediaR's linear algebra functionality.

## Scripts Overview

### 1. `linear_algebra_demo.R`
**Purpose**: Demonstrates the linear algebra functionality and shows the side-by-side comparison framework.

**Features**:
- ✅ Function availability checking
- ✅ S3 method registration verification  
- ✅ Base R linear algebra demonstrations
- ✅ GPU test framework (ready for when registration is fixed)
- ✅ Mathematical identity verification

**Usage**:
```bash
Rscript inst/benchmarks/linear_algebra_demo.R
```

**What it shows**:
- All S3 methods are properly registered
- Base R functionality works correctly
- Test framework is ready for GPU validation
- Mathematical identities are verified (Q*R=A, L*L^T=A, A*v=λ*v)

### 2. `linear_algebra_validation.R`
**Purpose**: Comprehensive side-by-side validation of GPU vs CPU implementations.

**Features**:
- ✅ Multiple test matrices (symmetric, positive definite, rectangular)
- ✅ Direct result comparison with tolerance checking
- ✅ Mathematical identity verification
- ✅ Error handling and reporting
- ✅ Detailed pass/fail reporting

**Test Coverage**:
- **Determinant**: GPU vs base R comparison
- **Linear solve**: System solving verification
- **LU decomposition**: Factorization and reconstruction
- **QR decomposition**: Orthogonal and triangular factor comparison
- **Cholesky decomposition**: Positive definite factorization
- **Eigenvalue decomposition**: Symmetric eigenvalue/eigenvector computation

**Usage**:
```bash
Rscript inst/benchmarks/linear_algebra_validation.R
```

### 3. `linear_algebra_benchmark.R`
**Purpose**: Performance benchmarking across different matrix sizes.

**Features**:
- ✅ Multi-size testing (100×100, 500×500, 1000×1000, 2000×2000)
- ✅ Multiple runs for statistical accuracy
- ✅ Automatic speedup calculation
- ✅ Summary statistics and rankings
- ✅ Expected performance predictions

**Metrics Tracked**:
- Execution time (mean ± std dev)
- GPU vs CPU speedup ratios
- Performance scaling with matrix size
- Operation-specific performance characteristics

**Usage**:
```bash
Rscript inst/benchmarks/linear_algebra_benchmark.R
```

## Linear Algebra Functions Implemented

### Core Functions
| Function | Description | cuSOLVER Backend | Status |
|----------|-------------|------------------|---------|
| `det.gpuTensor()` | Matrix determinant | `cusolverDnXgetrf` | ✅ Complete |
| `solve.gpuTensor()` | Linear system solving | `cusolverDnXgetrf` + `cusolverDnXgetrs` | ✅ Complete |
| `lu_decompose()` | LU factorization | `cusolverDnXgetrf` | ✅ Complete |
| `qr.gpuTensor()` | QR decomposition | `cusolverDnXgeqrf` + `cusolverDnXorgqr` | ✅ Complete |
| `chol.gpuTensor()` | Cholesky decomposition | `cusolverDnXpotrf` | ✅ Complete |
| `eigen.gpuTensor()` | Symmetric eigendecomposition | `cusolverDnXsyevd` | ✅ Complete |

### Technical Details

#### Data Type Support
- **float32** (`FLOAT32`): All functions support single precision
- **float64** (`FLOAT64`): All functions support double precision
- **Type Safety**: Template-based implementation with compile-time type checking

#### Error Handling
- **cuSOLVER Errors**: Comprehensive error checking with descriptive messages
- **Mathematical Errors**: Specific handling for singular matrices, non-positive definite matrices
- **Memory Management**: Automatic cleanup of GPU workspace memory

#### Memory Efficiency
- **Workspace Queries**: Dynamic workspace allocation based on cuSOLVER requirements
- **Contiguous Copies**: Automatic handling of non-contiguous input tensors
- **In-place Operations**: Where possible, operations modify input tensors to save memory

## Expected Performance Characteristics

### Matrix Size Scaling
```
Matrix Size    Expected GPU Speedup
100×100        2-5x    (GPU overhead dominates)
500×500        5-15x   (Sweet spot begins)
1000×1000      10-30x  (GPU advantages clear)
2000×2000      15-50x  (Large matrix benefits)
```

### Operation-Specific Performance
```
Operation      Expected GPU Speedup    Notes
det()          10-30x                  cuSOLVER LU very fast
solve()        8-25x                   LU + substitution
qr()           5-20x                   cuSOLVER QR optimized
chol()         12-35x                  Simple factorization
eigen()        3-15x                   Iterative, varies by convergence
```

## Current Status

### ❌ **CRITICAL ISSUES - SYSTEM NON-FUNCTIONAL**
- **RcppExports Registration BROKEN**: C++ functions completely inaccessible from R
- **GPU Tensor Creation FAILS**: Cannot create any GPU tensors
- **All GPU Linear Algebra Functions UNAVAILABLE**: No functions can be called
- **Validation and Benchmarking IMPOSSIBLE**: No GPU operations work

### 🔧 **What Actually Works**
- C++ code compiles without errors
- R S3 methods are registered in dispatch tables  
- Base R linear algebra functions work correctly
- Test framework structure is sound

### 🚨 **What Is Broken**
- `create_tensor_unified()` function not found
- All `tensor_*_unified()` C++ functions inaccessible
- GPU tensor creation completely non-functional
- All GPU linear algebra operations fail immediately

### ⚠️ **NOT READY FOR ANY TESTING**
The system is currently in a completely non-functional state:
1. **No GPU functionality works at all**
2. **Cannot create GPU tensors**
3. **Cannot perform any GPU operations**
4. **Scripts demonstrate failure, not success**

## Usage Examples

### Basic Linear Algebra Operations
```r
library(acediaR)

# Create test matrix
A <- matrix(c(4, 1, 2, 1, 3, 1, 2, 1, 5), 3, 3)
b <- c(1, 2, 3)

# Once GPU tensors work:
# A_gpu <- gpu_tensor(A)
# b_gpu <- gpu_tensor(b)

# Linear algebra operations:
# det_result <- det(A_gpu)           # Determinant
# solve_result <- solve(A_gpu, b_gpu) # Linear solve  
# qr_result <- qr(A_gpu)             # QR decomposition
# chol_result <- chol(A_gpu)         # Cholesky decomposition
# eigen_result <- eigen(A_gpu)       # Eigendecomposition
```

### Performance Comparison
```r
# Benchmark determinant computation
A_large <- matrix(rnorm(1000000), 1000, 1000)
A_large <- A_large %*% t(A_large) + diag(1000) * 0.1

# CPU timing
cpu_time <- system.time(det(A_large))

# GPU timing (when available)
# A_gpu <- gpu_tensor(A_large)  
# gpu_time <- system.time(det(A_gpu))
# speedup <- cpu_time[3] / gpu_time[3]
```

## Mathematical Verification

The validation scripts verify these mathematical identities:

1. **QR Decomposition**: `Q %*% R == A` (within numerical tolerance)
2. **Cholesky Decomposition**: `L %*% t(L) == A` (within numerical tolerance)  
3. **Eigenvalue Equation**: `A %*% v == λ * v` for each eigenvalue/eigenvector pair
4. **LU Solve Verification**: `A %*% x == b` after solving `A*x = b`
5. **Orthogonality**: `t(Q) %*% Q == I` for QR decomposition

## Contributing

When adding new linear algebra functions:

1. **Add C++ Implementation**: Use cuSOLVER backend in `src/TensorLinearAlgebra.cpp`
2. **Add R Wrapper**: Create S3 method in `R/gpuTensor.R`
3. **Update NAMESPACE**: Export the new S3 method
4. **Add Tests**: Include validation in the benchmark scripts
5. **Update Documentation**: Add to this README and roxygen docs

## References

- [cuSOLVER Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [LAPACK Reference](https://www.netlib.org/lapack/)
- [Matrix Computations (Golub & Van Loan)](https://www.amazon.com/Matrix-Computations-Gene-Golub/dp/1421407949) 