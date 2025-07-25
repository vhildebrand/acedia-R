# AcediaR - Refactoring & Implementation Plan

This document outlines a comprehensive, phased plan to evolve `acediaR` into a feature-complete and robust GPU tensor library for R. The plan addresses every known issue and gap identified in the initial static analysis reports: `CUDA_Library_Status.md`, `CUDA_Kernel_Coverage.md`, `R_Library_Status.md`, and `R_API_Coverage.md`.

---

## Phase 1: Code Hygiene & Foundational Fixes (Low Effort, High Impact)
*Goal: Clean the repository, remove dead code, and fix the most straightforward gaps in the R API.*

#### 1.1. Git Repository Cleanup
- **Action:** Add all build artifacts (`*.o`, `*.so`) and the orphaned `tensor_ops_templated_DEPRECATED.o` to `.gitignore`.
- **Action:** Remove existing build artifacts from git tracking using `git rm --cached <file>...`.
- **Files Affected:** `.gitignore`, `src/*.o`, `src/acediaR.so`, `src/tensor_ops_templated_DEPRECATED.o`.

#### 1.2. Remove Deprecated & Unused Source Files
- **Action:** Delete `src/tensor_ops_templated_DEPRECATED.cu`. Its logic is now in the modular kernel files.
- **Action:** Delete `src/kernels/tensor_wrappers.cu.disabled`. Its functionality is now handled by `src/Tensor*.cpp` binding files.
- **Action:** Move `R/benchmark-multiply.R` to a new `inst/benchmarks/` directory to exclude it from the installable package.
- **Files Affected:** `src/tensor_ops_templated_DEPRECATED.cu`, `src/kernels/tensor_wrappers.cu.disabled`, `R/benchmark-multiply.R`.

#### 1.3. Implement Missing R S3 Operators
- **Action:** In `R/gpuTensor.R`, create the `-.gpuTensor` S3 method for binary subtraction, which will call the existing `tensor_sub_unified`.
- **Action:** Create the `/.gpuTensor` S3 method for division, calling `tensor_div_unified`.
- **Action:** Implement the unary negation operator (`-x`) by creating a `-.gpuTensor` method that calls `tensor_scalar_mul_unified(x, -1.0)`.
- **Files Affected:** `R/gpuTensor.R`.

---

## Phase 2: Bridging the Gap (Exposing C++ to R)
*Goal: Make all existing C++ backend functionality fully available and robust at the R level.*

#### 2.1. Expose Full Reduction Capabilities
- **Action:** Modify the C++ reduction wrappers (`tensor_sum_unified`, etc.) to accept an `axis` argument.
- **Action:** Implement axis-aware reduction logic in `src/TensorReduction.cpp`. This requires replacing the current host-side final reduction with a full device-side parallel reduction.
- **Action:** Update the R S3 methods (`sum.gpuTensor`, `mean.gpuTensor`, etc.) to accept `axis` and `keep.dims` arguments.
- **Files Affected:** `src/TensorReduction.cpp`, `src/kernels/tensor_kernels.cu`, `R/gpuTensor.R`.

#### 2.2. Implement R Comparison Operators
- **Action:** In `R/gpuTensor.R`, create S3 methods for `>.gpuTensor`, `<.gpuTensor`, and `==.gpuTensor` that call the existing `tensor_gt_unified`, `tensor_lt_unified`, and `tensor_eq_unified` functions.
- **Files Affected:** `R/gpuTensor.R`.

#### 2.3. Expose `argmax` and Implement `argmin`
- **Action:** Create a C++ wrapper for `tensor_argmin_unified` in `TensorReduction.cpp`.
- **Action:** Implement the `argmin` kernel by modifying the generic `reduction_kernel` to track indices alongside values.
- **Action:** Create user-facing R functions `argmax(x, axis=NULL)` and `argmin(x, axis=NULL)` that call the backends.
- **Files Affected:** `R/gpuTensor.R`, `src/TensorReduction.cpp`, `src/kernels/tensor_kernels.cu`.

#### 2.4. Expose Advanced Transforms & Document
- **Action:** Create user-friendly R functions `pad_tensor()`, `repeat_tensor()`, etc., in `R/gpuTensor.R` with full `roxygen2` documentation, wrapping the existing C++ calls.
- **Files Affected:** `R/gpuTensor.R`.

---

## Phase 3: Core CUDA Kernel Expansion
*Goal: Fill all identified gaps in the low-level CUDA kernels.*

#### 3.1. Implement Missing Unary Kernels
- **Action:** In `src/kernels/tensor_ops.cu`, create new device functors: `PowOp` (for `pow(x, c)`), `FloorOp`, `CeilOp`, `RoundOp`, `ErfOp`.
- **Action:** Add corresponding C++ wrappers in `src/TensorMath.cpp` and expose as S3 methods in R (`^.gpuTensor`, `floor.gpuTensor`, etc.).
- **Files Affected:** `src/kernels/tensor_ops.cu`, `src/TensorMath.cpp`, `R/gpuTensor.R`.

#### 3.2. Implement Missing Binary & Strided Kernels
- **Action:** In `src/kernels/tensor_ops.cu`, create element-wise `MaxOp`, `MinOp`, and a binary `PowOp`.
- **Action:** Complete the partial implementations for `sub`, `mul`, and `div` to support strided operations on `fp16` and integer types.
- **Action:** Add broadcast support for `sub` and `div` in `TensorArithmetic.cpp`.
- **Action:** Add scalar `sub` and `div` variants.
- **Files Affected:** `src/kernels/tensor_ops.cu`, `src/TensorArithmetic.cpp`.

#### 3.3. Complete Slice/Mutation Functionality
- **Action:** Enhance `src/kernels/tensor_slice_update.cu` to support general slice assignment (`set_slice_kernel`).
- **Action:** Implement a kernel and C++ bindings for boolean mask assignment (`x[mask] <- value`).
- **Files Affected:** `src/kernels/tensor_slice_update.cu`, `src/TensorMutation.cpp`, `R/gpuTensor.R`.

---

## Phase 4: Advanced Features & Final Polish
*Goal: Implement major new features and address remaining architectural issues.*

#### 4.1. Integrate cuBLAS for High-Performance GEMM
- **Action:** In `src/TensorLinearAlgebra.cpp`, modify `tensor_matmul_unified` to delegate to `cublasSgemm`/`cublasDgemm` for 2D contiguous tensors.
- **Action:** Keep the existing custom `matmul_tiled_kernel` as a fallback.
- **Action:** Add the necessary cuBLAS library flags to `src/Makevars`.
- **Files Affected:** `src/TensorLinearAlgebra.cpp`, `src/cuda_utils.h` (for cuBLAS handle), `src/Makevars`.

#### 4.2. Address Autograd Placeholders
- **Action:** Remove the autograd placeholders to provide a stable, non-autograd library first.
- **Action:** Remove the `requires_grad` argument from all R function signatures.
- **Action:** Delete `src/AutogradOperations.h`, `src/ComputationGraph.h`, and `src/GraphCompiler.h`.
- **Files Affected:** `R/gpuTensor.R`, `src/AutogradOperations.h`, `src/ComputationGraph.h`, `src/GraphCompiler.h`.

#### 4.3. Final Documentation & Testing
- **Action:** Update all four analysis documents to reflect that all planned work is complete.
- **Action:** Add comprehensive `testthat` tests for every new feature and bug fix.
- **Action:** Write a new vignette in `vignettes/` demonstrating the complete feature set of the library.

---
_This plan provides a clear roadmap to a production-ready `acediaR` 1.0._ 