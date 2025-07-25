T# CUDA Library Status Report (acediaR)

_Last updated: 2025-07-25_

## 1. Directory‐level Overview

| Path | Role | Key Contents |
|------|------|--------------|
| `src/kernels/` | Low-level CUDA kernels & helpers | `tensor_kernels.cu`, `tensor_ops.cu`, `tensor_slice_update.cu`, `kernel_utils.cuh`, `tensor_wrappers.cu.disabled` |
| `src/` (root) | High-level C++/Rcpp bindings and tensor utilities | `Tensor*.cpp`, `gpuTensor.h`, `TensorRegistry.h`, `cuda_utils.h`, `ComputationGraph.h`, etc. |

## 2. Functional Coverage Snapshot

| Category | Implemented | Notes / Gaps |
|----------|-------------|--------------|
| **Element-wise Unary** | `exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `relu`, `sin`, `cos`, `abs`, `square` | `pow`, `floor`, `ceil`, `erf`, `round` _missing_ |
| **Element-wise Binary** | `add`, `sub`, `mul`, `div`, comparisons: `greater`, `less`, `equal` | `pow`, `max`, `min`, `logical_and/or/xor` _missing_ |
| **Scalar Ops** | add & multiply (kernel + strided) | subtract / divide by scalar not exposed yet |
| **Broadcast Ops** | add & multiply (float32/64) | missing other dtypes + ops |
| **Reductions** | Generic templated reduction kernel; wrappers for host-side final reduction | No exported `sum`, `mean`, `max`, `min`, etc., in R interface yet; needs thin wrappers |
| **Linear Algebra** | MatMul (tiled), Outer Product, MatVec, VecMat | No batched matmul, GEMM via cuBLAS, nor higher-level decompositions |
| **Tensor Transform** | Strided copy, Concat, Stack, Repeat, Pad | `permute`, `transpose`, `gather`, `scatter` not in kernels |
| **Indexing / Mutation** | In-place slice add (float/double) | General slice assignment & boolean indexing missing |
| **Type Conversion** | float32↔float16, bfloat16↔float32 (device ≥ 8.0), float64↔float32, int32↔float32 | Others (int16/uint8, bool) missing |
| **Autograd hooks** | Skeleton headers (`AutogradOperations.h`, etc.) present | Back-prop kernels & graph execution not implemented |

## 3. Per-file Status Details

| File | Status | Description |
|------|--------|-------------|
| `kernel_utils.cuh` | **Stable** | Device/host type-conversion helpers incl. fp16 & bf16 specialisations. |
| `tensor_kernels.cu` | **Stable / core** | Houses generic kernels for fill, unary/binary element-wise, reductions, strided copy, concat/stack/repeat/pad, tiled matmul, etc. |
| `tensor_ops.cu` | **Stable** | Functor definitions + launch wrappers for operations listed above. Contains generic reduction driver. |
| `tensor_slice_update.cu` | **Partial** | Only supports _add scalar_ into rectangular slice for float32/64. |
| `tensor_wrappers.cu.disabled` | **Deprecated** | Old C-style wrappers; superseded by modular approach. File is excluded from build. |
| `tensor_ops_templated_DEPRECATED.cu` | **Deprecated** | Monolithic legacy implementation; kept for reference only. |
| `TensorArithmetic.cpp` | **Stable** | Rcpp layer for binary ops (+, −, *, /) incl. broadcast & strides. |
| `TensorMath.cpp` | **Stable** | Rcpp layer for unary math ops (exp, log, sqrt, etc.). |
| `TensorActivation.cpp` | **Stable** | Exposes activations (relu, sigmoid, tanh) to R. |
| `TensorReduction.cpp` | **Partial** | Hosts reduction logic but still relies on CPU fallback for final reduce; R-level `sum`/`mean` wrappers TODO. |
| `TensorLinearAlgebra.cpp` | **Partial** | Wraps matmul/vecmat/outer product; cuBLAS integration planned. |
| `TensorMutation.cpp` | **Partial** | Slice/reshape/contiguous utilities; GPU implementations incomplete for complex cases. |
| `UnifiedTensorInterface.cpp` | **Boilerplate** | Lightweight aggregator; minimal logic. |
| Other `Tensor*.cpp` | **Varies** | Provide high-level API; call through to kernels above. |

## 4. Redundant / Unused Assets

* `tensor_ops_templated_DEPRECATED.cu` – superseded by modular layout.
* `tensor_wrappers.cu.disabled` – wrappers now split across specific .cpp bindings.
* Generated object files (`*.o`, `acediaR.so`) checked-in; should be removed from source control.
* `tensor_ops_templated_DEPRECATED.o` – orphaned artifact.

## 5. Known Broken / Incomplete Areas

* **Reduction wrappers:** No user-facing functions for `tensor_sum`, `tensor_mean`, etc. despite kernel support.
* **Half-precision strided kernels:** Only contiguous variants compiled for fp16; stride-aware versions are declared but not implemented for some ops.
* **LeakyReLU functor:** Defined in `tensor_ops.cu` but never launched from host; missing R binding.
* **Broadcast divide/subtract:** Only add/mul have broadcast helpers; div/sub not yet implemented.
* **Autograd:** Graph structures exist but backward kernels & tape execution are stubbed.
* **DataType coverage:** bf16 conversions compile only when `__CUDA_ARCH__ >= 800`; R-side plumbing absent.

## 6. Recommendations

1. **Expose reduction ops** by adding thin extern "C" wrappers and Rcpp bindings.
2. **Expand unary/binary kernel set** (pow, floor, ceil, logical ops) – reuse existing templates.
3. **Implement broadcast div/sub** & update `TensorArithmetic.cpp` call-sites.
4. **Complete fp16 stride-aware kernels** or document as unsupported.
5. **Remove build artifacts** from repo; add them to `.gitignore`.
6. **Leverage cuBLAS** for matmul > 1024 × 1024 and batched GEMM.
7. **Finish autograd backend** or remove placeholder headers to avoid confusion.

---
Generated by automated codewalk of `src/` contents. 