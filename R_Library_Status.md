# R Library Status Report (acediaR)

_Last updated: 2025-07-25_

## 1. Directory‐level Overview

| Path | Role | Key Contents |
|------|------|--------------|
| `R/` | High-level user-facing API written in R | `gpuTensor.R`, utility helpers, vectorised convenience wrappers, auto-generated `RcppExports.R` |
| `R/gpuTensor.R` | Core OO wrapper around C++ tensors | S3 methods (`+.gpuTensor`, `*.gpuTensor`, `as.array`, etc.) |
| `R/gpu-utils.R` | Small diagnostics helpers | `gpu_status()`, docstrings only – actual C++ calls in `RcppExports.R` |
| `R/gpu_multiply.R`, `R/gpu_add.R` | Stand-alone vector helpers | Provide GPU-accelerated add/mul with graceful CPU fallback |
| `R/benchmark-multiply.R` | Benchmark script | Non-exported helper to profile GPU vs CPU multiply |
| `R/RcppExports.R` | Auto-generated glue | `.Call()` stubs exposing >60 C++ entry points |
| `R/acediaR-package.R` | Package metadata | `.onLoad()` hook (sets options, etc.) |

---

## 2. Functional Coverage Snapshot

| Category | Implemented R Helpers | Notes / Gaps |
|----------|----------------------|--------------|
| **Tensor creation / I/O** | `gpu_tensor`, `empty_tensor`, `as_tensor`, `as.array`, S3 coercions | OK – but `requires_grad` placeholder only issues warning |
| **Arithmetic (tensor ⟂ tensor)** | S3 operators `+`, `-`, `*`, `/` call unified C++ API | Division wrapper missing (`/.gpuTensor` not defined) |
| **Scalar-tensor ops** | `tensor_scalar_add_unified`, `tensor_scalar_mul_unified` via operators | No standalone R helpers; relies on S3 dispatch |
| **Linear algebra** | `matmul`, `matvec`, `vecmat`, `outer_product` | No batched GEMM; need qr/svd wrappers |
| **Reductions** | `sum.gpuTensor`, `mean`, etc. (implemented inside `gpuTensor.R`) | Internally call `tensor_sum_unified` – works, but no axis-wise reductions |
| **Views / reshaping** | `view`, `reshape`, `transpose`, `permute` wrappers present | Need advanced stride checks & contiguous enforcement |
| **Broadcasting** | Implemented implicitly in C++ kernels; R side just passes tensors | No user-facing helper to verify broadcastability |
| **Vector convenience** | `gpu_add`, `gpu_multiply`, `gpu_scale`, `gpu_dot` | Only add/mul/scale/dot currently; others TBD |
| **Diagnostics** | `gpu_available`, `gpu_info`, `gpu_memory_available`, `gpu_status` | Adequate |

---

## 3. Per-file Status Details

| File | Status | Description |
|------|--------|-------------|
| `gpuTensor.R` | **Stable** | Main high-level tensor OO; ~1.5k LOC; most API surfaces here. Some TODO tags around autograd + broadcasting. |
| `gpu-utils.R` | **Stable** | Thin wrappers + docstrings; delegates to C++ via `RcppExports`. |
| `gpu_multiply.R` | **Stable** | Element-wise multiply helper with fallback. |
| `gpu_add.R` | **Stable** | Mirror of multiply for addition. |
| `benchmark-multiply.R` | **Auxiliary** | Benchmark script; not exported. |
| `RcppExports.R` | **Auto** | 70+ `.Call` bindings; OK but occasionally out-of-sync with C++ (verify during releases). |
| `acediaR-package.R` | **Metadata** | Sets namespace hooks, package options. |

---

## 4. Redundant / Unused Assets

* `benchmark-multiply.R` could move to `inst/benchmark/` to avoid attachment in production.
* Stand-alone `gpu_add` / `gpu_multiply` partially overlap with tensor methods; consider deprecating.
* Some roxygen topics (e.g., `requires_grad`) reference features not yet implemented.

---

## 5. Known Broken / Incomplete Areas

* **Autograd:** R helpers accept `requires_grad` but backend not wired.
* **Operator gaps:** No S3 method for `/` or unary `-` yet; need wrappers.
* **Axis reductions:** `sum`/`mean` only global; `dim` argument not supported.
* **Broadcast validation:** silent recycling may mask shape errors; add explicit check.
* **DType conversions:** R helper `to_numeric` works, but no `to_dtype()` wrapper.

---

## 6. Recommendations

1. Implement remaining S3 operators (`/.gpuTensor`, unary `-`, comparisons) using existing C++ calls.
2. Add axis/keepdims parameters to reduction functions, mapping to C++ stride-aware kernels.
3. Wire up autograd flags – or remove until backend lands to avoid user confusion.
4. Sync `RcppExports.R` generation during CI to avoid outdated bindings.
5. Consolidate vector helpers (`gpu_add`, `gpu_multiply`) into a single templated helper or encourage use of `gpu_tensor` path.

---
Generated via automated scan of `R/` directory. 