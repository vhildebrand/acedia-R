# acediaR – Phase 6 Execution Plan  
## Advanced Array / Matrix Functionality (4–6 weeks)

### Vision  
Deliver full statistical, FFT and advanced linear-algebra capabilities so that acediaR reaches feature-parity with base-R for dense arrays and prepares the ground for sparse & autograd work.

---

## A. Linear-Algebra Completion (6.1)

| Week | Item | Key Work | Status |
|------|------|----------|---------|
| 1 | **Infrastructure** | • Link `-lcusolver`, `-lcublasLt` in `src/Makevars`  <br/>• Add `cusolver_utils.h` with lazy handle getter | ✅ **COMPLETE** |
| 1–2 | **LU &\#8594; solve / det** | • Wrap `cusolverDnXgetrf/getrs`  <br/>• Expose `lu_decompose()`, `solve()`, `det()` R wrappers | ✅ **COMPLETE** |
| 2 | **QR** | • `cusolverDnXgeqrf` + `orgqr/ormqr`  <br/>• R method `qr.gpuTensor` | ✅ **COMPLETE** |
| 2–3 | **Cholesky** | • `cusolverDnXpotrf`  <br/>• `chol.gpuTensor` | ✅ **COMPLETE** |
| 3 | **Symmetric eigen** | • `cusolverDnXsyevd`  <br/>• `eigen.gpuTensor` (values + vectors) | ✅ **COMPLETE** |
| 3 | **Batched GEMM** | • Expose `batched_matmul()` for 3-D tensors | 🔄 **PENDING** |

### Testing & Docs
• Compare against base-R (`solve`, `det`, `chol`, `qr`, `eigen`) on small matrices.  
• Add benchmarks to `inst/benchmarks/`.

---

## B. Statistics Kernels (6.2)

| Week | Item | Key Work |
|------|------|----------|
| 3 | **Elementary** | • `sd` (sqrt(var)), `mad`, `range` using existing reductions |
| 3–4 | **Order statistics** | • `quantile`, `median` via GPU radix / thrust sort |
| 4 | **Counting** | • `tabulate`, `hist` using histogram kernel with atomics |

### Testing & Docs
• Parity with base-R for all edge cases (NA/NaN, ties).  
• Pathological large-N performance tests.

---

## C. FFT Support (6.3)

| Week | Item | Key Work |
|------|------|----------|
| 4 | **Build** | • Link `-lcufft`, create `TensorFFT.cpp` |
| 4–5 | **Wrappers** | • 1-D & ND real/complex transforms via `cufftPlanMany` |
| 5 | **R interface** | • `fft.gpuTensor`, `ifft`, `Re`, `Im`, `Mod` |

### Testing & Docs
• Validate against base-R `fft` (tolerance ≈ 1e-6).  
• Signal-processing vignette snippet.

---

## D. Broadcast & Type Promotion Polish (6.4)

| Week | Item | Key Work |
|------|------|----------|
| 1 | **Rules audit** | • Ensure scalar recycling on any axis matches base-R semantics |
| 1 | **Mixed dtypes** | • Verify promotion table (int→float, fp16↔fp32, etc.) |
| 1–2 | **Tests** | • Exotic shapes: `(3,1,5)+(1,4,1)` etc. |

---

## E. Sparse Tensor MVP (Stretch)

If time allows (weeks 5–6):
1. Implement CSR/COO classes.  
2. Bind cuSPARSE for dense×sparse `matmul`.  
3. Provide `sparse_tensor()` constructor and basic ops.

---

## Timeline Summary

| Week | Focus |
|------|-------|
| 1 | cuSOLVER linking, LU/solve, broadcast rules |
| 2 | QR, Cholesky, det; stats groundwork |
| 3 | Eigen, batched GEMM; sd/mad/range |
| 4 | Quantile/median, tabulate/hist; FFT infrastructure |
| 5 | FFT R wrappers, validation, benchmarks |
| 6 | Buffer for sparse MVP, documentation, CRAN-check pass |

---

## Deliverables
* New C++ files: `TensorLinearAlgebra.cpp` extensions, `TensorStats.cpp`, `TensorFFT.cpp`, `cusolver_utils.h`.  
* New R methods: `chol.gpuTensor`, `qr.gpuTensor`, `fft.gpuTensor`, etc.  
* 100 % unit-test coverage for added features.  
* Benchmarks and updated vignettes.  
* Roadmap update and green CI. 