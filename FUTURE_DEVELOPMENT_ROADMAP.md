# acediaR – Future Development Roadmap (Phases 5 → 7)

> **Vision**  Extend acediaR from a feature-complete core tensor engine to a *GPU-first replacement for ordinary R numerical computing* – culminating in a differentiable deep-learning-ready stack.

---
## Phase 5  Robustness, Coverage Gaps & Packaging (≈ 2-3 weeks)
| ID | Item | Key Work |
|----|------|-----------|
| 5.1 | **Bug-Hardening** | • Eliminate remaining illegal-memory-access in axis reductions & cuBLAS views  <br/>• Add `CUDA_CHECK` after every kernel / API call  <br/>• GPU-side unit tests covering odd strides, empty dims, NA/NaN, etc. |
| 5.2 | **Base-R “Math” Group** | Implement / verify: `cumprod`, `cumsum`, `diff`, `range`, `any`, `all`, `which.min`, `which.max`.<br/>Use parallel prefix / segmented reductions where possible. |
| 5.3 | **Random Number Utilities** | Bind cuRAND – uniform, normal, Bernoulli, Poisson.  <br/>Expose `rand_tensor()`, `rnorm.gpuTensor`, … |
| 5.4 | **Memory & Perf Tooling** | Real-time GPU memory snapshot, leak checker in CI.  <br/>Benchmarks in `inst/benchmarks/`; target ≥ 10× CPU speedup. |
| 5.5 | **R-Package Polish** | Pass `R CMD check --as-cran` clean.  <br/>Set up CUDA GitHub Actions matrix (Ubuntu GPU, CPU fallbacks on macOS/Windows). |

**Deliverable → v1.0 CRAN release, green test matrix, rock-solid core.**

---
## Phase 6  Advanced Array / Matrix Functionality (≈ 4-6 weeks)
| ID | Item | Key Work |
|----|------|-----------|
| 6.1 | **Linear-Algebra Completion** | cuSOLVER bindings for LU / QR / Cholesky; `solve`, `det`, `chol`, `qr`, `eigen` (symmetric).  <br/>Batched GEMM for 3-D tensors. |
| 6.2 | **Statistics Kernels** | `sd`, `mad`, `quantile`, `median`, `tabulate`, `hist` – parallel radix sort & reductions where needed. |
| 6.3 | **FFT Support** | cuFFT wrapper: `fft`, `ifft`, `Re`, `Im`, `Mod`. |
| 6.4 | **Broadcast & Type Promotion** | Full numpy-style rules, scalar recycling identical to base R. |
| 6.5 | **Sparse Tensor MVP** | CSR / COO storage; `sparse_tensor()`, dense × sparse `matmul`. |

**Deliverable → v1.1 – parity with most of `base`, `stats`, and `Matrix` for dense arrays.**

---
## Phase 7  Autograd, High-Level APIs & Ecosystem (≈ 6-8 weeks)
| ID | Item | Key Work |
|----|------|-----------|
| 7.1 | **Autograd Core** | Dynamic tape + backward kernels for every primitive. |
| 7.2 | **Optimisers** | SGD, Adam, RMSProp, gradient clipping, parameter groups. |
| 7.3 | **Neural-Network Primitives** | Layers: `linear`, `conv2d`, `batch_norm`, activation ops; optional cuDNN wrappers. |
| 7.4 | **Data Pipelining** | GPU-pinned host buffers, async copy, minibatch iterator. |
| 7.5 | **Interoperability** | Zero-copy conversion with `torch` for R; NumPy `.npy/.npz` IO via `RcppCNPy`. |

**Deliverable → v2.0 – differentiable GPU tensors ready for deep-learning prototypes.**

---
### Cross-Cutting Principles
* **Performance first** – target ≥ 80 % of vendor libraries.  
* **Parallelism everywhere** – default stream per R session + optional user streams.  
* **Exact R semantics** – recycling rules, NA/NaN propagation, S3 dispatch.  
* **100 % test coverage** – CPU reference comparison for every op.  
* **Docs & Examples** – every new feature lands with roxygen, vignette section, benchmark snippet.

---
### Immediate Kick-Off Checklist (Week 1)
1. Create GitHub issues for the 8 failing tests (label *Phase 5-Bug*).  
2. Draft cuRAND `rand_tensor()` API; add `-lcurand` to `Makevars`.  
3. Audit `perform_axis_reduction` & view GEMM path for out-of-bounds accesses.  
4. Configure CUDA GitHub Actions workflow.  
5. Publish roadmap in `FUTURE_DEVELOPMENT_ROADMAP.md` (this file) and link from `README`.  

> *This roadmap is a living document – refine timelines and scope after each milestone review.* 