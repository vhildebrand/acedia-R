# Proposed Clean Architecture for acediaR

## Current Problem
We have two GPU implementations in `gpu_add()` controlled by `use_gpuvector` parameter, which creates maintenance burden and user confusion.

## Proposed Solution: Clear Separation of Concerns

### 1. **Basic GPU Functions** (CPU ↔ GPU)
Simple functions that take R vectors and return R vectors:
```r
# For users who want simple GPU acceleration
result <- gpu_add(r_vector_a, r_vector_b)        # CPU→GPU→CPU
result <- gpu_multiply(r_vector_a, r_vector_b)   # CPU→GPU→CPU  
result <- gpu_dot_product(r_vector_a, r_vector_b) # CPU→GPU→CPU
```

### 2. **Advanced GPU Objects** (GPU-Resident)
For users doing complex GPU workflows where data stays on GPU:
```r
# For users doing GPU-resident computing
gpu_a <- as.gpuVector(r_vector_a)         # CPU→GPU (once)
gpu_b <- as.gpuVector(r_vector_b)         # CPU→GPU (once)

# Chain operations on GPU (no CPU transfers)
gpu_result <- gpu_a + gpu_b               # GPU only
gpu_result2 <- gpu_result * gpu_a         # GPU only
gpu_result3 <- gpu_result2 + gpu_b        # GPU only

final <- as.vector(gpu_result3)           # GPU→CPU (once)
```

### 3. **Benefits of This Approach**

#### Clear Use Cases:
- **Basic functions**: Easy GPU acceleration for simple tasks
- **GPU objects**: Efficient for complex workflows with chained operations

#### No Dual Implementations:
- Each function has ONE implementation
- No confusing `use_gpuvector` parameters
- Cleaner codebase and testing

#### Performance Optimal:
- Basic functions: Optimized for single operations
- GPU objects: Optimized for chained operations

## Implementation Plan

### Phase 1: Remove Dual Implementation
1. Remove `use_gpuvector` parameter from `gpu_add()`
2. Keep only the original C interface implementation for `gpu_add()`
3. Keep separate `gpu_add_vectors()` for gpuVector objects
4. Update documentation to clarify when to use each approach

### Phase 2: Consistent API (Future)
```r
# Basic GPU functions (CPU↔GPU)
gpu_add(a, b)           # Simple vector addition
gpu_multiply(a, b)      # Simple element-wise multiplication
gpu_dot_product(a, b)   # Simple dot product

# GPU object operations (GPU-resident)
gpu_a + gpu_b           # Addition on GPU
gpu_a * gpu_b           # Multiplication on GPU
sum(gpu_a * gpu_b)      # Dot product on GPU
```

### Phase 3: Advanced Features
- Matrix operations following same pattern
- BLAS Level 2 & 3 functions
- Statistical functions

## Migration Strategy

### For Current Sprint 2:
1. **Keep both implementations temporarily** for backward compatibility
2. **Add deprecation warning** to `use_gpuvector` parameter
3. **Update documentation** to recommend proper usage patterns

### For Sprint 3:
1. **Remove dual implementation**
2. **Focus on expanding both paths** with new operations
3. **Establish clear performance benchmarks** for when to use each

This approach provides:
- ✅ Clear mental model for users
- ✅ Single implementation per function
- ✅ Optimal performance for different use cases  
- ✅ Maintainable codebase
- ✅ Consistent API design 