# Test Consolidation Plan

## Current State (2,686 total lines)
- **Root directory:** 5 informal test scripts (716 lines)
- **tests/testthat/:** 7 formal test files (1,970 lines)
- **Issues:** Redundancy, inconsistent structure, maintenance burden

## Proposed Structure (Reduced to ~1,500 lines)

### Core Test Files (tests/testthat/)

1. **test-core-operations.R** (~300 lines)
   - Basic tensor creation, arithmetic (+, -, *, /)  
   - Scalar operations
   - Data type validation
   - *Consolidates:* test-dtype-validation.R + parts of comprehensive

2. **test-linear-algebra.R** (~250 lines)
   - Matrix multiplication, outer product
   - Matrix-vector, vector-matrix operations
   - Transpose, permute operations
   - *Consolidates:* test_new_ops.R + linear algebra from comprehensive

3. **test-memory-views.R** (~200 lines)
   - Contiguous/non-contiguous handling
   - Views, slicing, broadcasting
   - Memory efficiency verification
   - *Consolidates:* All 4 non-contiguous test files + test-views-broadcasting.R

4. **test-advanced-ops.R** (~200 lines)
   - Reductions (sum, mean, max, min, var, prod)
   - Activations (relu, sigmoid, softmax)
   - Comparisons, concatenation, stacking
   - *Consolidates:* test-new-gpu-ops.R + advanced ops from comprehensive

5. **test-performance.R** (~300 lines)
   - GPU vs CPU benchmarks
   - Memory usage verification
   - Parallel execution verification
   - *Consolidates:* Both performance test files + execution verification

6. **test-integration.R** (~250 lines)
   - End-to-end workflows
   - Mixed operations
   - Error handling and fallbacks
   - Edge cases and regression tests

### Removed Files
- All 5 root directory test files → integrated into formal structure
- test-tensor-comprehensive.R → split across specialized files
- Redundant performance files → merged into single performance suite

## Benefits
- **50% reduction** in total test code
- **Elimination** of redundancy
- **Consistent** testthat structure
- **Better organization** by functionality
- **Easier maintenance** and debugging

## Migration Steps
1. Create new consolidated test files
2. Migrate and deduplicate test logic
3. Verify coverage is maintained
4. Remove old files
5. Update CI/documentation 