# Tensor Operations - Modular CUDA Kernels

This directory contains the modular CUDA kernel implementations for the acediaR tensor library.

## File Organization

```
src/kernels/
├── kernel_utils.cuh       # Type conversion utilities and common device functions
├── tensor_kernels.cu      # Core CUDA kernel implementations
├── tensor_ops.cu          # Operation functors & launch helper functions
└── tensor_wrappers.cu     # C-style wrapper functions for R interface
```

## Compilation

The build system compiles only `tensor_wrappers.cu`, which automatically includes the other components via the include chain:

```
tensor_wrappers.cu 
  ↳ includes tensor_ops.cu
     ↳ includes tensor_kernels.cu
        ↳ includes kernel_utils.cuh
```

## Architecture

### 1. kernel_utils.cuh
- Type conversion templates (`convert_type<To, From>`)
- Half precision and BFloat16 specializations
- Common device-side utilities

### 2. tensor_kernels.cu
- Raw CUDA kernel implementations
- Generic templated kernels (elementwise, reduction, matmul, broadcast, etc.)
- Device-side code only

### 3. tensor_ops.cu
- Operation functors (AddOp, MulOp, ExpOp, etc.)
- Host-side launch helper functions
- Template instantiation for common types

### 4. tensor_wrappers.cu
- C-style extern "C" wrapper functions
- Direct interface to R/Rcpp
- Type-specific function declarations

## Adding New Operations

To add a new tensor operation (e.g., `tanh`):

1. **Add functor** in `tensor_ops.cu`:
   ```cpp
   struct TanhOp {
       template<typename T>
       __device__ T operator()(const T& a) const { return tanh(a); }
   };
   ```

2. **Add C wrapper** in `tensor_wrappers.cu`:
   ```cpp
   void tensor_tanh_float32(float* result, const float* input, size_t n) {
       launch_elementwise_unary(result, input, n, TanhOp{});
       cudaDeviceSynchronize();
   }
   ```

3. **Update R interface** in `../UnifiedTensorInterface.cpp`

## Benefits

- **Maintainability**: Easy to find and modify specific functionality
- **Compilation Speed**: Only changed modules need recompilation
- **Parallel Development**: Multiple developers can work on different operation types
- **Industry Standard**: Follows common CUDA project patterns
- **Extensibility**: Simple pattern for adding new operations 