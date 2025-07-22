// tensor_ops_templated_DEPRECATED.cu
// DEPRECATED: This monolithic file has been split into modular components
// 
// The code from this file has been reorganized into the following structure:
//
// src/kernels/
// ├── kernel_utils.cuh       - Type conversion utilities and common device functions
// ├── tensor_kernels.cu      - Core CUDA kernel implementations  
// ├── tensor_ops.cu          - Operation functors & launch helpers
// └── tensor_wrappers.cu     - C-style wrapper functions for R interface
//
// This file is kept for reference only and should not be compiled.
// To use the new modular structure, compile only tensor_wrappers.cu which 
// includes the other components automatically.
//
// Benefits of the new structure:
// - Better maintainability and organization
// - Faster incremental compilation
// - Easier parallel development
// - Industry-standard CUDA project layout
//
// DO NOT COMPILE THIS FILE - IT IS FOR REFERENCE ONLY 