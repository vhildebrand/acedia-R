/**
 * UnifiedTensorInterface.cpp
 * 
 * This file has been refactored for better maintainability. The monolithic implementation
 * has been split into modular components:
 * 
 * - TensorCreation.cpp     : Tensor creation and factory functions
 * - TensorDataAccess.cpp   : Data conversion, host-device transfer, metadata access
 * - TensorArithmetic.cpp   : Binary arithmetic operations (+, -, *, /, scalar ops)
 * - TensorLinearAlgebra.cpp: Matrix operations (matmul, etc.)
 * - TensorShape.cpp        : Shape operations (view, reshape, transpose, permute)
 * - TensorMath.cpp         : Unary math operations (exp, log, sqrt)
 * - TensorReduction.cpp    : Reduction operations (sum, mean, max, min)
 * 
 * This file now contains only common headers and utility functions shared across modules.
 */

#include <Rcpp.h>
#include "gpuTensor.h"
#include "TensorRegistry.h"
#include "cuda_utils.h"
#include <memory>

using namespace Rcpp;

// This file now serves as a header for the modular tensor interface.
// All functionality has been moved to specialized modules.
