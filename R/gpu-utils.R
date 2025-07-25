#' Check GPU Availability
#'
#' Checks if CUDA-capable GPU is available and accessible.
#'
#' @return Logical indicating if GPU is available for computations
#' 
#' @details
#' This function performs runtime detection of CUDA GPU availability.
#' It checks for CUDA driver installation, device presence, and basic functionality.
#' 
#' @examples
#' \dontrun{
#' if (gpu_available()) {
#'   cat("GPU acceleration enabled\n")
#' } else {
#'   cat("Using CPU-only mode\n")
#' }
#' }
# gpu_available function is auto-generated in RcppExports.R
# No manual definition needed here

#' Get GPU Information
#'
#' Returns detailed information about available CUDA GPUs.
#'
#' @return Character string with GPU device information
#' 
#' @details
#' Provides detailed information about detected CUDA devices including
#' device names, compute capabilities, and memory information.
#' 
#' @examples
#' \dontrun{
#' cat(gpu_info())
#' }
# gpu_info function is auto-generated in RcppExports.R
# No manual definition needed here

#' Get Available GPU Memory
#'
#' Returns the amount of free GPU memory in bytes.
#'
#' @return Numeric value of available GPU memory in bytes (0 if no GPU)
#' 
#' @details
#' Queries the available GPU memory. Useful for determining if large
#' operations will fit in GPU memory.
#' 
#' @examples
#' \dontrun{
#' mem_gb <- gpu_memory_available() / 1e9
#' cat("Available GPU memory:", round(mem_gb, 1), "GB\n")
#' }
# gpu_memory_available function is auto-generated in RcppExports.R
# No manual definition needed here

#' Comprehensive GPU Status
#'
#' Returns comprehensive information about GPU availability and status.
#'
#' @return List containing GPU availability, device information, memory status
#' 
#' @details
#' Provides a complete overview of GPU status for diagnostic purposes.
#' Useful for troubleshooting and determining optimal usage patterns.
#' 
#' @examples
#' \dontrun{
#' status <- gpu_status()
#' print(status)
#' }
#' 
#' @export
gpu_status <- function() {
  list(
    available = gpu_available(),
    info = gpu_info(),
    memory_available = gpu_memory_available(),
    memory_gb = gpu_memory_available() / 1e9,
    recommended_use = gpu_available() && gpu_memory_available() > 1e9
  )
} 