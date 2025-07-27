\name{determinant.gpuTensor}
\alias{determinant.gpuTensor}
\title{Determinant of a GPU Matrix}
\description{
Computes the determinant of a square gpuTensor matrix. The calculation is performed using the result of an LU decomposition on the GPU.
}
\usage{
\method{determinant}{gpuTensor}(x, logarithm = TRUE, ...)
}
\arguments{
  \item{x}{A square \code{gpuTensor} matrix.}
  \item{logarithm}{Logical. If TRUE (the default), the function returns the logarithm of the absolute value of the determinant, which is useful for avoiding overflow or underflow with very large or small determinants.}
  \item{...}{Additional arguments, currently ignored.}
}
\value{
If \code{logarithm} is TRUE, a list containing the modulus (log of the absolute value of the determinant) and the sign. If FALSE, a single numeric value representing the determinant.
}
\examples{
\dontrun{
if (acediaR::gpu_is_available()) {
  a_mat <- matrix(c(2, -1, 4, 3), 2, 2)
  a_gpu <- as_tensor(a_mat)
  
  # Get log determinant
  log_det <- determinant(a_gpu)
  print(log_det)

  # Get raw determinant
  raw_det <- determinant(a_gpu, logarithm = FALSE)
  print(raw_det) # Expected: 10
}
}
}
\seealso{
\code{\link{lu_decompose}}
} 