\name{chol.gpuTensor}
\alias{chol.gpuTensor}
\title{Cholesky Decomposition of a GPU Matrix}
\description{
Computes the Cholesky decomposition of a symmetric, positive-definite gpuTensor matrix. This function returns the lower triangular factor L such that the input matrix 'x' is equal to L \%*\% t(L).
}
\usage{
\method{chol}{gpuTensor}(x, ...)
}
\arguments{
  \item{x}{A symmetric, positive-definite \code{gpuTensor} matrix.}
  \item{...}{Additional arguments, currently ignored.}
}
\value{
A \code{gpuTensor} representing the lower triangular Cholesky factor 'L'.
}
\examples{
\dontrun{
if (acediaR::gpu_is_available()) {
  a_mat <- matrix(c(4, 1, 1, 3), 2, 2)
  a_gpu <- as_tensor(a_mat)
  l_gpu <- chol(a_gpu)
  print(as.array(l_gpu))
  # Reconstruct original matrix: a_recon <- matmul(l_gpu, transpose(l_gpu))
}
}
}
\seealso{
\code{\link{solve.gpuTensor}}, \code{\link{gpu_eigen}}
} 