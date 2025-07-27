\name{solve.gpuTensor}
\alias{solve.gpuTensor}
\title{Solve a System of Linear Equations on the GPU}
\description{
Solves the linear system a \%*\% x = b for x, where 'a' is a square gpuTensor matrix and 'b' is a gpuTensor vector or matrix. This function uses LU decomposition provided by the cuSOLVER library.
}
\usage{
\method{solve}{gpuTensor}(a, b, ...)
}
\arguments{
  \item{a}{A square \code{gpuTensor} matrix representing the coefficients of the linear system.}
  \item{b}{A \code{gpuTensor} vector or matrix representing the right-hand side of the linear system.}
  \item{...}{Additional arguments, currently ignored.}
}
\value{
A \code{gpuTensor} containing the solution 'x' to the system.
}
\examples{
\dontrun{
if (acediaR::gpu_is_available()) {
  a_mat <- matrix(c(2, 1, -1, -3, -1, 2, -2, 1, 2), 3, 3)
  b_vec <- c(8, -11, -3)
  a_gpu <- as_tensor(a_mat)
  b_gpu <- as_tensor(b_vec)
  x_gpu <- solve(a_gpu, b_gpu)
  print(as.array(x_gpu))
  # Expected output should be close to: [2, 3, -1]
}
}
}
\seealso{
\code{\link{lu_decompose}}
} 