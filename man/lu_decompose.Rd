\name{lu_decompose}
\alias{lu_decompose}
\title{LU Decomposition of a GPU Matrix}
\description{
Performs the LU decomposition of a square gpuTensor matrix using the cuSOLVER library. The function returns a combined L (lower triangular) and U (upper triangular) matrix along with pivot indices.
}
\usage{
lu_decompose(x)
}
\arguments{
  \item{x}{A square \code{gpuTensor} matrix.}
}
\value{
A list containing two \code{gpuTensor} objects:
\item{lu}{A \code{gpuTensor} containing the combined L and U factors of the decomposition.}
\item{ipiv}{A \code{gpuTensor} of integer type containing the pivot indices.}
}
\examples{
\dontrun{
if (acediaR::gpu_is_available()) {
  a_mat <- matrix(c(2, 4, -1, 3), 2, 2)
  a_gpu <- as_tensor(a_mat)
  lu_res <- lu_decompose(a_gpu)
  print(lu_res$lu)
  print(lu_res$ipiv)
}
}
}
\seealso{
\code{\link{solve.gpuTensor}}
} 