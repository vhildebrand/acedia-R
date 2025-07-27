\name{gpu_eigen}
\alias{gpu_eigen}
\title{Eigenvalue Decomposition of a Symmetric GPU Matrix}
\description{
Computes the eigenvalues and, optionally, the eigenvectors of a symmetric gpuTensor matrix using the cuSOLVER library.
}
\usage{
gpu_eigen(x, symmetric = TRUE, only.values = FALSE, ...)
}
\arguments{
  \item{x}{A symmetric \code{gpuTensor} matrix.}
  \item{symmetric}{Logical. If TRUE (the default), the matrix is assumed to be symmetric. Currently, only symmetric matrices are supported.}
  \item{only.values}{Logical. If TRUE, only the eigenvalues are computed and returned, which can be more efficient. If FALSE (the default), both eigenvalues and eigenvectors are returned.}
  \item{...}{Additional arguments, currently ignored.}
}
\value{
If \code{only.values} is TRUE, a \code{gpuTensor} vector containing the eigenvalues.
Otherwise, a list with two components:
\item{values}{A \code{gpuTensor} vector of the eigenvalues.}
\item{vectors}{A \code{gpuTensor} matrix whose columns are the corresponding eigenvectors.}
}
\examples{
\dontrun{
if (acediaR::gpu_is_available()) {
  a_mat <- matrix(c(1, -1, -1, 1), 2, 2)
  a_gpu <- as_tensor(a_mat)
  
  # Get both eigenvalues and eigenvectors
  eigen_res <- gpu_eigen(a_gpu)
  print(as.array(eigen_res$values))
  print(as.array(eigen_res$vectors))

  # Get only eigenvalues
  eigen_vals <- gpu_eigen(a_gpu, only.values = TRUE)
  print(as.array(eigen_vals))
}
}
}
\seealso{
\code{\link{chol.gpuTensor}}, \code{\link{qr.gpuTensor}}
} 