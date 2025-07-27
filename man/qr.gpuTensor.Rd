\name{qr.gpuTensor}
\alias{qr.gpuTensor}
\title{QR Decomposition of a GPU Matrix}
\description{
Computes the QR decomposition of a gpuTensor matrix 'x' into an orthogonal matrix 'Q' and an upper triangular matrix 'R', such that x = Q \%*\% R.
}
\usage{
\method{qr}{gpuTensor}(x, ...)
}
\arguments{
  \item{x}{A \code{gpuTensor} matrix.}
  \item{...}{Additional arguments, currently ignored.}
}
\value{
A list containing two \code{gpuTensor} objects:
\item{Q}{The orthogonal matrix 'Q'.}
\item{R}{The upper triangular matrix 'R'.}
}
\examples{
\dontrun{
if (acediaR::gpu_is_available()) {
  a_mat <- matrix(rnorm(12), 4, 3)
  a_gpu <- as_tensor(a_mat)
  qr_res <- qr(a_gpu)
  Q_gpu <- qr_res$Q
  R_gpu <- qr_res$R
  # Verify decomposition: a_recon <- matmul(Q_gpu, R_gpu)
  print(dim(Q_gpu))
  print(dim(R_gpu))
}
}
}
\seealso{
\code{\link{gpu_eigen}}, \code{\link{lu_decompose}}
} 