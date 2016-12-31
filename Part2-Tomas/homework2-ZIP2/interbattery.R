#apply inter-batterys anaylsis
interbattery <- function(X, Y) {
  Vxy = var(X, Y)
  Vyx = var(Y, X)
  
  # compute the eigenvalues
  eigen = eigen(Vyx %*% Vxy)
  
  # get the rank of the variance matrix and keep only as many eigenvectors
  B = eigen$vectors
  mu_sq = eigen$values
  mu = sqrt(mu_sq)
  
  # transformation matrix A
  A = Vxy %*% B
  for (i in 1:length(mu))
  {
    A[,i] = A[,i] / mu[i]
  }
  
  T = X %*% A
  U = Y %*% B
  
  r <- list(
    A = A,
    B = B,
    T = T,
    U = U,
    mu_sq = mu_sq,
    n_components = 10
  )
  
  return(r)
}