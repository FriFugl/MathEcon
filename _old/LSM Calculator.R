simulateStockPath <- function(n, periods,dt, s0, r, sigma){
  
  paths <- data.frame(rep(s0, n))
  colnames(paths) <- c('s_0')
  
  for (i in 1:periods){
    z <- rnorm(n=n, mean=0, sd=1)
    s <- paths[,ncol(paths)]*exp((r-(sigma^2)/2)*dt+sqrt(dt)*sigma*z)
    
    paths[[paste0("s_", i)]] <- s
  }
  
  return(paths)
}

calcPutPayoff <- function(prices, strike){
  payoffs <- pmax(strike - prices, 0)
  
  return(payoffs)
}

LSM <- function(stockpaths,r, strike){
  periods <- ncol(stockpaths)
  
  #Cash flow matrix which will be continually updated
  stockpaths$cf <- calcPutPayoff(stockpaths[,periods], strike)
  
  for (i in 1:(periods - 2)){
    ITM_indices <- which(stockpaths[, periods - i] < strike) # Indices of in-the-money paths
    OTM_indices <- setdiff(seq_len(nrow(stockpaths)), ITM_indices) # Indices of out-of-the-money paths
    
    ITM_Paths <- stockpaths[ITM_indices, ] # Subsetting the data
    
    Y <- ITM_Paths$cf/(1+r) # Discounted cashflow from keeping the in-the-money options
    X <- ITM_Paths[,periods - i] # Value of the stock
    
    # Regression step
    X_matrix <- cbind(1, X, X^2)
    
    beta <- solve(t(X_matrix) %*% X_matrix) %*% (t(X_matrix) %*% Y)
    beta_0 <- beta[1]; beta_1 <- beta[2]; beta_2 <- beta[3]
    
    continuationValue <- beta_0 + beta_1 * X + beta_2 * (X^2)
    exerciseValue <- strike - X
    
    # Updating the cashflow matrix
    stockpaths$cf[ITM_indices] <- ifelse(exerciseValue > continuationValue, exerciseValue, stockpaths$cf[ITM_indices]/(1+r)) 
    stockpaths$cf[OTM_indices] <- stockpaths$cf[OTM_indices] / (1 + r)
  
  }
  
  return (sum(stockpaths$cf/(1+r))/nrow(stockpaths))
}

s_0 <- c(1,1,1,1,1,1,1,1)
s_1 <- c(1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88)
s_2 <- c(1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22)
s_3 <- c(1.34, 1.54, 1.03, 0.92, 1.52, 0.9, 1.01, 1.34)
paths <- data.frame(s_0, s_1, s_2, s_3)

LSM(paths,0.06, 1.1)

