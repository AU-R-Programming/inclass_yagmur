# Standard quadratic loss function
loss <- function(mu, x) {

  val <- sum((x - mu)^2)

  return(val)

}

# Absolute (L1) loss function
loss_l1 <- function(mu, x) {

  val <- sum(abs(x - mu))

  return(val)

}

########## Functions implementing the two optimization routines from previous classes (opt_mean1 and opt_mean2). These take a generic loss function "fn" as an argument.

#' @title Mean optimization
#'
#' @description Compute the minimum of mean.
#'
#' @param x A \code{matrix} data
#'
#' @param fn A \code{string} containing the function to be optimized.
#' @param iter A \code{numeric} (integer) used to denote the number of iterations.
#' @return A \code{number} containing the minimum of loss function:
#'
#' @author Yagmur Yavuz Ozdemir
#' @importFrom stats runif
#' @export
#' @examples
#' opt_mean1(rnrom(10, 2, 1), fn = loss)
opt_mean1 <- function(x, fn, iter = 100) {

  mu_hat <- runif(iter, min(x), max(x)) # vector of "suggested" sample means
  loss_val <- rep(NA, iter) # vector to store corresponding values of the loss function

  for(i in 1:iter) {

    loss_val[i] <- fn(mu_hat[i], x) # evaluate the loss function for all suggested sample means

  }

  return(mu_hat[which.min(loss_val)])

}

# Second optimization function

#' @title Mean optimization 2
#'
#' @description Compute the minimum of mean.
#'
#' @param init A \code{numeric} initial point
#'
#' @param x A \code{matrix} data
#'
#' @param fn A \code{string} containing the function to be optimized.
#' @param iter A \code{numeric} (integer) used to denote the number of iterations.
#' @return A \code{number} containing the minimum of loss function:
#'
#' @author Yagmur Yavuz Ozdemir
#' @importFrom stats runif
#' @export
#' @examples
#' opt_mean2(init=runif(1, min(rnorm(10)), max(rnorm(10))) , fn = loss, rnorm(10) )
opt_mean2 <- function(init = runif(1, min(x), max(x)), fn, x, iter = 100) {

  val_init <- fn(init, x) # one starting/initial loss function value

  for(i in 1:iter) {

    init0 <- runif(1, min(x), max(x)) # suggest a new value for the sample mean
    val_init0 <- fn(init0, x) # evaluate the loss function at this new value

    if(val_init0 < val_init){ # if the new value of the loss function is smaller, accept (update) the new suggested value of the sample mean (and save the new value of the loss function)

      init <- init0
      val_init <- val_init0

    }

  }

  return(init)

}


## Study the different optimization functions
data <- rnorm(100, mean = 3, sd = 1) # generate data
opt_mean1(fn = loss, x = data)
opt_mean2(fn = loss, x = data)
optimize(f = loss, interval = c(min(data), max(data)), x = data) # one parameter optimization function in R
mean(data) # compare previous results to the true sample mean

## Study the different optimization functions using the L1 loss
opt_mean1(fn = loss_l1, x = data)
opt_mean2(fn = loss_l1, x = data)
optimize(f = loss_l1, interval = c(min(data), max(data)), x = data)
mean(data)
median(data) # the optimizations are estimating the median!


#### Create functions to estimate parameters of a basic "neural net". An overview related to this example can be found at https://www.kdnuggets.com/2016/11/quick-introduction-neural-networks.html

# Generate example dataset
set.seed(1)
gender <- sample(c(0,1), size = 100, replace = TRUE) # x1 (input)
age <- round(runif(100, 18, 80)) # x2 (input)
xb <- 3.5*gender + 0.2*age - 9 # w1*x1 + w2*xw + b
p <- 1/(1 + exp(-xb))
y <- rbinom(n = 100, size = 1, prob = p) # output

# Sigmoid function (activation function)
sigmoid <- function(z) {

  return(1 / (1 + exp(-z)))

}

# Loss function (quadratic) to find the parameter vector (weights and biases) of the neural net. In the loss function we are collecting the values of weights and bias into one vector: theta = c(w1, w2, b)
loss_net <- function(theta, y, X) {

  X_net <- cbind(X, rep(1, dim(X)[1])) # adding a column of ones to estimate the bias "b" (third element of the vector theta)
  obj <- sum((y - sigmoid(X_net%*%theta))^2)

  return(obj)

}

# The neural net function (a user-friendly wrapper for the optimization)
neural_net <- function(y, X, init.val = rep(0, dim(X)[2] + 1)) {

  fit <- optim(par = init.val, fn = loss_net, y = y, X = X)

  return(fit$par)

}

X <- cbind(gender, age) # create matrix of inputs
weights <- neural_net(y = y, X = X) # are the estimated weights the same as those used to simulate the data?
