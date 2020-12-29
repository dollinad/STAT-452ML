### We will regularly need to shuffle a vector. This function
### does that for us.
shuffle = function(X){
  new.order = sample.int(length(X))
  new.X = X[new.order]
  return(new.X)
}

### We will also often need to calculate MSE using an observed
### and a prediction vector. This is another useful function.
get.MSPE = function(Y, Y.hat){
  return(mean((Y - Y.hat)^2))
}