library(MASS)   # For ridge regression
library(glmnet) # For LASSO
library(pls)
library(mgcv)
library(rpart)
library(rpart.plot)
library(nnet)
data_raw <- read.csv("Data2020.csv")
test_raw <- read.csv("Data2020testX.csv")
data <- na.omit(data_raw[, c(1:16)])
test <- na.omit(test_raw, c(1:15))

#data <- data.frame("Y"=data1$Y, "X2"=data1$X2, "X4"=data1$X4, "X10"=data1$X10, "X12"=data1$X12)
#pairs(data)
#pairs(data[, c(1:8)])
#pairs(data[, c(1, 9, 10, 11, 12, 13, 14, 15, 16)])
#pc1 <- prcomp(x=data[, 2:16], scale.=TRUE)
#pc <- prcomp(x=data1[,c(3,5,13)], scale.=TRUE)
#vars = pc$sdev^2

#plot(1:length(vars), vars, main = "Variability Explained", 
#     xlab = "Principal Component", ylab = "Variance Explained")
#c.vars = cumsum(vars)   ### Cumulative variance explained
#rel.c.vars = c.vars / max(c.vars)   ### Cumulative proportion of 
### variance explained
#plot(1:length(rel.c.vars), rel.c.vars,
#     main = "Proportion of Variance Explained by First W PCs",
#     xlab = "W", ylab = "Proportion of Variance Explained")


source("Helper Functions (1).R")


set.seed(2928893)


# data1$set <- ifelse(runif(n=nrow(data1))>0.5, yes=2, no=1)
# 
# library(leaps)
# #  Note: default is to limit to 8-variable models.  
# #  Add nvmax argument to increase when needed.
# allsub1 <- regsubsets(x=data1[data1$set==1,2:15], 
#                       y=data1[data1$set==1,1], nbest=1)
# allsub2 <- regsubsets(x=data1[data1$set==2,2:15], 
#                       y=data1[data1$set==2,1], nbest=1)
# 
# 
# # Store summary() so we can see BICs 
# summ.1 <- summary(allsub1)
# summ.2 <- summary(allsub2)
# 
# summ.1
# summ.2
# 
# names(summ.1)
# summ.1$bic
# summ.2$bic

# Plot of results in a special form
# x11(h=7, w=10, pointsize=12)
# par(mfrow=c(1,2))
# plot(allsub1, main="All Subsets on half of Project data")
# plot(allsub2, main="All Subsets on other half of Project data")

siz <- c(1,3,5,7, 9)
dec <- c(0.001, 0.1, 0.5, 1, 2)

rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}
max.terms = 15
### Let's define a function for constructing CV folds
get.folds = function(n, K) {
  ### Get the appropriate number of fold labels
  n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
  fold.ids.raw = rep(1:K, times = n.fold) # Generate extra labels
  fold.ids = fold.ids.raw[1:n] # Keep only the correct number of labels
  
  ### Shuffle the fold labels
  folds.rand = fold.ids[sample.int(n)]
  
  return(folds.rand)
}

### Number of folds
K = 10

### Construct folds
n = nrow(data) # Sample size
folds = get.folds(n, K)

### Create a container for MSPEs. Let's include ordinary least-squares ######
all.models = c("LS", "Step", "Ridge", "LASSO-Min", "LASSO-1se", "PLS", "Full", "mincv", "1-se", "PPR", "GAM", "NN", "nodes", "shrinkage","Terms", "optimalComp")
all.MSPEs = array(0, dim = c(K, length(all.models)))
colnames(all.MSPEs) = all.models

### Begin cross-validation ####
for(i in 1:K){
  ### Split data
  data.train = data[folds != i,]
  data.valid = data[folds == i,]
  n.train = nrow(data.train)
  X.train = data.train[, -1]
  X.valid = data.valid[, -1]
  ### Get response vectors
  Y.train = data.train$Y
  Y.valid = data.valid$Y
  
  ##### least squares  ####
  fit.ls = lm(Y ~ ., data = data.train)
  pred.ls = predict(fit.ls, newdata = data.valid)
  MSPE.ls = get.MSPE(Y.valid, pred.ls)
  all.MSPEs[i, "LS"] = MSPE.ls
  
  ##### stepwise ######
  fit.start = lm(Y ~ 1, data = data.train)
  fit.end = lm(Y ~ .^2, data = data.train)
  
  step.BIC = step(fit.start, list(upper = fit.end), k = log(n.train),
                  trace = 0)
  
  pred.step.BIC = predict(step.BIC, data.valid)
  
  err.step.BIC = get.MSPE(Y.valid, pred.step.BIC)
  
  all.MSPEs[i, "Step"] = err.step.BIC
  
  ####### RIDGE #######
  lambda.vals = seq(from = 0, to = 100, by = 0.05)
  ### Use the lm.ridge() function to fit a ridge regression model. The
  ### syntax is almost identical to the lm() function, we just need
  ### to set lambda equal to our list of candidate values.
  fit.ridge = lm.ridge(Y ~ ., lambda = lambda.vals, 
                       data = data.train)
  ind.min.GCV = which.min(fit.ridge$GCV)
  lambda.min = lambda.vals[ind.min.GCV]
  all.coefs.ridge = coef(fit.ridge)
  coef.min = all.coefs.ridge[ind.min.GCV,]
  
  matrix.valid.ridge = model.matrix(Y ~ ., data = data.valid)
  pred.ridge = matrix.valid.ridge %*% coef.min
  MSPE.ridge = get.MSPE(Y.valid, pred.ridge)
  all.MSPEs[i, "Ridge"] = MSPE.ridge
  
  ##### LASSO #######
  matrix.train.raw = model.matrix(Y ~ ., data = data.train)
  matrix.train = matrix.train.raw[,-1]
  
  all.LASSOs = cv.glmnet(x = matrix.train, y = Y.train)
  
  lambda.min = all.LASSOs$lambda.min
  lambda.1se = all.LASSOs$lambda.1se
  
  coef.LASSO.min = predict(all.LASSOs, s = lambda.min, type = "coef")
  coef.LASSO.1se = predict(all.LASSOs, s = lambda.1se, type = "coef")
  
  included.LASSO.min = predict(all.LASSOs, s = lambda.min, 
                               type = "nonzero")
  included.LASSO.1se = predict(all.LASSOs, s = lambda.1se, 
                               type = "nonzero")
  
  matrix.valid.LASSO.raw = model.matrix(Y ~ ., data = data.valid)
  matrix.valid.LASSO = matrix.valid.LASSO.raw[,-1]
  pred.LASSO.min = predict(all.LASSOs, newx = matrix.valid.LASSO,
                           s = lambda.min, type = "response")
  pred.LASSO.1se = predict(all.LASSOs, newx = matrix.valid.LASSO,
                           s = lambda.1se, type = "response")
  MSPE.LASSO.min = get.MSPE(Y.valid, pred.LASSO.min)
  all.MSPEs[i, "LASSO-Min"] = MSPE.LASSO.min
  
  MSPE.LASSO.1se = get.MSPE(Y.valid, pred.LASSO.1se)
  all.MSPEs[i, "LASSO-1se"] = MSPE.LASSO.1se
  
  
  ####PLS#####
  
  fit.pls = plsr(Y ~ ., data = data.train, validation = "CV",
                 segments = 5)
  
  CV.pls = fit.pls$validation # All the CV information
  PRESS.pls = CV.pls$PRESS    # Sum of squared CV residuals
  CV.MSPE.pls = PRESS.pls / nrow(data.train)  # MSPE for internal CV
  ind.best.pls = which.min(CV.MSPE.pls) # Optimal number of components
  all.MSPEs[i, "optimalComp"] = ind.best.pls
  
  ### Get predictions and calculate MSPE on the validation fold
  ### Set ncomps equal to the optimal number of components
  pred.pls = predict(fit.pls, data.valid, ncomp = ind.best.pls)
  MSPE.pls = get.MSPE(Y.valid, pred.pls)
  all.MSPEs[i, "PLS"] = MSPE.pls
  
  
  ## Regression trees 
  fit.tree = rpart(Y ~.,  data = data.train, cp = 0, method = "anova")
  info.tree = fit.tree$cptable
  info.tree1 <- fit.tree$cptable[,c(2:5,1)]
  
  ind.min = which.min(info.tree[,"xerror"])
  CP.min.raw = info.tree[ind.min, "CP"]
  
  if(ind.min == 1){
    ### If minimum CP is in row 1, store this value
    CP.min = CP.min.raw
  } else{
    ### If minimum CP is not in row 1, average this with the value from the
    ### row above it.
    
    ### Value from row above
    CP.above = info.tree[ind.min-1, "CP"]
    
    ### (Geometric) average
    CP.min = sqrt(CP.min.raw * CP.above)
  }
  
  fit.tree.min = prune(fit.tree, cp = CP.min)
  
  err.min = info.tree[ind.min, "xerror"]
  se.min = info.tree[ind.min, "xstd"]
  threshold = err.min + se.min
  
  ind.1se = min(which(info.tree[1:ind.min,"xerror"] < threshold))
  CP.1se.raw = info.tree[ind.1se, "xerror"]
  if(ind.1se == 1){
    ### If best CP is in row 1, store this value
    CP.1se = CP.1se.raw
  } else{
    ### If best CP is not in row 1, average this with the value from the
    ### row above it.
    
    ### Value from row above
    CP.above = info.tree[ind.1se-1, "CP"]
    
    ### (Geometric) average
    CP.1se = sqrt(CP.1se.raw * CP.above)
  }
  
  ### Prune the tree
  fit.tree.1se = prune(fit.tree, cp = CP.1se)
  
  pred.tree = predict(fit.tree, data.valid)
  MSPE.full = get.MSPE(Y.valid, pred.tree)
  
  pred.mincv = predict(fit.tree.min, data.valid)
  MSPE.mincv = get.MSPE(Y.valid, pred.mincv)
  
  pred.1se = predict(fit.tree.1se, data.valid)
  MSPE.1se = get.MSPE(Y.valid, pred.1se)
  
  
  all.MSPEs[i, "Full"] = MSPE.full
  all.MSPEs[i, "mincv"] = MSPE.mincv
  all.MSPEs[i, "1-se"] = MSPE.1se
  
  
  
  ######## GAM #######
  fit.gam = gam(data=data.train,
                formula = Y ~ s(X1) + s(X2) + s(X3) + X4 + s(X5) + s(X6) +
                  s(X7) + s(X8) + s(X9) + X10 + s(X11) + X12 + s(X13) + s(X14) + s(X15),
                family = gaussian(link=identity))
  summary(fit.gam)
  pred.gam = predict(fit.gam, data.valid)
  MSPE.gam = get.MSPE(Y.valid, pred.gam)
  all.MSPEs[i, "GAM"] = MSPE.gam
  
  K.ppr = 5
  n.trainppr = nrow(data.train)
  folds.ppr = get.folds(n.trainppr, K.ppr)
  MSPEs.ppr = array(0, dim = c(K.ppr, max.terms))
  for(j in 1:K.ppr){
    ### Split the training data.
    ### Be careful! We are constructing an internal validation set by 
    ### splitting the training set from outer CV.
    train.ppr = data.train[folds.ppr != j,]
    valid.ppr = data.train[folds.ppr == j,] 
    Y.valid.ppr = valid.ppr$Y
    
    ### We need to fit several different PPR models, one for each number
    ### of terms. This means another for loop (make sure you use a different
    ### index variable for each loop).
    for(l in 1:max.terms){
      ### Fit model
      fit.ppr = ppr(Y ~ ., data = train.ppr, 
                    max.terms = max.terms, nterms = l, sm.method = "gcvspline")
      
      ### Get predictions and MSPE
      pred.ppr = predict(fit.ppr, valid.ppr)
      MSPE.ppr = get.MSPE(Y.valid.ppr, pred.ppr) # Our helper function
      
      ### Store MSPE. Make sure the indices match for MSPEs.ppr
      MSPEs.ppr[j, l] = MSPE.ppr
    }
    
  }
  ######### Get average MSPE for each number of terms #####
  ave.MSPE.ppr = apply(MSPEs.ppr, 2, mean)
  
  ### Get optimal number of terms
  best.terms = which.min(ave.MSPE.ppr)
  
  ### Fit PPR on the whole CV training set using the optimal number of terms 
  fit.ppr.best = ppr(Y ~ ., data = data.train,
                     max.terms = max.terms, nterms = best.terms, sm.method = "gcvspline")
  
  ### Get predictions, MSPE and store results
  pred.ppr.best = predict(fit.ppr.best, data.valid)
  MSPE.ppr.best = get.MSPE(Y.valid, pred.ppr.best) # Our helper function
  
  all.MSPEs[i, "PPR"] = MSPE.ppr.best
  all.MSPEs[i, "Terms"] = best.terms
  
  
  ######## CV ########
  nrounds = 5
  R=5
  V=2
  n2 = nrow(X.train)
  folds = matrix(NA, nrow=n2, ncol=R)
  for(r in 1:R){
    folds[,r]=floor((sample.int(n2)-1)*V/n2) + 1
  }
  # Start loop over all reps and folds.  
  for (r in 1:R){ 
    for(v in 1:V){
      
      y.1 <- as.matrix(Y.train[folds[,r]!=v])
      x.1.unscaled <- as.matrix(X.train[folds[,r]!=v,]) 
      x.1 <- rescale(x.1.unscaled, x.1.unscaled) 
      
      #Test
      y.2 <- as.matrix(Y.train[folds[,r]==v])
      x.2.unscaled <- as.matrix(X.train[folds[,r]==v,]) # Original data set 2
      x.2 = rescale(x.2.unscaled, x.1.unscaled)
      
      # Start counter to add each model's MSPE to row of matrix
      qq=1
      # Start Analysis Loop for all combos of size and decay on chosen data set
      for(d in dec){
        for(s in siz){
          
          ## Restart nnet nrounds times to get best fit for each set of parameters 
          MSE.final <- 9e99
          #  check <- MSE.final
          for(p in 1:nrounds){
            nn <- nnet(y=y.1, x=x.1, linout=TRUE, size=s, decay=d, maxit=500, trace=FALSE)
            MSE <- nn$value/nrow(x.1)
            if(MSE < MSE.final){ 
              MSE.final <- MSE
              nn.final <- nn
            }
          }
          pred.nn = predict(nn.final, newdata=x.2)
          MSPEs.cv[qq,(r-1)*V+v+2] = mean((y.2 - pred.nn)^2)
          qq = qq+1
        }
      }
    }
  }
  MSPEs.cv
  (MSPEcv = apply(X=MSPEs.cv[,-c(1,2)], MARGIN=1, FUN=mean))
  (MSPEcv.sd = apply(X=MSPEs.cv[,-c(1,2)], MARGIN=1, FUN=sd))
  MSPEcv.CIl = MSPEcv - qt(p=.95, df=R*V-1)*MSPEcv.sd/sqrt(R*V)
  MSPEcv.CIu = MSPEcv + qt(p=.95, df=R*V-1)*MSPEcv.sd/sqrt(R*V)
  (all.cv = cbind(MSPEs.cv[,1:2],round(cbind(MSPEcv,MSPEcv.CIl, MSPEcv.CIu),2)))
  all.cv[order(MSPEcv),]
  u <- data.frame(all.cv[order(MSPEcv),])
  all.MSPEs[i, "nodes"] = u[1, ]$V1
  all.MSPEs[i, "shrinkage"] = u[1, ]$V2
  pred.nn = predict(nn.final, newdata=X.valid)
  all.MSPEs[i, "NN"] = mean((Y.valid - pred.nn)^2)
  
}

boxplot(all.MSPEs[,c(1:12)], main = paste0("CV MSPEs over ", K, " folds"), las=2)

### Calculate RMSPEs
all.RMSPEs = apply(all.MSPEs[,c(1:12)], 1, function(W){
  best = min(W)
  return(W / best)
})
all.RMSPEs = t(all.RMSPEs)

### Make a boxplot of RMSPEs #####
#boxplot(all.RMSPEs[,c(1:7)], main = paste0("CV RMSPEs over ", K, " folds"), las=2)

### One model is much worse than the others. Let's zoom in on the
### good models.
#boxplot(all.RMSPEs[,c(1:7)], ylim = c(1, 1.05),
#        main = paste0("CV RMSPEs over ", K, 
#                      " folds (enlarged to show texture)"),las=2)


fit.gam = gam(data=data,
              formula = Y ~ s(X1) + s(X2) + s(X3) + X4 + s(X5) + s(X6) +
                s(X7) + s(X8) + s(X9) + X10 + s(X11) + X12 + s(X13) + s(X14) + s(X15),
              family = gaussian(link=identity))
summary(fit.gam)
pred.gam = predict(fit.gam, test)
write.table(pred.gam, "/Users/dollina/Desktop/Submit.csv", sep = ",",row.names = FALSE, col.names=FALSE)
