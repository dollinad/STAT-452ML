## Project 2  ###
# Methods to try 
## Basic 
    # KNN K=?
    # KNN CV-min
    # KNN CV-1se
## Linear
    # Logistic Regression 
    # Logistic Regression - LASSO CV-min
    # Logistic Regression - LASSO CV-1se
    # LDA 
    # QDA
## Non-Linear
    # GAM
    # Naive Bayes 
    # Naive Bayes with PCA
## Trees
    # Classification Trees Full
    # Classification Trees CV-min
    # Classification Trees CV-1se
    # Random Forests 
    # Boosting 
## Neural Nets 
    # Naive 
    # Tuned 
## Support Vectors 
    # Naive support vector machine
    # Tuned support vector machine

set.seed(46685327, kind="Mersenne-Twister")

library(FNN)
library(tidyverse)
library(pls)
library(nnet)
library(car)
library(glmnet)
library(MASS)
#library(mgcv)   
library(klaR) 
library(rpart)
library(randomForest)
library(e1071)
library(caret)
library(nnet)
library(foreach)
library(doSNOW)
library("doParallel")

data_raw <- read.csv("P2Data2020.csv")
test_raw <- read.csv("P2Data2020testX.csv")
data <- na.omit(data_raw[, c(1:17)])
test <- na.omit(test_raw[, c(1:16)])
data$Y <- as.factor(data$Y)

scale.1 <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

### Rescale x1 using the means and SDs of x2
scale.3 <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}


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
all.models = c("KNN-Naive", "KNNCV", "Chosen K", "KNNSE", "Chosen SE", "multinom","LASSO", "LASSOCV", "LASSOSE", "LDA", "QDA", "GAM",
              "Naive Bayes GK", "NB PCA ND", "NB PCA KD", "Full Tree", "TreeCV", "Tree1se", 
              "RFD", "RF Tuned", "RF V", "RF NS", "SVM", "SVMT", "NNetD", "NNetT", "NNet Node", "NNet Decay" )
all.MC = array(0, dim = c(K, length(all.models)))
colnames(all.MC) = all.models

for(i in 1:K){
  print(paste0(i, " CofC ", K))
  data.train = data[folds != i,]
  data.valid = data[folds == i,]
  n.train = nrow(data.train)
  X.train = data.train[, -1]
  X.valid = data.valid[, -1]
  
  ### Get response vectors
  Y.train = data.train$Y
  Y.valid = data.valid$Y
  
  ###### KNN #######
  # Creating training and test X matrices, then scaling them.
  X.train.unscaled <- as.matrix(X.train)
  X.train.scaled <- scale.1(X.train.unscaled, X.train.unscaled)
  X.valid.unscaled <- as.matrix(X.valid)
  X.valid.scaled <- scale.1(X.valid.unscaled, X.train.unscaled)
  # Fit the 1-NN function using set 1 to train AND test 
  #   (compute training error)
  knnfit.1.1 <- knn(train=X.train.scaled, test=X.train.scaled, cl=Y.train, k=1)
  # Create Confusion Matrix and misclass rate
  table(knnfit.1.1, Y.train,  dnn=c("Predicted","Observed"))
  misclass.knn1.1 <- 
    mean(ifelse(knnfit.1.1 == Y.train, yes=0, no=1))
  
  # Fit the 1-NN function using set 1 to train and set2 to test 
  #   (compute test error)
  knnfit.1.2 <- knn(train=X.train.scaled, test=X.valid.scaled, cl=Y.train, k=1)
  # Create Confusion Matrix and misclass rate
  table(knnfit.1.2, Y.valid,  dnn=c("Predicted","Observed"))
  misclass.knn1.2 <- 
    mean(ifelse(knnfit.1.2 == Y.valid, yes=0, no=1))
  all.MC[i, "KNN-Naive"] = misclass.knn1.2
  
  #  Now we tune using cv.knn(), which does Leave-one-out (n-fold) CV
  # I created the steps below to fit a sequence of k  values to tune the knn.
  #  Enter the maximum k as kmax.  Run knn on training set and predict test set.
  #  Then compute test misclassification proportion as the output,
  #  Need to change the data sets in the two lines in the "runknn" function.
  kmax <- 200
  k <- matrix(c(1:kmax), nrow=kmax)
  runknn <- function(x){
    knncv.fit <- knn.cv(train=X.train.scaled, cl=Y.train, k=x)
    # Fitted values are for deleted data from CV
    mean(ifelse(knncv.fit == Y.train, yes=0, no=1))
  }
  
  mis <- apply(X=k, MARGIN=1, FUN=runknn)
  mis.se <- sqrt(mis*(1-mis)/nrow(data.valid)) #SE of misclass rates
  
  # k for Minimum CV error
  mink = which.min(mis)
  all.MC[i, "Chosen K"] = mink
  #Trying the value of k with the lowest validation error on test data set.
  knnfitmin.2 <- knn(train=X.train.scaled, test=X.valid.scaled, cl=Y.train, k=mink)
  
  table(knnfitmin.2, Y.valid,  dnn=c("Predicted","Observed"))
  misclass.2.knnmin <- mean(ifelse(knnfitmin.2 == Y.valid, yes=0, no=1))
  all.MC[i, "KNNCV"]= misclass.2.knnmin
  all.MC[i, "Chosen K"] = mink
  
  # Less variable models have larger k, so find largest k within 
  #   1 SE of minimum validation error 
  serule = max(which(mis<mis[mink]+mis.se[mink]))
  all.MC[i, "Chosen SE"] = serule
  knnfitse.2 <- knn(train=X.train.scaled, test=X.valid.scaled, cl=Y.train, k=serule)
  table(knnfitse.2, Y.valid,  dnn=c("Predicted","Observed"))
  misclass.2.knnse <- mean(ifelse(knnfitse.2 == Y.valid, yes=0, no=1))
  all.MC[i, "Chosen K"] = mink
  all.MC[i, "KNNSE"]= misclass.2.knnse
  
  
  
  
  ###### multinom/LASSO ######
  train <- data.frame(cbind(rescale(X.train, X.train), Y=Y.train))
  valid <- data.frame(cbind(rescale(X.valid, X.train), Y=Y.valid))
  mod.fit <- multinom(data=train, formula=Y ~ ., trace=TRUE, maxit=600)
  
  pred.class.1 <- predict(mod.fit, newdata=train, type="class")
  pred.class.2 <- predict(mod.fit, newdata=valid, type="class")
  ul.misclass.train <- mean(ifelse(pred.class.1 == train$Y, yes=0, no=1))
  mul.misclass.test <- mean(ifelse(pred.class.2 == valid$Y, yes=0, no=1))
  all.MC[i, "multinom"] = mul.misclass.test
  
  # Estimated probabilities for test set
  pred.probs.2 <- predict(mod.fit, newdata=valid, type="probs")
  #round(head(pred.probs.2),3)
  # Test set confusion matrix
  #table(set2$type, pred.class.2, dnn=c("Obs","Pred"))
  
  logit.fit <- glmnet(x=as.matrix(train[,-17]), y=train[,17], family="multinomial")
  logit.prob.2 <- predict(logit.fit, s=0, type="response",
                          newx=as.matrix(valid[,-17]))
  #round(head(logit.prob.2[,,1]), 3)
  
  las0.pred.train <- predict(object=logit.fit, s=0, type="class",
                             newx=as.matrix(train[,-17]))
  las0.pred.test <- predict(logit.fit, s=0, type="class",
                            newx=as.matrix(valid[,-17]))
  las0misclass.train <- 
    mean(ifelse(las0.pred.train == train$Y, 
                yes=0, no=1))
  las0misclass.test <- 
    mean(ifelse(las0.pred.test == valid$Y,
                yes=0, no=1))
  all.MC[i, "LASSO"] = las0misclass.test
  
  # "Optimal" LASSO Fit
  logit.cv <- cv.glmnet(x=as.matrix(train[,-17]), 
                        y=train[,17], family="multinomial")
  #logit.cv
  #x11()
  #plot(logit.cv)
  
  ## Find nonzero lasso coefficients
  #c <- coef(logit.fit,s=logit.cv$lambda.min) 
  #cmat <- cbind(as.matrix(c[[1]]), as.matrix(c[[2]]), 
  #as.matrix(c[[3]]))
  #round(cmat,2)
  #cmat!=0
  lambda.min = logit.cv$lambda.min
  lambda.1se = logit.cv$lambda.1se
  lascv.pred.min <- predict(object=logit.cv, type="class", 
                            s=lambda.min, 
                            newx=as.matrix(valid[,-17]))
  lascv.pred.se <- predict(logit.cv, type="class", 
                           s=lambda.1se, 
                           newx=as.matrix(valid[,-17]))
  lascvmisclass.min <- 
    mean(ifelse(lascv.pred.min == valid$Y, yes=0, no=1))
  lascvmisclass.se <- 
    mean(ifelse(lascv.pred.se == valid$Y, yes=0, no=1))
  all.MC[i, "LASSOCV"] = lascvmisclass.min
  all.MC[i, "LASSOSE"] = lascvmisclass.se
  
  ###### LDA/QDA/GAM #####
  X.train.DA = scale.3(data.train[,-1], data.train[,-1])
  X.valid.DA = scale.3(data.valid[,-1], data.train[,-1])
  
  ### Fit an LDA model using the lda() funtion from the MASS package. This
  ### function uses predictor/response syntax.
  fit.lda = lda(X.train.DA, Y.train)
  ### We get predictions by extracting the class object from the predict()
  ### function's output.
  pred.lda = predict(fit.lda, X.valid.DA)$class
  #table(Y.valid, pred.lda, dnn = c("Obs", "Pred"))
  miss.lda = mean(Y.valid != pred.lda)
  all.MC[i, "LDA"] = miss.lda
  ### Finally, QDA works much the same way as LDA.
  
  fit.qda = qda(X.train.DA, Y.train)
  
  pred.qda = predict(fit.qda, X.valid.DA)$class
  
  #table(Y.valid, pred.qda, dnn = c("Obs", "Pred"))
  miss.qda = mean(Y.valid != pred.qda)
  all.MC[i, "QDA"] = miss.lda
  
  ##### GAM#### 
  #data.train1 = data.train
  #data.train1$Y0 <- as.numeric(data.train1$Y)-1
  
  #gam.m <- gam(data=data.train1, list(Y0
  #                                  ~s(X1) + s(X2) + s(X3) + s(X4) +
  #                                   s(X5) + s(X6) + s(X7) + s(X8) +
  #                                  s(X9) + s(X10) + s(X11) + s(X12) +
  #                                 s(X13) + s(X14) + s(X15) + s(X16),
  #                              ~s(X1) + s(X2) + s(X3) + s(X4) +
  #                               s(X5) + s(X6) + s(X7) + s(X8) +
  #                              s(X9) + s(X10) + s(X11) + s(X12) +
  #                             s(X13) + s(X14) + s(X15) + s(X16),
  #                          ~s(X1) + s(X2) + s(X3) + s(X4) +
  #                           s(X5) + s(X6) + s(X7) + s(X8) +
  #                          s(X9) + s(X10) + s(X11) + s(X12) +
  #                         s(X13) + s(X14) + s(X15) + s(X16),
  #                      ~s(X1) + s(X2) + s(X3) + s(X4) +
  #                       s(X5) + s(X6) + s(X7) + s(X8) +
  #                      s(X9) + s(X10) + s(X11) + s(X12) +
  #                     s(X13) + s(X14) + s(X15) + s(X16)),
  #                  family=multinom(K=4)) 
  #pred.prob.m <- predict(gam.m, newdata=data.train1, type="response")
  #pred.class.m <- apply(pred.prob.m,1,function(x) which(max(x)==x)[1])-1
  
  #pred.prob.2m <- predict(gam.m, newdata=data.valid, type="response")
  #pred.class.2m <- apply(pred.prob.2m,1,function(x) which(max(x)==x)[1])-1
  
  #misclassm.train <- mean(pred.class.m != as.numeric(data.train$Y)-1)
  #misclassm.test <- mean(pred.class.2m != as.numeric(data,train$Y)-1)
  #all.MC[i, "GAM"] = misclassm.test    
  
  ###### Naive Bayes #### 
  NBn <- NaiveBayes(x=data.train[,-1], grouping=data.train[,1], usekernel=FALSE)
  NBk.pred.train <- predict(NBn, newdata=data.train[,-1])
  NBk.pred.test <- predict(NBn, newdata=data.valid[,-1])
  # Error rates
  NBkmisclass.train <- mean(ifelse(NBk.pred.train$class == data.train$Y, yes=0, no=1))
  NBkmisclass.test <- mean(ifelse(NBk.pred.test$class == data.valid$Y, yes=0, no=1))
  all.MC[i, "Naive Bayes GK"] =  NBkmisclass.test
  
  ####################################################################
  # Run PCA before Naive Bayes to decorrelate data
  pc <-  prcomp(x=data.train[,-1], scale.=TRUE)
  
  # Create the same transformations in all three data sets 
  #   and attach the response variable at the end
  #   predict() does this 
  xi.1 <- data.frame(pc$x, Y = as.factor(data.train$Y))
  xi.2 <- data.frame(predict(pc, newdata=data.valid), Y = as.factor(data.valid$Y))
  
  #  First with normal distributions
  NBn.pc <- NaiveBayes(x=xi.1[,-17], grouping=xi.1[,17], usekernel=FALSE)
  NBnpc.pred.train <- predict(NBn.pc, newdata=xi.1[,-17], type="class")
  NBnpc.pred.test <- predict(NBn.pc, newdata=xi.2[,-17], type="class")
  # Error rates
  NBnPCmisclass.train <- mean(ifelse(NBnpc.pred.train$class == xi.1$Y, yes=0, no=1))
  NBnPCmisclass.test <- mean(ifelse(NBnpc.pred.test$class == xi.2$Y, yes=0, no=1))
  all.MC[i, "NB PCA ND"] =  NBnPCmisclass.test
  
  # Repeat, using kernel density estimates
  NBk.pc <- NaiveBayes(x=xi.1[,-17], grouping=xi.1[,17], usekernel=TRUE)
  NBkpc.pred.train <- predict(NBk.pc, newdata=xi.1[,-17], type="class")
  NBkpc.pred.test <- predict(NBk.pc, newdata=xi.2[,-17], type="class")
  # Error rates
  NBkPCmisclass.train <- mean(ifelse(NBkpc.pred.train$class == xi.1$Y, yes=0, no=1))
  NBkPCmisclass.test <- mean(ifelse(NBkpc.pred.test$class == xi.2$Y, yes=0, no=1))
  all.MC[i, "NB PCA KD"] =  NBkPCmisclass.test
  
  #### Trees 
  wh.tree <- rpart(data=data.train, Y ~ ., method="class", cp=0)
  # Find location of minimum error
  cpt = wh.tree$cptable
  minrow <- which.min(cpt[,4])
  # Take geometric mean of cp values at min error and one step up 
  cplow.min <- cpt[minrow,1]
  cpup.min <- ifelse(minrow==1, yes=1, no=cpt[minrow-1,1])
  cp.min <- sqrt(cplow.min*cpup.min)
  
  # Find smallest row where error is below +1SE
  se.row <- min(which(cpt[,4] < cpt[minrow,4]+cpt[minrow,5]))
  # Take geometric mean of cp values at min error and one step up 
  cplow.1se <- cpt[se.row,1]
  cpup.1se <- ifelse(se.row==1, yes=1, no=cpt[se.row-1,1])
  cp.1se <- sqrt(cplow.1se*cpup.1se)
  
  # Creating a pruned tree using a selected value of the CP by CV.
  wh.prune.cv.1se <- prune(wh.tree, cp=cp.1se)
  # Creating a pruned tree using a selected value of the CP by CV.
  wh.prune.cv.min <- prune(wh.tree, cp=cp.min)
  # Predict results of classification. "Vector" means store class as a number
  pred.train.cv.1se <- predict(wh.prune.cv.1se, newdata=data.train, type="class")
  pred.train.cv.min <- predict(wh.prune.cv.min, newdata=data.train, type="class")
  pred.train.full <- predict(wh.tree, newdata=data.train, type="class")
  
  # Predict results of classification. "Vector" means store class as a number
  pred.test.cv.1se <- predict(wh.prune.cv.1se, newdata=data.valid, type="class")
  pred.test.cv.min <- predict(wh.prune.cv.min, newdata=data.valid, type="class")
  pred.test.full <- predict(wh.tree, newdata=data.valid, type="class")
  
  misclass.train.cv.1se <- mean(ifelse(pred.train.cv.1se == data.train$Y, yes=0, no=1))
  misclass.train.cv.min <- mean(ifelse(pred.train.cv.min == data.train$Y, yes=0, no=1))
  misclass.train.full <- mean(ifelse(pred.train.full == data.train$Y, yes=0, no=1))
  
  misclass.test.cv.1se <- mean(ifelse(pred.test.cv.1se == data.valid$Y, yes=0, no=1))
  misclass.test.cv.min <- mean(ifelse(pred.test.cv.min == data.valid$Y, yes=0, no=1))
  misclass.test.full <- mean(ifelse(pred.test.full == data.valid$Y, yes=0, no=1))
  all.MC[i, "Full Tree"] =  misclass.test.full
  all.MC[i, "TreeCV"] = misclass.test.cv.min
  all.MC[i, "Tree1se"] = misclass.test.cv.1se
  
  ####Random Forests###
  wh.rf <- randomForest(data=data.train, Y~., 
                        importance=TRUE, keep.forest=TRUE)
  # Predict results of classification. 
  pred.rf.train <- predict(wh.rf, newdata=data.train, type="response")
  pred.rf.test <- predict(wh.rf, newdata=data.valid, type="response")
  #"vote" gives proportions of trees voting for each class
  pred.rf.vtrain <- predict(wh.rf, newdata=data.train, type="vote")
  pred.rf.vtest <- predict(wh.rf, newdata=data.valid, type="vote")
  misclass.train.rf <- mean(ifelse(pred.rf.train == data.train$Y, yes=0, no=1))
  misclass.test.rf <- mean(ifelse(pred.rf.test == data.valid$Y, yes=0, no=1))
  all.MC[i, "RFD"] = misclass.test.cv.1se
  
  ##Tuning 
  set.seed(879417)
  reps=5
  varz = 1:16
  nodez = c(2,3,4,5,6,7,8)
  
  NS = length(nodez)
  M = length(varz)
  rf.oob = matrix(NA, nrow=M*NS, ncol=reps)
  
  for(r in 1:reps){
    print(paste0(r, " of ", reps))
    counter=1
    for(m in varz){
      for(ns in nodez){
        wh.rfm <- randomForest(data=data.train, Y~., 
                               mtry=m, nodesize=ns)
        rf.oob[counter,r] = mean(predict(wh.rfm, type="response") != data.train$Y)
        counter=counter+1
      }
    }
  }
  
  parms = expand.grid(nodez,varz)
  row.names(rf.oob) = paste(parms[,2], parms[,1], sep="|")
  
  mean.oob = apply(rf.oob, 1, mean)
  a <- as.matrix(mean.oob[order(mean.oob)])
  min.oob = apply(rf.oob, 2, min)
  #boxplot(rf.oob, use.cols=FALSE, las=2)
  #boxplot(t(rf.oob)/min.oob, use.cols=TRUE, las=2, 
         # main="RF Tuning Variables and Node Sizes")
  chosenP <- row.names(a)[1]
  toStr <- str_split(chosenP, "|")
  vars <- strtoi(toStr[[1]][2])
  NS <- strtoi(toStr[[1]][4])
  all.MC[i, "RF V"] <- vars
  all.MC[i, "RF NS"] <- NS
  
  wh.rf.tun <- randomForest(data=data.train, Y~., mtry=vars, nodesize=NS,
                            importance=TRUE, keep.forest=TRUE)
  # Predict results of classification. 
  pred.rf.train.tun <- predict(wh.rf.tun, newdata=data.train, type="response")
  pred.rf.test.tun <- predict(wh.rf.tun, newdata=data.valid, type="response")
  #"vote" gives proportions of trees voting for each class
  pred.rf.vtrain.tun <- predict(wh.rf.tun, newdata=data.train, type="vote")
  pred.rf.vtest.tun <- predict(wh.rf.tun, newdata=data.valid, type="vote")
  #head(cbind(pred.rf.test.tun,pred.rf.vtest.tun))
  
  misclass.train.rf.tun <- mean(ifelse(pred.rf.train.tun == data.train$Y, yes=0, no=1))
  misclass.test.rf.tun <- mean(ifelse(pred.rf.test.tun == data.valid$Y, yes=0, no=1))
  all.MC[i, "RF Tuned"] <- misclass.test.rf.tun
  
  #### SVM ###
  svm.1.1 <- svm(data=data.train, Y ~ ., kernel="radial", 
                  gamma=1, cost=1, cross=10)
  pred1.1.1 <- predict(svm.1.1, newdata=data.train)
  # #table(pred1.1.1, set1$type,  dnn=c("Predicted","Observed"))
  misclass1.1.1 <- mean(ifelse(pred1.1.1 == data.train$Y, yes=0, no=1))
   
  pred2.1.1 <- predict(svm.1.1, newdata=data.valid)
  # #table(pred2.1.1, data.valid$type,  dnn=c("Predicted","Observed"))
  misclass2.1.1 <- mean(ifelse(pred2.1.1 == data.valid$Y, yes=0, no=1))
  all.MC[i, "SVM"] <- misclass2.1.1 
  
  #############################################
  # Try tuning with caret::train
  trcon = trainControl(method="repeatedcv", number=10, repeats=2,
                       returnResamp="all")
  parmgrid = expand.grid(C=10^c(0:5), sigma=10^(-c(5:0)))
  
  tuned.nnet <- train(x=data.train[,-1], y=data.train$Y, method="svmRadial", 
                      preProcess=c("center","scale"), trace=FALSE, 
                      tuneGrid=parmgrid, trControl = trcon)
  
  names(tuned.nnet)
  tuned.nnet$results[order(-tuned.nnet$results[,3]),]
  tuned.nnet$bestTune
  head(tuned.nnet$resample)
  tail(tuned.nnet$resample)
  
  # Let's rearrange the data so that we can plot the bootstrap resamples in 
  #   our usual way, including relative to best
  resamples = reshape(data=tuned.nnet$resample[,-2], idvar=c("C", "sigma"), 
                      timevar="Resample", direction="wide")
  head(resamples)
  (best = apply(X=resamples[,-c(1,2)], MARGIN=2, FUN=max))
  C.sigma <- paste(log10(resamples[,1]),"-",log10(resamples[,2]))
  #boxplot.matrix(x=t(t(1-resamples[,-c(1:2)])), use.cols=FALSE, names=C.sigma,
                 #main="Misclassification rates for different Cost-Gamma", las=2)
  #boxplot.matrix(x=t(t(1-resamples[,-c(1:2)])/(1-best)), use.cols=FALSE, names=C.sigma,
                 #main="Relative Misclass rates for different Cost-Gamma", las=2)
  #boxplot(t(t(1-resamples[,-c(1:2)])/(1-best)) ~ resamples[,1], xlab="C", ylab="Relative Error")
  #boxplot(t(t(1-resamples[,-c(1:2)])/(1-best)) ~ resamples[,2], xlab="Sigma", ylab="Relative Error")
  mean.svm = as.matrix(apply(t(t(1-resamples[,-c(1:2)])/(1-best)), 1, mean))
  row.names(mean.svm) = C.sigma
  indexMin <- which.min(mean.svm)
  combin <- rownames(mean.svm)[indexMin]
  toStrSVM <- str_split(combin, "-")
  C <- strtoi(substring(toStrSVM[[1]][1], 1, 1))
  sigma <- strtoi(toStrSVM[[1]][3])
  
  svm.wh.tun <- svm(data=data.train, Y ~ ., kernel="radial", 
                    gamma=10^(-sigma), cost=10^(C))
  pred1.wh.tun <- predict(svm.wh.tun, newdata=data.train)
  misclass1.wh.tun <- mean(ifelse(pred1.wh.tun == data.train$Y, yes=0, no=1))
  pred2.wh.tun <- predict(svm.wh.tun, newdata=data.valid)
  misclass2.wh.tun <- mean(ifelse(pred2.wh.tun == data.valid$Y, yes=0, no=1))
  all.MC[i, "SVMT"] <- misclass2.wh.tun
  
  # ##NNEt
  # X.train.unscaled.NN <- as.matrix(X.train)
  # X.train.scaled.NN <- rescale(X.train.unscaled.NN, X.train.unscaled.NN)
  # #Prove that it worked
  # apply(X=X.train.scaled.NN, MARGIN=2, FUN=min)
  # apply(X=X.train.scaled.NN, MARGIN=2, FUN=max)
  # 
  # X.valid.unscaled.NN <- as.matrix(X.valid)
  # X.valid.scaled.NN <- rescale(X.valid.unscaled.NN, X.train.unscaled.NN)
  # apply(X=X.valid.scaled.NN, MARGIN=2, FUN=min)
  # apply(X=X.valid.scaled.NN, MARGIN=2, FUN=max)
  # y.train.NN <- class.ind(data.train[,1])
  # y.valid.NN <- class.ind(data.valid[,1])
  # ###########################################################################
  # # nnet{nnet} can do regression or classification.
  # ### Fit to set 1, Test on Set 2
  # nn.1.0 <- nnet(x=X.train.scaled.NN, y=y.train.NN, size=1, maxit=1000, softmax=TRUE)
  # # Train error
  # p1.nn.1.0m <-predict(nn.1.0, newdata=X.train.scaled.NN, type="raw")
  # #round(head(p1.nn.1.0m), 3)
  # p1.nn.1.0 <-predict(nn.1.0, newdata=X.train.scaled.NN, type="class")
  # #table(p1.nn.1.0, as.factor(set1$type),  dnn=c("Predicted","Observed"))
  # misclass1.1.0 <- mean(ifelse(p1.nn.1.0 == data.train$Y, yes=0, no=1))
  # 
  # # Test set error
  # p2.nn.1.0 <-predict(nn.1.0, newdata=X.valid.scaled.NN, type="class")
  # table(p2.nn.1.0, as.factor(data.valid$Y),  dnn=c("Predicted","Observed"))
  # misclass2.1.0 <- mean(ifelse(p2.nn.1.0 == data.valid$Y, yes=0, no=1))
  # all.MC[i, "NNetD"] <- misclass2.1.0
  # 
  # #For simplicity, rename data as "train.x" and "train.y"
  # train.x = data.train[,-1]
  # train.y.class = data.train[,1] 
  # train.y = class.ind(train.y.class)
  # 
  # #  Let's do R=2 reps of V=10-fold CV.
  # set.seed(74375641)
  # V=10
  # R=2 
  # n2 = nrow(train.x)
  # # Create the folds and save in a matrix
  # foldsNN = matrix(NA, nrow=n2, ncol=R)
  # for(r in 1:R){
  #   foldsNN[,r]=floor((sample.int(n2)-1)*V/n2) + 1
  # }
  # 
  # # Grid for tuning parameters and number of restarts of nnet
  # siz <- c(1,3,4,5,6,7,10)
  # dec <- c(0.0001,0.001,0.002, 0.003)
  # nrounds=10
  # 
  # # Prepare matrix for storing results: 
  # #   row = 1 combination of tuning parameters
  # #   column = 1 split
  # #   Add grid values to first two columns
  # 
  # Mis.cv = matrix(NA, nrow=length(siz)*length(dec), ncol=V*R+2)
  # Mis.cv[,1:2] = as.matrix(expand.grid(siz,dec))
  # cl <- makeCluster(3, type = "SOCK")
  # registerDoSNOW(cl)
  # foreach(j=1:nrounds) %dopar% {
  #   # Start loop over all reps and folds.  
  #   for (r in 1:R){ 
  #     for(v in 1:V){
  #       
  #       y.1 <- as.matrix(train.y[foldsNN[,r]!=v,])
  #       x.1.unscaled <- as.matrix(train.x[foldsNN[,r]!=v,]) 
  #       x.1 <- rescale(x.1.unscaled, x.1.unscaled) 
  #       
  #       #Test
  #       y.2 <- as.matrix(train.y[foldsNN[,r]==v],)
  #       x.2.unscaled <- as.matrix(train.x[foldsNN[,r]==v,]) # Original data set 2
  #       x.2 = rescale(x.2.unscaled, x.1.unscaled)
  #       
  #       # Start counter to add each model's misclassification to row of matrix
  #       qq=1
  #       # Start Analysis Loop for all combos of size and decay on chosen data set
  #       for(d in dec){
  #         for(s in siz){
  #           
  #           ## Restart nnet nrounds times to get best fit for each set of parameters 
  #           Mi.final <- 1
  #           print(paste0(j ," of ", nrounds))
  #           nn <- nnet(y=y.1, x=x.1, size=s, decay=d, maxit=2000, softmax=TRUE, trace=FALSE)
  #           Pi <- predict(nn, newdata=x.1, type ="class")
  #           Mi <- mean(Pi != as.factor(data.train[foldsNN[,r]!=v,1]))
  #           
  #           if(Mi < Mi.final){ 
  #             Mi.final <- Mi
  #             nn.final <- nn
  #           }
  #           pred.nn = predict(nn.final, newdata=x.2, type="class")
  #           Mis.cv[qq,(r-1)*V+v+2] = mean(pred.nn != as.factor(train.y.class[foldsNN[,r]==v]))
  #           qq = qq+1
  #         }
  #       }
  #     }
  #   }
  # }
  # stopCluster(cl)
  # 
  # Mis.cv
  # 
  # (Micv = apply(X=Mis.cv[,-c(1,2)], MARGIN=1, FUN=mean))
  # (Micv.sd = apply(X=Mis.cv[,-c(1,2)], MARGIN=1, FUN=sd))
  # Micv.CIl = Micv - qt(p=.975, df=R*V-1)*Micv.sd/sqrt(R*V)
  # Micv.CIu = Micv + qt(p=.975, df=R*V-1)*Micv.sd/sqrt(R*V)
  # (all.cv = cbind(Mis.cv[,1:2],round(cbind(Micv,Micv.CIl, Micv.CIu),2)))
  # all.cv[order(Micv),]
  # 
  # siz.dec <- paste("NN",Mis.cv[,1],"-",Mis.cv[,2])
  # lowt = apply(Mis.cv[,-c(1,2)], 2, min)
  # relMi = t(Mis.cv[,-c(1,2)])/lowt
  # (RRMi = apply(X=relMi, MARGIN=2, FUN=mean))
  # (RRMi.sd = apply(X=relMi, MARGIN=2, FUN=sd))
  # RRMi.CIl = RRMi - qt(p=.975, df=R*V-1)*RRMi.sd/sqrt(R*V)
  # RRMi.CIu = RRMi + qt(p=.975, df=R*V-1)*RRMi.sd/sqrt(R*V)
  # (all.rrcv = cbind(Mis.cv[,1:2],round(cbind(RRMi,RRMi.CIl, RRMi.CIu),2)))
  # all.rrcv[order(RRMi),]
  # 
  # x.1.unscaled <- as.matrix(data.train[,-1])
  # x.2.unscaled <- as.matrix(data.valid[,-1])
  # x.1 <- rescale(x.1.unscaled, x.1.unscaled)
  # x.2 <- rescale(x.2.unscaled, x.1.unscaled)
  # 
  # y.1 <- class.ind(data.train[,1])
  # y.2 <- class.ind(data.valid[,1])
  # sizeP <- all.rrcv[order(RRMi),][1,][1]
  # all.MC[i, "NNet Node"] <- sizeP
  # DecayP <- all.rrcv[order(RRMi),][1,][2]
  # all.MC[, "NNet Decay"] <- DecayP
  # Mi.final = 1
  # for(i in 1:10){
  #   nn <- nnet(y=y.1, x=x.1, size=sizeP, decay=DecayP, maxit=2000, softmax=TRUE, trace=FALSE)
  #   Pi <- predict(nn, newdata=x.1, type="class")
  #   Mi <- mean(Pi != as.factor(data.train[,1]))
  #   
  #   if(Mi < Mi.final){ 
  #     Mi.final <- Mi
  #     nn.final <- nn
  #   }
  # }
  # 
  # # Train error
  # p1.nn.3.01 <-predict(nn.final, newdata=x.1, type="class")
  # misclass1.3.01 <- mean(ifelse(p1.nn.3.01 == data.train$Y, yes=0, no=1))
  # 
  # # Test set error
  # p2.nn.3.01 <-predict(nn.final, newdata=x.2, type="class")
  # misclass2.3.01 <- mean(ifelse(p2.nn.3.01 == data.valid$Y, yes=0, no=1))
  # #table(p2.nn.3.01, as.factor(data.valid$Y),  dnn=c("Predicted","Observed"))
  # all.MC[i, "NNetT"] <- misclass2.3.01
}
all.RMC = apply(all.MC[,c(-3,-5,-12,-21, -22, -25, -26, -27, -28)], 1, function(W){
  best = min(W)
  return(W / best)
})
all.RMC = t(all.RMC)

boxplot(all.MC[,c(-3,-5,-12,-21, -22, -25, -26, -27, -28)], main = paste0("CV MC over ", K, " folds"), las=2)
boxplot(all.RMC[,c(-3,-5,-12,-21, -22, -25, -26, -27, -28)], ylim = c(1, 4.5),
        main = paste0("CV RMCs over ", K, 
                      " folds (enlarged to show texture)"), las=2)
