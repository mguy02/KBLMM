#Guy Martin

library(pls)
library(calibrate)
library(FactoMineR)

set.seed(1337)

#1. Read the "zip_train.dat" and "zip_test.dat" files provided.

train.full <- read.table("zip_train.dat")
test <- read.table("zip_test.dat")

#Select a 5% random sample (without replacement) of the train data
proportion <- 0.05
n.full <- dim(train.full)[1]  #original size of the training set

n <- round(n.full * proportion) #new size of the training set


train <- train.full[sample(1:n.full,n),]

#2. Define the response matrix (Y) and the predictor matrix (X).


X.train <- train[,-1]
X.test <- test[,-1]

head(X.test)
#n is the number of elements in our training set
#n.test is the number of elements in our test set
#p is the number of explanatory variables for X
#q is the number of explanatory variables for Y

n <- nrow(X.train)
n.test <- nrow(X.test)
p <- ncol(X.train)
q <- 10

Y.train <- matrix(0, n, q)

for(i in 1:n)
{
  index <- train[i,1]+1
  Y.train[i, index] <- 1
}

Y.test <- matrix(0, n.test, q)

for(i in 1:n.test)
{
  index <- test[i,1]+1
  Y.test[i, index] <- 1
}


#Center the predictor matrix.
X.train.s <- as.matrix(scale(X.train, center = TRUE, scale=FALSE))
Y.train.s <- as.matrix(Y.train)



#3. Perform a multivariate regression with the training data.
mreg <- lm(Y.train.s ~ X.train.s)

summary.mreg <- summary(mreg)

#Compute the average R2.
avgR2 <- mean(sapply(summary.mreg, function(m) m$r.squared))



#4. Compute the average of the R2 by Leave One Out.
R2cv.values <- rep(0, n)

for(i in 1:n)
{
  Xcv <- X.train.s[-i,]
  Ycv <- Y.train.s[-i,]
  mreg.cv <- lm(Ycv ~ Xcv)
  
  R2cv.values[i] <- mean(sapply(summary(mreg.cv), function(m) m$r.squared))
}

(R2cv <- mean(R2cv.values))

R2cv

#5. Predict the responses in the test data, be aware of the appropriate
#centering. 
X.test.s <- scale(X.test, center = TRUE, scale=FALSE)

Y.pred.mreg.matrix <- as.matrix(X.test.s) %*% mreg$coefficients[-1,]


#Compute the average R2 in the test data.
test_scale <- data.frame(X.train.s,Y.train.s)
Yhat = predict(mreg,test_scale)
RSS = colSums((Y.train.s-Yhat)^2)
TSS = apply(Y.train.s,2,function(x){sum((x-mean(x))^2)})
r2 = mean(1 - (RSS/TSS))
r2



#6. Assign every test individual to the maximum response and compute the error rate.

dim(Y.pred.mreg.matrix)

head(Y.pred.mreg.matrix)

Y.pred.mreg.numbers <- c()

for(i in 1:n.test)
{
  Y.pred.mreg.numbers[i] <- which.max(Y.pred.mreg.matrix[i,])-1
}


(mreg.acc <- mean(Y.pred.mreg.numbers == test[,1])) #accuracy
(mreg.err <- 1-mreg.acc) #error rate


#7. Perform a PCR (using LOO). Decide how many components you retain for prediction.

nd <- 100 #number of components

pc <- pcr(Y.train.s~X.train.s, validation="LOO", ncomp=nd)

#8.1 Predict the responses in the test data

Y.pred.pcr.matrix <- as.matrix(X.test.s) %*% pc$coefficients[,,nd]

#8.2 Assign every test individual to the maximum response and compute the error rate.


Y.pred.pcr.numbers <- c()

for(i in 1:n.test)
{
  Y.pred.pcr.numbers[i] <- which.max(Y.pred.pcr.matrix[i,])-1
}


(pcr.acc <- mean(Y.pred.pcr.numbers == test[,1])) #accuracy
(pcr.err <- 1-pcr.acc) #error rate
