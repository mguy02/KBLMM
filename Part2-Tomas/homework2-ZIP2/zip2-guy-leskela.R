#Guy Martin
#Hannes Leskela

library(nnet)
source("interbattery.R")

set.seed(42)

#1. Read the "zip_train.dat" and "zip_test.dat" files provided.

train.full <- read.table("zip_train.dat")
test <- read.table("zip_test.dat")

#Select a 5% random sample (without replacement) of the train data
proportion <- 0.05
n.full <- dim(train.full)[1]  #original size of the training set

n <- floor(n.full * proportion) #new size of the training set


train <- train.full[sample(n.full,n),]

#2. Define the response matrix (Y) and the predictor matrix (X).


X.train <- as.matrix(train[,-1])
X.test <- as.matrix(test[,-1])

Y.train = class.ind(train[,1])
Y.test= class.ind(test[,1])



#Center the predictor matrix.
X.train.means <- colMeans(X.train)
X.train <-scale(X.train, scale=FALSE)
X.test <- scale(X.test, center = X.train.means, scale=FALSE)



#3. Perform the Inter Batteries Analysis following the formulae given
#in the slides. Be aware that Y is not of full rank.
#Decide how many components you retain for prediction?

iba = interbattery(X.train, Y.train)


#4. Predict the responses in the test data, be aware of the appropriate
#centering. 

# create the components for the test set

T.test = X.test %*% iba$A

#Find number of components by calculating accuracy
accuracy <- function(model, X, y) {
  predictions = X %*% model$coefficients
  labels = apply(predictions, 1, function(x) which.max(x)) - 1
  return(sum(labels == y) / length(labels))
}

train.acc <- c()
models <- c()
for (i in 1:iba$n_components) {
  model = lm(Y.train ~ iba$T[,1:i] - 1)
  acc = accuracy(model, iba$T[,1:i], train[,1])
  train.acc[i] = acc
  models[[i]] <- model
  print(paste("Accuracy with", i, "components:", acc))
}

(num.components <- which.max(train.acc))


plot(train.acc, xlab="Number of components", ylab="Accuracy", ylim=c(0,1), main = "Selecting the number of components")
lines(train.acc, col="blue", lwd=2)
abline(v=num.components, lty=3, col= "red")

model = models[[num.components]] #we pick the good model


#Compute the average R2 in the test data.
pr <- resid(model)/(1 - lm.influence(model)$hat)
(PRESS <- sum(pr^2))


#Predicting the responses
pred = T.test[,1:num.components] %*% model$coefficients
labels = apply(pred, 1, function(x) which.max(x)) - 1

# Calculate error rate
(acc = sum(labels == test[,1]) / length(labels))
(err = 1-acc)



