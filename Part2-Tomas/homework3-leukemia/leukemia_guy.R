#Guy Martin
#
# Although cancer classification has improved over the past 30 years, there has
# been no general approach for identifying new cancer classes (class discovery) or for
# assigning tumors to known classes (class prediction). The problem focuses in finding a
# classifier using PLS1 regression approach on gene expression monitoring by DNA
# microarrays, to automatically differentiate between acute myeloid leukemia (AML) and
# acute lymphoblastic leukemia (ALL)

# We will work on two datasets : the training set \texttt{data_set_ALL_AML_train.csv}
# with 38 samples and the test set \texttt{data_set_ALL_AML_independent.csv} with
# 34 samples

#These datasets contain measurements corresponding to ALL and AML samples from
#Bone Marrow and Peripheral Blood.

######

#Importing libraries

library(pls)
library(dplyr)  #for select() function

library(SDMTools) # for creating confusion matrix
library(ggplot2) #for ploting confusion matrix


######

# 1. Read the data files "data_set_ALL_AML_train.csv" and
# "data_set_ALL_AML_independent.csv".

train <- read.csv("data_set_ALL_AML_train.csv", header = TRUE, sep = ";")
test <- read.csv("data_set_ALL_AML_independent.csv", header = TRUE, sep = ";")




#Enter the response manually into R
# from the "table_ALL_AML_predic.doc" document (the response corresponds to
# the "actual" column of such word document).

# We will map 
#   ALL -> 0
#   AML -> 1
#In this file, there are 27 ALL in the training data and 20 in the testing data.
#The rest is AML.

train.response <- append(rep(0,27), rep(1,11))
test.response <- append(rep(0,20), rep(1,14))


#Only numeric information is pertinent to solve the problem.
train <- select(train, contains("X"))
test <- select(test, contains("X"))


# 2. Form the data matrices X and Xt, containing the gene expression for the 38
# training samples and 34 test samples.

X <- data.frame(t(train))
Xt <- data.frame(t(test))

rownames(X)
rownames(Xt)

# We can observe that the labels are not exactly in order, but no need to reorder.
# Indeed, it corresponds well to the table_ALL_AML_predic.doc file.


# We center but not scale the data
means = colMeans(X)
X <- data.frame(scale(X,scale = F))
Xt <- data.frame(scale(Xt,center=means,scale=F))

# Trainging and Testing dataset
training <- as.data.frame(cbind(train.response,X))
testing <- as.data.frame(cbind(test.response,Xt))


# 3. Perform the PLS1 regression of the training data. Select the number of PLS1
# components.

pls1 <- plsr(train.response ~ ., data = training, validation="LOO")


# Plot the components importance
plot(RMSEP(pls1), legendpos = "topright", main = 'RMSE vs Number of Components',xlab = 'Number of Components', ylab = 'Root Mean Squared Error')

# We observe there is not much improvement if we take more than 10 components.
# Then we can zoom more

pls2 <- plsr(train.response ~ ., ncomp=10, data = training, validation="LOO")
plot(RMSEP(pls2), ncomp, legendpos = "topright", main = 'RMSE vs Number of Components',xlab = 'Number of Components', ylab = 'Root Mean Squared Error')

#We see that after taking 4 components, the RMSE does not decrease that much if we keep
#taking more components

ncomp <- 4
abline(v = ncomp, lty=2, col="blue")


# 4. Project the test data as supplementary individuals onto the selected PLS1
# components (be aware of centering the test data respect to the mean of the
# training data).

test.proj <- as.matrix(Xt) %*% pls2$projection[, 1:ncomp]



# 5. Perform a joint plot of the train and test individuals in the plane of the two first
# PLS1 components, differentiating those individuals with ALL from those with
# AML.


plot(pls2$scores[, 1:2], col = as.factor(train.response))
points(test.proj[,1:2], pch=17, col=as.factor(test.response))
legend("topright",c("ALL","AML","ALL(test)","AML(test)"), pch=c(1,1,17,17),col=c("black","red","black","red"))


#On the training data, ALL and AML are well separated. On the testing data they are less
#separated.

R2(pls2)

#And in fact, we see that we do not explain most of the variance that we can as we only
#use two components.



# 6. Obtain the logistic regression model (or any other model of your choice) to
# predict the response in the training data, using as input the selected PLS1
# components.

train.pls.data <- data.frame(pls2$scores[,1:4])


model <- glm(as.factor(train.response) ~ .,family=binomial(link='logit'),data=train.pls.data)

pred.train <- unname(predict(model, train.pls.data))
pred.train <- as.numeric(pred.train > 0.5)

acc.train <- sum(pred.train==train.response)/length(pred.train)

paste("Accuracy on training data:", acc.train)

# We have an accuracy of 100% on the training data. While this is a perfect score, this
# may be overfitting. It is more accurate then to see the accuracy on test data.

# 7. Predict the probability of AML leukemia in the test sample.

pred.test.prob <- predict(model, data.frame(test.proj) , type="response")
pred.test <- unname(pred.test.prob)
pred.test <- as.numeric(pred.test > 0.5)

acc.test <- sum(pred.test==test.response)/length(pred.test)

paste("Accuracy on testing data:", acc.test)

cf.m <- confusion.matrix(test.response, pred.test)


obs <- factor(c(0, 0, 1, 1))
pred <- factor(c(0, 1, 0, 1))
Y      <- c(cf.m[1], cf.m[2], cf.m[3], cf.m[4])

df <- data.frame(obs, pred, Y)

ggplot(data =  df, mapping = aes(x = obs, y = pred)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_bw() + theme(legend.position = "none")

#Predict the probability of AML (which is mapped to 1)
result <- data.frame(cbind(test.response,pred.test.prob))
result <- result[result$test.response==1,]
result$test.response <- rep("AML", length(result$test.response))
result$pred.test.prob <- round(result$pred.test.prob, 4)

indexes.AML <- c("X50", "X51", "X52", "X53", "X54", "X57", "X58", "X60", "X61", "X62", 
             "X63", "X64", "X65", "X66")

paperProbAML <- read.table("paperProbAML.txt")
names(paperProbAML) <- "PS" #prediciton strength

result <- result[indexes.AML,]
result$paperProb <- paperProbAML
result

