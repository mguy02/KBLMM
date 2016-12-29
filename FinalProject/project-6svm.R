#
#Martin Guy
#Hannes Leskela
#December, 2016
#Final Project KBLMM Course - UPC
#
#Data : "Grasp-and-Lift EEG Detection" Kaggle dataset
#
#This is the 6-SVM approach. We train 6 SVM independently on each classes.
#
#

set.seed(42)

source("utility-common.R")
source("utility-6svm.R")


########
#######



dataFolder <- "data"

dataset.1 <- load.data(dataFolder, numSubject=1)
dataset.2 <- load.data(dataFolder, numSubject=2)
dataset.3 <- load.data(dataFolder, numSubject=3)


training.ratio <- 0.60


data.1 <- splitData(dataset.1, training.ratio)
data.2 <- splitData(dataset.2, training.ratio)
data.3 <- splitData(dataset.3, training.ratio)

#############################
# SELECTING THE C PARAMETER #
#############################

#Note: the functions are in comment to not execute them on sourcing, as we already found the C parameter

#res.1 <- compare.c.all(data.1$X_train, data.1$labels_train, C=c(600, 1000))
#The time between C=600 and C=1000 for all models sums up around one minute, and for 4 classes
#the accuracy is better so we decide to take C=1000.

#res.2 <- compare.c.all(data.2$X_train, data.2$labels_train, C=c(0.1, 1, 100, 300, 600, 1000))

#res.3 <- compare.c.all(data.3$X_train, data.3$labels_train, C=c(0.1, 1, 100, 300, 600, 1000))
#We can observe that C=1000 increase the accuracy on training compared to C=600 for all classes (by 2 points)
#except for Replace which lose 1 point. However we still select C=1000 as it is better for other classes.

######################
# MODELS COMPUTATION #
######################

clf_list.1 <- create.svms(data.1$X_train, data.1$labels_train, C=1000)
clf_list.2 <- create.svms(data.2$X_train, data.2$labels_train, C=1000)
clf_list.3 <- create.svms(data.3$X_train, data.3$labels_train, C=1000)


## Check accuracy on training
(acc.train.1 <- 1-mean(sapply(clf_list.1, function(x) error(x))))
(acc.train.2 <- 1-mean(sapply(clf_list.2, function(x) error(x))))
(acc.train.3 <- 1-mean(sapply(clf_list.3, function(x) error(x))))



###################################
# TESTING THE MODELS ON TEST DATA #
###################################

t_pred.1 <- pred.svms(clf_list.1, data.1$X_test)
t_pred.2 <- pred.svms(clf_list.2, data.2$X_test)
t_pred.3 <- pred.svms(clf_list.3, data.3$X_test)

head(t_pred.1)


## Check accuracy on testing independently for each classes
(acc.1 <- acc.svms(t_pred.1, data.1$labels_test))
(acc.2 <- acc.svms(t_pred.2, data.2$labels_test))
(acc.3 <- acc.svms(t_pred.3, data.3$labels_test))


mean(acc.1)
mean(acc.2)
mean(acc.3)

barplot(acc.1, names=1:6, ylim=c(0,1), main="Accuracy of the 6 classes (subject 1)")
barplot(acc.2, names=1:6, ylim=c(0,1), main="Accuracy of the 6 classes (subject 2)")
barplot(acc.3, names=1:6, ylim=c(0,1), main="Accuracy of the 6 classes (subject 3)")



roc.1 <- sapply(1:6, function(x) ROC(t_pred.1, data.1$labels_test, x))
roc.2 <- sapply(1:6, function(x) ROC(t_pred.2, data.2$labels_test, x))
roc.3 <- sapply(1:6, function(x) ROC(t_pred.3, data.3$labels_test, x))

plotROCs(roc.1, 1)
plotROCs(roc.2, 2)
plotROCs(roc.3, 3)

