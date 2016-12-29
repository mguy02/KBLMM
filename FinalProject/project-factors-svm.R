#
#Martin Guy
#Hannes Leskela
#December, 2016
#Final Project KBLMM Course - UPC
#
#Data : "Grasp-and-Lift EEG Detection" Kaggle dataset
#
#This is the multi-class SVM approach (using all-versus-all strategy implemented in kernlab library).
#We consider a dataset with only labels rows that have exactly one 1.
#

set.seed(42)

source("utility-common.R")
source("utility-factors.R")


########
#######



dataFolder <- "dataOnlyOne1-sub5"

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

#res.1 <- compare.c(data.1$X_train, data.1$labels_train.dec, C=c(0.1, 1, 100, 300, 600, 1000))

#C=1000 leads to the highest training accuracy (71%) . However, it takes 11 min to compute, compared to
#C=300 that takes only 2.33 min to compute with 65% accuracy on training. As fast training is important too in EEG,
#C=300 should be a better choice then.

#res.2 <- compare.c(data.2$X_train, data.2$labels_train.dec, C=c(0.1, 1, 100, 300, 600))

#for subject 2 we can select C=300 (computing in 3.98 min, accurary on training of 69,5%) 
#or C=600 (computing in 5.56 min, accuracy on training of 71,9%) 
#we  gain 2.4 points at a cost of 1.5 min
#we decided here to take C=600

#res.3 <- compare.c(data.3$X_train, data.3$labels_train.dec, C=c(1000))

#for subject 3, we have
# C             300     600     1000
# acc(%)        59,7    65,4    66,6
# time(min)     1.51    2.18    3.36
#
#From 600 to 1000 we don't gain that much of accuracy, so we will use C=600


######################
# MODELS COMPUTATION #
######################


model.1 <- create.svm(data.1$X_train, data.1$labels_train.dec, C=300)
model.2 <- create.svm(data.2$X_train, data.2$labels_train.dec, C=600)
model.3 <- create.svm(data.3$X_train, data.3$labels_train.dec, C=600)


## Check accuracy on training
(acc.train.1 <- 1-error(model.1))
(acc.train.2 <- 1-error(model.2))
(acc.train.3 <- 1-error(model.3))



###################################
# TESTING THE MODELS ON TEST DATA #
###################################

t_pred.1 <- pred.svm(model.1, data.1$X_test)
t_pred.2 <- pred.svm(model.2, data.2$X_test)
t_pred.3 <- pred.svm(model.3, data.3$X_test)

#we can see the proportion of predicted classes
table(t_pred.1)
table(data.1$labels_test.dec)

## Check accuracy on testing (exact labels)
(acc.1 <- acc.svm(t_pred.1, data.1$labels_test.dec))
(acc.2 <- acc.svm(t_pred.2, data.2$labels_test.dec))
(acc.3 <- acc.svm(t_pred.3, data.3$labels_test.dec))



#convert predictions in binary to compute the accuracy independently on each classes
t_pred.1.bin <- decdf.to.bindf(t_pred.1)
t_pred.2.bin <- decdf.to.bindf(t_pred.2)
t_pred.3.bin <- decdf.to.bindf(t_pred.3)



## Check accuracy on testing independently for each classes (like the 6SVM approach)
(acc.1.bin <- sapply(1:6, function (x) acc.classes(t_pred.1.bin, data.1$labels_test.bin, x)))
(acc.2.bin <- sapply(1:6, function (x) acc.classes(t_pred.2.bin, data.2$labels_test.bin, x)))
(acc.3.bin <- sapply(1:6, function (x) acc.classes(t_pred.3.bin, data.3$labels_test.bin, x)))


mean(acc.1.bin)
mean(acc.2.bin)
mean(acc.3.bin)


barplot(acc.1.bin, names=1:6, ylim=c(0,1), main="Accuracy of the 6 classes (subject 1)")
barplot(acc.2.bin, names=1:6, ylim=c(0,1), main="Accuracy of the 6 classes (subject 2)")
barplot(acc.3.bin, names=1:6, ylim=c(0,1), main="Accuracy of the 6 classes (subject 3)")



roc.1.bin <- sapply(1:6, function(x) ROC(t_pred.1.bin, data.1$labels_test.bin, x))
roc.2.bin <- sapply(1:6, function(x) ROC(t_pred.2.bin, data.2$labels_test.bin, x))
roc.3.bin <- sapply(1:6, function(x) ROC(t_pred.3.bin, data.3$labels_test.bin, x))

plotROCs(roc.1.bin, 1)
plotROCs(roc.2.bin, 2)
plotROCs(roc.3.bin, 3)
