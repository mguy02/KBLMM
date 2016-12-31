#Guy Martin
#

######

#Importing libraries

library(pls)
library(nnet) #for class.ind

library(gplots) #for heatmap (confusion matrix)

######



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

Y.train <- class.ind(train[,1])
Y.test <- class.ind(test[,1])



#Center the predictor matrix.
X.train.means <- colMeans(X.train)
X.train <-scale(X.train, scale=FALSE)
X.test <- scale(X.test, center = X.train.means, scale=FALSE)



# 3. Perform a PLSR2 using "CV" or "LOO" for validation.
#Decide how many components you retain for prediction?


pls1 <- plsr(Y.train ~ X.train, validation="CV")


# Plot the components importance
plot(R2(pls1), legendpos = "bottomleft", main = 'R2 vs Number of Components',xlab = 'Number of Components', ylab = 'Squared Error')


# We observe there is is getting worse as we take more and more components. Let's zoom to 30 components.
# Then we can zoom more

pls2 <- plsr(Y.train ~ X.train, ncomp=30, validation="CV")


R2.cv <- R2(pls2)$val[1,,]
#head(R2.cv)

R2.cv.mean <- apply(R2.cv,2,mean)
R2.cv.mean

plot(R2.cv.mean,type="l",xaxt="n")
axis(1,at=1:ncol(R2.cv),lab=colnames(R2.cv),tick=FALSE)

nd <- which.max(R2.cv.mean)-1
abline(v = nd, lty=2, col="blue")
print(paste("Number of selected components:", nd))


#Then, we select 18 components here as the mean of the R2 value is the greatest.

#Let's see how much of the variance is explained with this number of components
var.exp <- rep(0, nd)
curr <- 0

for(i in 1:nd)
{
  curr <- curr + pls2$Xvar[i]
  var.exp[i] = curr/pls2$Xtotvar
}

var.exp <- var.exp*100

print(paste("Variance explained with", nd,"components:", round(var.exp[nd]*100)/100))


### SEE IF USE BELOW ###

# scores plot
plot(pls2, plottype = "scores", comps = 1:2, type="n", main="X Scores")
text(pls2$scores, labels=rownames(pls2$scores), col=as.numeric(as.factor(1:10)))
axis(side=1, pos= 0, labels = F, col="gray")
axis(side=3, pos= 0, labels = F, col="gray")
axis(side=2, pos= 0, labels = F, col="gray")
axis(side=4, pos= 0, labels = F, col="gray")
legend("topright",c(levels(as.factor(0:9))),col=c(1:10),lty=1)

print(paste("Variance explained with", 2,"components:", round(var.exp[2]*100)/100))



# correlation plot

# plot of correlations

n <- nrow(X.train)
p <- ncol(X.train)
q <- ncol(Y.train)

corXp2 <- cor(X.train,pls2$scores)
corYp2 <- cor(Y.train,pls2$scores)
corXYp2 <- rbind(corXp2,corYp2)
plot(corXYp2,ylim=c(-1,1),xlim=c(-1,1),asp=1,type="n",main="Correlations with components")
text(corXYp2,labels=rownames(corXYp2),col=c(rep(1,p),rep(2,q)),adj=1.1,cex=0.85)
arrows(rep(0,(p+1)),rep(0,(p+1)),corXYp2[,1],corXYp2[,2],col=c(rep(1,p),rep(2,q)),length=0.07)
axis(side=1,pos=0,labels=F,col="gray")
axis(side=2,pos=0,labels=F,col="gray")
axis(side=3,pos=0,labels=F,col="gray")
axis(side=4,pos=0,labels=F,col="gray")
circle()


### SEE IF USE ABOVE ###



#4. Predict the responses in the test data, be aware of the appropriate
#centering. 

test.proj <- as.matrix(X.test) %*% pls2$projection[, 1:nd]


train.pls.data <- data.frame(pls2$scores[,1:nd])


model <- lm(Y.train~., data=train.pls.data)
PRESS  <- apply((model$residuals/(1-ls.diag(model)$hat))^2,2,sum)
R2cv   <- 1 - PRESS/(sd(Y.train)^2*(n-1))
print("R2 on training:")
print(R2cv, digit=4)


pred.test.prob <- predict(model, data.frame(test.proj) , type="response")


### kable(data, format = "markdown", align='l')

pr <- resid(model)/(1 - lm.influence(model)$hat)
(PRESS <- sum(pr^2))


#Compute the average R2 in the test data.
test_scale <- data.frame(X.test,Y.test)
Yhat = predict(model,test_scale)
RSS = colSums((Y.train-Yhat)^2)
TSS = apply(Y.train,2,function(x){sum((x-mean(x))^2)})
r2 = mean(1 - (RSS/TSS))
r2

#5. Assign every test individual to the maximum response and compute the error rate.

pred.test.numbers <- c()

n.test <- nrow(pred.test.prob)

for(i in 1:n.test)
{
  pred.test.numbers[i] <- unname(which.max(pred.test.prob[i,])-1)
}


(pred.acc <- mean(pred.test.numbers == test[,1])) #accuracy
(pred.err <- 1-pred.acc) #error rate

#confusion matrix
cf.m <- table(pred.test.numbers, test[,1])
mm <- as.matrix(cf.m)
my_palette <- colorRampPalette(c("blue", "red"))(n = 351)

heatmap.2(x = mm, Rowv = FALSE, Colv = FALSE, dendrogram = "none",
          cellnote = mm, notecol = "black", notecex = 1,
          trace = "none", key = FALSE, margins = c(7, 11),
          ylab="Predicted", xlab="Observation",
          symm=F, symkey=F, symbreaks=T, scale="none",
          col=my_palette )

