

load.data <- function (dataFolder="data", numSubject)
{
  basepath <- paste(dataFolder, "filtered_", sep="/")
  
  filename.sig <- paste(basepath, numSubject ,".csv", sep="")
  filename.lab <- paste(basepath, numSubject ,"_labels.csv", sep="")
  
  signals <- read.csv(filename.sig, header=T)
  labels <- read.csv(filename.lab, header=T)
  
  for(i in 1:6)
  {
    labels[,i] <- as.factor(labels[,i])
  }
  
  return(list(
    signals = signals,
    labels = labels
  ))
}

splitData <- function(dataset, training_ratio)
{
  data <- dataset$signals
  labels <- dataset$labels
  
  
  N <- length(data[,1])
  ntrain <- round(N*training.ratio)
  training_part <- sample(N, ntrain)
  
  X_train = data[training_part,]
  X_test <- data[-training_part,]
  
  labels_train <- labels[training_part,]
  labels_test <- labels[-training_part,]
  
  
  return(list(
    X_train = X_train,
    X_test = X_test,
    labels_train = labels_train,
    labels_test = labels_test
  ))
}

#Compare 1 SVM for each C
compare.c <- function(X, t, C.range)
{

  n = length(C.range)
  
  error.value = rep(0,n)
  
  classes.weights <- sum(table(t))/table(t)
  total.time <- 0
  
  for (i in 1:n)
  {
    print(paste("  > start with C =", C.range[i]))
    
    
    start.time <- Sys.time()
    model <- ksvm(as.matrix(X), t, type="C-svc", C=C.range[i], kernel='rbfdot',scaled=c(), class.weights=classes.weights,cross=5)
    end.time <- Sys.time()
    
    error.value[i] = cross(model)
    
    time.taken <- end.time - start.time
    total.time <- total.time + time.taken
    print(time.taken)
  }
  
  return(error.value)
}

#Compare the different C applied for each labels
compare.c.all <- function(X, t, C.range)
{
  n <- length(C.range)
  error.df <- data.frame(C.range, rep(0,n), rep(0,n), rep(0,n), rep(0,n), rep(0,n), rep(0,n))
  colnames(error.df) <- c("C", classes.name)
  
  for(i in 1:6)
  {
    print(paste("> Comparing for label ", classes.name[i], "<"))
    
    error.df[,i+1] <- compare.c(X, t[classes.name[i]], C.range)
  }
  
  return(error.df)
}

create.svms <- function(X_train, labels_train, verbose=T, C=1)
{
  clf_list <- c()
  start.time <- Sys.time()
  for(i in 1:6)
  {
	t <- labels_train[classes.name[i]]
	classes.weights <- sum(table(t))/table(t)
    clf_list <- c(clf_list, list(ksvm(as.matrix(X_train), labels_train[,i], type="C-svc", C=C, kernel='rbfdot',scaled=c(), class.weights=classes.weights)))
  }
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  if (verbose)
  {
    print(time.taken)
  }
  
  return(clf_list)
}

pred.svms <- function(clf_list, X_test)
{
  #create dataframe prediction
  n.row <- length(X_test[,1])
  t_pred <- data.frame(rep(0, n.row), rep(0, n.row), rep(0, n.row), rep(0, n.row), rep(0, n.row), rep(0, n.row))
  colnames(t_pred) <- classes.name
  
  for(i in 1:6)
  {
    t_pred[classes.name[i]] <- predict(clf_list[[i]], as.matrix(X_test))  #now predict on the test set
  }
  
  return(t_pred)
}

#accuracy on a particular class
acc.classes <- function (t_pred, labels_test, numClass)
{
  class.checked <- classes.name[numClass]
  acc.hs <- sum(t_pred[[class.checked]] == labels_test[[class.checked]])/length(t_pred[[class.checked]])
  return  (acc.hs)
}

#accuracy independently on each classes
#CAUTION: Different from acc.svm of utility-factors.R
acc.svms <- function(t_pred, labels_test)
{
  
  return(sapply(1:6, function(x) acc.classes(t_pred, labels_test, x)))
}




#not used now but was used to check the data
analyse.factors <- function(labels)
{
  for(i in 1:6)
  {
    cat(classes.name[i])
    print(table(labels[,i]))
    cat("\n")
  }
}