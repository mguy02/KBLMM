



bin.to.dec <- function(x)
{
  packBits(rev(c(rep(FALSE, 32-length(x)%%32), as.logical(x))), "integer")
}

dec.to.bin <- function (num)
{
  #convert it to bin then remove extra 0
  bin <- as.integer(paste(sapply(strsplit(paste(rev(intToBits(num))),""),`[[`,2),collapse=""))
  
  #pad to 6 zeroes
  bin <- strsplit(sprintf("%06d", bin), "")
  
  return(as.vector(sapply(bin, as.integer)))
}

load.data <- function (dataFolder = "data", numSubject)
{
  basepath <- paste(dataFolder, "filtered_", sep="/")
  
  filename.sig <- paste(basepath, numSubject ,".csv", sep="")
  filename.lab <- paste(basepath, numSubject ,"_labels.csv", sep="")
  
  signals <- read.csv(filename.sig, header=T)
  labels <- read.csv(filename.lab, header=T)
  
  n <- length(labels[,1])
  #print(n)
  lab <- rep(0, n)
  
  for(i in 1:n)
  {
    lab[i] <- as.numeric(bin.to.dec(labels[i,]))
  }
  
  lab <- as.factor(lab)
  
  for(i in 1:6)
  {
    labels[,i] <- as.factor(labels[,i])
  }
  
  return(list(
    signals = signals,
    labels.dec = lab,
    labels.bin = labels
  ))
}

splitData <- function(dataset, training_ratio)
{
  data <- dataset$signals
  labels.dec <- dataset$labels.dec
  labels.bin <- dataset$labels.bin
  
  N <- length(data[,1])
  ntrain <- round(N*training.ratio)
  training_part <- sample(N, ntrain)
  
  X_train = data[training_part,]
  X_test <- data[-training_part,]
  
  labels_train.dec <- labels.dec[training_part]
  labels_test.dec <- labels.dec[-training_part]
  
  labels_train.bin <- labels.bin[training_part,]
  labels_test.bin <- labels.bin[-training_part,]
  
  return(list(
    X_train = X_train,
    X_test = X_test,
    labels_train.dec = labels_train.dec,
    labels_test.dec = labels_test.dec,
    labels_train.bin = labels_train.bin,
    labels_test.bin = labels_test.bin
  ))
}


compare.c <- function(X, t, C.range)
{
  n = length(C.range)
  
  error.value = rep(0,n)
  
  
  classes.weights <- sum(table(t))/table(t)
  total.time <- 0

  for (i in 1:n)
  {
    print(paste("> start with C =", C.range[i]))
    
    
    start.time <- Sys.time()
    model <- ksvm(as.matrix(X), t, type="C-svc", C=C.range[i], kernel='rbfdot',scaled=c(), class.weights=classes.weights,cross=5)
    end.time <- Sys.time()
    
    error.value[i] = cross(model)
    
    time.taken <- end.time - start.time
    total.time <- total.time + time.taken
    print(time.taken)
    
  }
  
  print("Total Time:")
  print(total.time)
  
  return(error.value)
}

create.svm <- function(X_train, labels_train, verbose=T, C=1)
{
  
  classes.weights <- sum(table(labels_train))/table(labels_train)
  
  start.time <- Sys.time()

  model <- ksvm(as.matrix(X_train), labels_train, type="C-svc", C=C, kernel='rbfdot',scaled=c(), class.weights=classes.weights)
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  if (verbose)
  {
    print(time.taken)
  }

  return(model)
}

#Create a dataframe prediction of the SVM
pred.svm <- function(model, X_test)
{
  #create dataframe prediction
  n.row <- length(X_test[,1])
  t_pred <- rep(0, n.row)
  
  t_pred <- predict(model, as.matrix(X_test))  #now predict on the test set
  
  
  return(t_pred)
}

#accuracy on exact labels
#CAUTION: Different from acc.svm of utility-6svm.R
acc.svm <- function(t_pred, labels_test)
{
  acc <- 0
  
  n <- length(labels_test)
  
  acc <- sum(t_pred == labels_test)
  
  return(acc/n)
}

decdf.to.bindf <- function(t_pred.dec)
{
  n.row <- 6
  
  t_pred <- data.frame(rep(0, n.row), rep(0, n.row), rep(0, n.row), rep(0, n.row), rep(0, n.row), rep(0, n.row))
  colnames(t_pred) <- classes.name
  
  for(i in 1:length(t_pred.dec))
  {
    #print(paste("On convertit", t_pred[i]))
    t_pred[i,] <- dec.to.bin(as.integer(as.character(t_pred.dec[i])))
  }
  return(t_pred)
}

#accuracy on a particular class
acc.classes <- function (t_pred.bin, labels_test.bin, numClass)
{
  class.checked <- classes.name[numClass]
  acc.hs <- sum(t_pred.bin[[class.checked]] == labels_test.bin[[class.checked]])/length(t_pred.bin[[class.checked]])
  return  (acc.hs)
}