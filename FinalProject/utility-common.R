library(kernlab)
library(gplots)
library(ROCR)

classes.name <- c('HandStart','FirstDigitTouch',    #class names
                  'BothStartLoadPhase','LiftOff',
                  'Replace','BothReleased')


ROC <- function(t_pred, labels_test, numClass)
{
  class.checked <- classes.name[numClass]
  
  pred <- as.numeric(t_pred[[class.checked]])-1
  labs <- as.numeric(labels_test[[class.checked]])-1
  
  
  pr <- prediction(pred, labs)
  
  auc <- performance(pr, measure = "auc")
  (auc <- auc@y.values[[1]])
  
  prf <- performance(pr, measure = "tpr", x.measure = "fpr")
  
  
  return(list(
    auc=auc,
    prf=prf
  ))
}

plotROCs <- function(rocs, numSubject)
{
  couleurs <- c(1:6)	#colors already taken
  
  roc = rocs[,1]
  
  auc <- roc[["auc"]]
  prf <- roc[["prf"]]
  title <- paste("ROC Curve subject", numSubject, "(mean(AUC) = ")
  
  avgauc <- auc
  
  plot(prf, col=couleurs[1], lwd = 3)
  
  for(i in 2:6)
  {
    roc = rocs[,i]
    
    auc <- roc[["auc"]]
    prf <- roc[["prf"]]
    
    avgauc <- avgauc + auc
    
    plot(prf, add=TRUE, col=couleurs[i], lwd = 3)
  }
  
  avgauc <- avgauc/6
  
  
  legend("bottomright",classes.name,fill=couleurs)
  
  title(main=paste(title, (floor(avgauc*1000)/1000),")", sep=""))
  abline(a=0, b=1, lty=3)
}