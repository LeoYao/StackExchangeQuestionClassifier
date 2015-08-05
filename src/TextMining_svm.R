
rm(list=ls(all=TRUE))

library("mda")
library("tm")
library("plyr")
library("class")
library("wordcloud")
library("SnowballC")
library("e1071")
library("nnet")
library("randomForest")
library("ipred")
library("lsa")
library("rpart")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!TODO: You need to change the path to the folder containing the data folder
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
setwd("~/Google Drive/NCSU/TextBooks/DM/Project")

paths <- "data"
topics <- c(1:10)
mystopwords <- c(stopwords("english"), "can", "shall", "may", "the")

cleanCorpus <- function(corpus){
  corpus.tmp <- tm_map(corpus, content_transformer(tolower),mc.cores=1)
  corpus.tmp <- tm_map(corpus.tmp, stemDocument, language="english",mc.cores=1)
  corpus.tmp <- tm_map(corpus.tmp, content_transformer(removeWords), mystopwords,mc.cores=1)
  corpus.tmp <- tm_map(corpus.tmp, removePunctuation,mc.cores=1)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace,mc.cores=1)
  return(corpus.tmp)
}

generateTDMWithBound <- function(tpc, path, lower, upper){
  s.dir <- sprintf("%s/%s", path, tpc)
  s.cor <- VCorpus(DirSource(directory = s.dir), readerControl = list(reader=readPlain))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl, control=list(bounds=list(global=c(lower,upper))))
  result <- list(topic=tpc, tdm=s.tdm)
}

bindTDM <- function(tdm)
{
  s.mat <- t(data.matrix(tdm[["tdm"]]))
  s.df <- as.data.frame(s.mat, stringAsFactor= FALSE)
  s.df <- cbind(s.df, TOPIC   = rep(tdm[["topic"]], nrow(s.df)))
  return (s.df)
}

bindTDMWithSamping <- function(tdm, sampling)
{
  s.mat <- t(data.matrix(tdm[["tdm"]]))
  s.mat <- s.mat[c(1:round(nrow(s.mat)*sampling)), ]
  s.df <- as.data.frame(s.mat, stringAsFactor= FALSE)
  s.df <- cbind(s.df, TOPIC   = rep(tdm[["topic"]], nrow(s.df)))
  return (s.df)
}


buildCrossValInd <- function(stack, folderCnt)
{
  ind <- c()
  for (t in c(1:10))
  {
    docCnt <- nrow(stack[stack$TOPIC == t,])
    testDataCnt <- ceiling(docCnt/folderCnt)
    deductCntForLastFolder <- folderCnt*testDataCnt - docCnt
    
    for (i in c(1:folderCnt))
    {
      if (i == folderCnt)
      {
        ind <- c(ind, rep(i,testDataCnt - deductCntForLastFolder))
      }
      else
      {
        ind <- c(ind, rep(i,testDataCnt))
      }
    }
  }
  return (ind)
}

#Main
low <- 50
high <- 1000
result.lsa <- data.frame()
result.old <- data.frame()
for (low in c(50, 100, 150))
{
  for (high in c(700,1000,1100))
  {
    #Build TDM
    tdm <- lapply(topics, generateTDMWithBound, path=paths, lower=low, upper=high)
    
    #Append TDM with topic as a feature
    tdm.bindedData <- lapply(tdm, bindTDM)
    #tdm.bindedData <- lapply(tdm, bindTDMWithSamping, sampling=0.1)
    
    #Merge diffent TDMs
    tdm.stack <- rbind.fill(tdm.bindedData)
    tdm.stack[is.na(tdm.stack)] <- 0
    
    #LSA
    tdm.stack.features <- tdm.stack[,names(tdm.stack) !='TOPIC']
    tdm.stack.feature.matrix <- t(tdm.stack.features)
    
    tdm.stack.feature.matrix <- lw_bintf(tdm.stack.feature.matrix) * gw_idf(tdm.stack.feature.matrix)
    tdm.stack.feature.lsa <- lsa(tdm.stack.feature.matrix,dims=dimcalc_share(0.3))
    #tdm.stack.feature.lsa <- lsa(tdm.stack.feature.matrix)
    tdm.stack.feature.lsa.textmatrix <- as.textmatrix(tdm.stack.feature.lsa)
    tdm.stack.feature.lsa.df <- as.data.frame(t(tdm.stack.feature.lsa.textmatrix[,]))
    #workround for NB
    tdm.stack.feature.lsa.df <- tdm.stack.feature.lsa.df - min(tdm.stack.feature.lsa.df)
    
    tdm.stack.lsa.df <- cbind(tdm.stack.feature.lsa.df, TOPIC=tdm.stack$TOPIC)
    
    #CrossValidation
    folderCnt <- 5
    tdm.ind <- buildCrossValInd(tdm.stack.lsa.df,folderCnt)
    
    err_rates.lsa <- c()
    precisions.lsa <- c()
    recalls.lsa <- c()
    err_rates.old <- c()
    precisions.old <- c()
    recalls.old <- c()
    err_var.lsa <- c()
    err_var.old <- c()
    
    {
      testCnt.lsa <- 0
      errorCnt.lsa <- 0
      correctCntForTp1Precission.lsa <- 0
      predCntForTp1.lsa <- 0
      correctCntForTp1Recall.lsa <- 0
      actualCntForTp1.lsa <- 0
      testCnt.old <- 0
      errorCnt.old <- 0
      correctCntForTp1Precission.old <- 0
      predCntForTp1.old <- 0
      correctCntForTp1Recall.old <- 0
      actualCntForTp1.old <- 0
      prec_tmp.old <- 0
      recal_tmp.old <- 0
      prec_tmp.lsa <- 0
      recal_tmp.lsa <- 0
      
      err_arr.lsa <- c()
      err_arr.old <- c()
      
      for (i in c(1:folderCnt))
      {
        print(paste("Begin: lower", low, "upper", high, "cross folder", i))
        #Determine test and train data for LSA
        test.lsa <- tdm.stack.lsa.df[tdm.ind == i,]
        train.lsa <- tdm.stack.lsa.df[tdm.ind != i, ]
        test.lsa.features <- test.lsa[,names(test.lsa) !='TOPIC']
        test.lsa.classes <- as.factor(test.lsa[,names(test.lsa) =='TOPIC'])
        train.lsa.features <- train.lsa[,names(train.lsa) !='TOPIC']
        train.lsa.classes <- as.factor(train.lsa[,names(train.lsa) =='TOPIC'])
        
        #Determine test and train data for old data
        test.old <- tdm.stack[tdm.ind == i,]
        train.old <- tdm.stack[tdm.ind != i, ]
        test.old.features <- test.old[,names(test.old) !='TOPIC']
        test.old.classes <- as.factor(test.old[,names(test.old) =='TOPIC'])
        train.old.features <- train.old[,names(train.old) !='TOPIC']
        train.old.classes <- as.factor(train.old[,names(train.old) =='TOPIC'])
        
        
        #SVM
        data.svm.lsa <- svm(train.lsa.features, train.lsa.classes, scale=F)
        pred_value.lsa <- predict(data.svm.lsa, test.lsa.features, type='class')
        tb.lsa <- data.frame(pred = pred_value.lsa, actual = test.lsa.classes)
        data.svm.old <- svm(train.old.features, train.old.classes, scale=F)
        pred_value.old <- predict(data.svm.old, test.old.features, type='class')
        tb.old <- data.frame(pred = pred_value.old, actual = test.old.classes)
                
        #statistics for LSA
        errorCnt.lsa <- errorCnt.lsa + nrow(tb.lsa[tb.lsa$pred != tb.lsa$actual,])
        testCnt.lsa <- testCnt.lsa + nrow(test.lsa)
        err_arr.lsa <- c(err_arr.lsa, errorCnt.lsa/testCnt.lsa)
        
        for (tpi in c(1:10))
        {
          correctCntForTpPrecission.lsa <- nrow(tb.lsa[tb.lsa$pred == tb.lsa$actual & tb.lsa$pred == tpi,])
          predCntForTp.lsa <- nrow(tb.lsa[tb.lsa$pred == tpi,])
          correctCntForTpRecall.lsa <- nrow(tb.lsa[tb.lsa$pred == tb.lsa$actual & tb.lsa$actual == tpi,])
          actualCntForTp.lsa <- actualCntForTp1.lsa + nrow(tb.lsa[tb.lsa$actual == tpi,])
          prec_tmp.lsa <- prec_tmp.lsa + correctCntForTpPrecission.lsa / predCntForTp.lsa
          recal_tmp.lsa <- recal_tmp.lsa + correctCntForTpRecall.lsa / actualCntForTp.lsa
        }
        #statistics for original data
        errorCnt.old <- errorCnt.old + nrow(tb.old[tb.old$pred != tb.old$actual,])
        testCnt.old <- testCnt.old + nrow(test.old)
        err_arr.old <- c(err_arr.old, errorCnt.old/testCnt.old)
          
        for (tpi in c(1:10))
        {
          correctCntForTpPrecission.old <- nrow(tb.old[tb.old$pred == tb.old$actual & tb.old$pred == tpi,])
          predCntForTp.old <- nrow(tb.old[tb.old$pred == tpi,])
          correctCntForTpRecall.old <- nrow(tb.old[tb.old$pred == tb.old$actual & tb.old$actual == tpi,])
          actualCntForTp.old <- actualCntForTp1.old + nrow(tb.old[tb.old$actual == tpi,])
          prec_tmp.old <- prec_tmp.old + correctCntForTpPrecission.old / predCntForTp.old
          recal_tmp.old <- recal_tmp.old + correctCntForTpRecall.old / actualCntForTp.old
        }
      }
      
      err_rates.lsa <- c(err_rates.lsa, errorCnt.lsa/testCnt.lsa)
      precisions.lsa <- c(precisions.lsa, prec_tmp.old/(folderCnt *10))
      recalls.lsa <- c(recalls.lsa, recal_tmp.lsa/(folderCnt *10))
      err_var.lsa <- c(err_var.lsa, var(err_arr.lsa, err_arr.lsa))
      
      err_rates.old <- c(err_rates.old, errorCnt.old/testCnt.old)
      precisions.old <- c(precisions.old, prec_tmp.old/(folderCnt *10))
      recalls.old <- c(recalls.old, recal_tmp.old/(folderCnt *10))
      err_var.old <- c(err_var.old, var(err_arr.old, err_arr.old))
    }
    result.lsa.tmp <- data.frame(lower=low, upper=high, error=err_rates.lsa, err_var = err_var.lsa, precision_tp1.lsa = precisions.lsa, recall_tp1.lsa = recalls.lsa, f_score_tp1.lsa = 2/(1/precisions.lsa + 1/recalls.lsa))
    result.old.tmp <- data.frame(lower=low, upper=high, error=err_rates.old, err_var = err_var.old, precision_tp1.old = precisions.old, recall_tp1.old = recalls.old, f_score_tp1.old = 2/(1/precisions.old + 1/recalls.old))
    
    result.lsa <- rbind(result.lsa, result.lsa.tmp)
    result.old <- rbind(result.old, result.old.tmp)
  }
}
result.lsa
result.old
