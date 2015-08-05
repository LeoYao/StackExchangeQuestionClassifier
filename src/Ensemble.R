rm(list=ls(all=TRUE))

library("mda")
library("tm")
library("plyr")
library("class")
library("SnowballC")
library("e1071")
library("nnet")
library("randomForest")
library("ipred")
library("rpart")
library("lsa")

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

getDataIndex <- function(stack)
{
  android_ind <- c()
  apple_ind <- c()
  electronics_ind <- c()
  gis_ind <- c()
  mathematica_ind <- c()
  photo_ind <- c()
  scifi_ind <- c()
  security_ind <- c()
  unix_ind <- c()
  wordpress_ind <- c()
  android_ind <- as.integer(rownames(stack[stack$TOPIC == '1',]))
  apple_ind <- as.integer(rownames(stack[stack$TOPIC == '2',]))
  electronics_ind <- as.integer(rownames(stack[stack$TOPIC == '3',]))
  gis_ind <- as.integer(rownames(stack[stack$TOPIC == '4',]))
  mathematica_ind <- as.integer(rownames(stack[stack$TOPIC == '5',]))
  photo_ind <- as.integer(rownames(stack[stack$TOPIC == '6',]))
  scifi_ind <- as.integer(rownames(stack[stack$TOPIC == '7',]))
  security_ind <- as.integer(rownames(stack[stack$TOPIC == '8',]))
  unix_ind <- as.integer(rownames(stack[stack$TOPIC == '9',]))
  wordpress_ind <- as.integer(rownames(stack[stack$TOPIC == '10',]))
  
  r.ind <- list(android_ind,electronics_ind,mathematica_ind,scifi_ind,unix_ind,
                gis_ind,security_ind,wordpress_ind,apple_ind,photo_ind)
  
  return(r.ind)
}

buildTrainTestIndex <- function(s, sampling, testTrainRatio)
{
  r <- sample(s,ceiling(length(s)*sampling), replace=F)
  r.test <- sample(r, ceiling(length(r)*testTrainRatio), replace=F)
  r.train <- r[!(r %in% r.test)]
  return (list(train=r.train,test=r.test,data=r))
}

mergeTrainTestIndex <- function(s)
{
  r.test <- c()
  r.train <- c()
  r <-c()
  for (i in seq(1, length(s)))
  {
    r.test <- c(r.test, s[[i]]$test)
    r.train <- c(r.train, s[[i]]$train)
    r <-c(r, s[[i]]$data)
  }
  return (list(train=r.train,test=r.test, data=r))
}

buildTrainTestIndex2 <- function(s, sampling, testTrainRatio, test2TestRatio)
{
  r = sample(s,ceiling(length(s)*sampling), replace=F)
  r.test <- sample(r, ceiling(length(r)*testTrainRatio), replace=F)
  r.train <- r[!(r %in% r.test)]
  r.test2 <- sample(r.test, ceiling(length(r.test)*test2TestRatio), replace=F)
  r.test <- r.test[!(r.test %in% r.test2)]
  return (list(train=r.train, test=r.test, test2=r.test2, data=r))
}

mergeTrainTestIndex2 <- function(s)
{
  r.test <- c()
  r.test2 <- c()
  r.train <- c()
  r <-c()
  for (i in seq(1, length(s)))
  {
    r.test <- c(r.test, s[[i]]$test)
    r.test2 <- c(r.test2, s[[i]]$test2)
    r.train <- c(r.train, s[[i]]$train)
    r <-c(r, s[[i]]$data)
  }
  return (list(train=r.train,test=r.test,test2=r.test2, data=r))
}


#Build TDM
tdm_svm_nb <- lapply(topics, generateTDMWithBound, path=paths, lower=50, upper=1000)
tdm_ann <- lapply(topics, generateTDMWithBound, path=paths, lower=50, upper=1100)
tdm_dt <- lapply(topics, generateTDMWithBound, path=paths, lower=100, upper=1100)
tdm_knn <- lapply(topics, generateTDMWithBound, path=paths, lower=120, upper=1100)

#Append TDM with topic as a feature
tdm_svm_nb.bindedData <- lapply(tdm_svm_nb, bindTDM)
tdm_ann.bindedData <- lapply(tdm_ann, bindTDM)
tdm_dt.bindedData <- lapply(tdm_dt, bindTDM)
tdm_knn.bindedData <- lapply(tdm_knn, bindTDM)


#Merge diffent TDMs
tdm_svm_nb.stack <- rbind.fill(tdm_svm_nb.bindedData)
tdm_svm_nb.stack[is.na(tdm_svm_nb.stack)] <- 0
tdm_ann.stack <- rbind.fill(tdm_ann.bindedData)
tdm_ann.stack[is.na(tdm_ann.stack)] <- 0
tdm_dt.stack <- rbind.fill(tdm_dt.bindedData)
tdm_dt.stack[is.na(tdm_dt.stack)] <- 0
tdm_knn.stack <- rbind.fill(tdm_knn.bindedData)
tdm_knn.stack[is.na(tdm_knn.stack)] <- 0

#Get indexes of each topic
set.seed(5678)
tdm.tpc_ind <- getDataIndex(tdm_svm_nb.stack)
#90% train, %7 preminary test (train for ensemble method), %3 final test
tdm.testTrain_ind <- lapply(tdm.tpc_ind, buildTrainTestIndex2, sampling=1, testTrainRatio=0.9, test2TestRatio = 0.3)
tdm.mergedTestTrainInd <- mergeTrainTestIndex2(tdm.testTrain_ind)

# Data for ANN is extracted from original data
tdm_ann.test1 <- tdm_ann.stack[tdm.mergedTestTrainInd$test,]
tdm_ann.test1.features <- tdm_ann.test1[,names(tdm_ann.test1)!='TOPIC']
tdm_ann.test1.classes <- tdm_ann.test1[,names(tdm_ann.test1)=='TOPIC']

tdm_ann.test2 <- tdm_ann.stack[tdm.mergedTestTrainInd$test2,]
tdm_ann.test2.features <- tdm_ann.test2[,names(tdm_ann.test2)!='TOPIC']

# the final target class shared by all classifiers
tdm.test1.classes <- tdm_ann.test1[,names(tdm_ann.test1)=='TOPIC']
tdm.test2.classes <- tdm_ann.test2[,names(tdm_ann.test2)=='TOPIC']

tdm_ann.train <- tdm_ann.stack[tdm.mergedTestTrainInd$train,]
tdm_ann.train.features <- tdm_ann.train[,names(tdm_ann.train)!='TOPIC']
tdm_ann.train.classes <- tdm_ann.train[,names(tdm_ann.train)=='TOPIC']


data.ann <- nnet(as.factor(TOPIC) ~ ., data = tdm_ann.train, size = 10, rang = 0.05, decay = 0, maxit = 2000, MaxNWts = 10000)
pred_value.ann <- predict(data.ann, tdm_ann.test1.features, type='class')
confusion(pred_value.ann, tdm_ann.test1.classes)
pred_value2.ann <- predict(data.ann, tdm_ann.test2.features, type='class')

#LSA for SVM and NB
tdm_svm_nb.stack.features <- tdm_svm_nb.stack[,names(tdm_svm_nb.stack) !='TOPIC']
tdm_svm_nb.stack.feature.matrix <- t(tdm_svm_nb.stack.features)
tdm_svm_nb.stack.feature.matrix <- lw_bintf(tdm_svm_nb.stack.feature.matrix) * gw_idf(tdm_svm_nb.stack.feature.matrix)
tdm_svm_nb.stack.feature.lsa <- lsa(tdm_svm_nb.stack.feature.matrix,dims=dimcalc_share(0.3))
tdm_svm_nb.stack.feature.lsa.textmatrix <- as.textmatrix(tdm_svm_nb.stack.feature.lsa)
tdm_svm_nb.stack.feature.lsa.df <- as.data.frame(t(tdm_svm_nb.stack.feature.lsa.textmatrix[,]))
tdm_svm_nb.stack.feature.lsa.df <- cbind(tdm_svm_nb.stack.feature.lsa.df, TOPIC=tdm_svm_nb.stack$TOPIC)
#workround for NB
tdm_svm_nb.stack.feature.lsa.df[,-ncol(tdm_svm_nb.stack.feature.lsa.df)] <- tdm_svm_nb.stack.feature.lsa.df[,-ncol(tdm_svm_nb.stack.feature.lsa.df)] - min(tdm_svm_nb.stack.feature.lsa.df[,-ncol(tdm_svm_nb.stack.feature.lsa.df)])

#Divide data for SVM and NB
tdm_svm_nb.test1.lsa <- tdm_svm_nb.stack.feature.lsa.df[tdm.mergedTestTrainInd$test,]
tdm_svm_nb.test1.features.lsa <- tdm_svm_nb.test1.lsa[,names(tdm_svm_nb.test1.lsa)!='TOPIC']
tdm_svm_nb.test1.classes.lsa <- tdm_svm_nb.test1.lsa[,names(tdm_svm_nb.test1.lsa)=='TOPIC']
tdm_svm_nb.test2.lsa <- tdm_svm_nb.stack.feature.lsa.df[tdm.mergedTestTrainInd$test2,]
tdm_svm_nb.test2.features.lsa <- tdm_svm_nb.test2.lsa[,names(tdm_svm_nb.test2.lsa)!='TOPIC']
tdm_svm_nb.test2.classes.lsa <- tdm_svm_nb.test2.lsa[,names(tdm_svm_nb.test2.lsa)=='TOPIC']
tdm_svm_nb.train.lsa <- tdm_svm_nb.stack.feature.lsa.df[tdm.mergedTestTrainInd$train,]
tdm_svm_nb.train.features.lsa <- tdm_svm_nb.train.lsa[,names(tdm_svm_nb.train.lsa)!='TOPIC']
tdm_svm_nb.train.classes.lsa <- tdm_svm_nb.train.lsa[,names(tdm_svm_nb.train.lsa)=='TOPIC']

#SVM
data.svm.lsa <- svm(tdm_svm_nb.train.features.lsa, as.factor(tdm_svm_nb.train.classes.lsa), scale=F)
pred_value.svm.lsa <- predict(data.svm.lsa, tdm_svm_nb.test1.features.lsa)
confusion(pred_value.svm.lsa, as.factor(tdm_svm_nb.test1.classes.lsa))
pred_value2.svm.lsa <- predict(data.svm.lsa, tdm_svm_nb.test2.features.lsa)

#NB
data.nb.lsa<-naiveBayes(tdm_svm_nb.train.features.lsa, as.factor(tdm_svm_nb.train.classes.lsa)) 
pred_value.nb.lsa <- predict(data.nb.lsa, tdm_svm_nb.test1.features.lsa)
confusion(pred_value.nb.lsa, as.factor(tdm_svm_nb.test1.classes.lsa))
pred_value2.nb.lsa <- predict(data.nb.lsa, tdm_svm_nb.test2.features.lsa)


#LSA for DT
tdm_dt.stack.features <- tdm_dt.stack[,names(tdm_dt.stack) !='TOPIC']
tdm_dt.stack.feature.matrix <- t(tdm_dt.stack.features)
tdm_dt.stack.feature.matrix <- lw_bintf(tdm_dt.stack.feature.matrix) * gw_idf(tdm_dt.stack.feature.matrix)
tdm_dt.stack.feature.lsa <- lsa(tdm_dt.stack.feature.matrix,dims=dimcalc_share(0.3))
tdm_dt.stack.feature.lsa.textmatrix <- as.textmatrix(tdm_dt.stack.feature.lsa)
tdm_dt.stack.feature.lsa.df <- as.data.frame(t(tdm_dt.stack.feature.lsa.textmatrix[,]))
tdm_dt.stack.feature.lsa.df <- cbind(tdm_dt.stack.feature.lsa.df, TOPIC=tdm_dt.stack$TOPIC)

#Divide data for DT
tdm_dt.test1.lsa <- tdm_dt.stack.feature.lsa.df[tdm.mergedTestTrainInd$test,]
tdm_dt.test1.features.lsa <- tdm_dt.test1.lsa[,names(tdm_dt.test1.lsa)!='TOPIC']
tdm_dt.test1.classes.lsa <- tdm_dt.test1.lsa[,names(tdm_dt.test1.lsa)=='TOPIC']
tdm_dt.test2.lsa <- tdm_dt.stack.feature.lsa.df[tdm.mergedTestTrainInd$test2,]
tdm_dt.test2.features.lsa <- tdm_dt.test2.lsa[,names(tdm_dt.test2.lsa)!='TOPIC']
tdm_dt.test2.classes.lsa <- tdm_dt.test2.lsa[,names(tdm_dt.test2.lsa)=='TOPIC']
tdm_dt.train.lsa <- tdm_dt.stack.feature.lsa.df[tdm.mergedTestTrainInd$train,]
tdm_dt.train.features.lsa <- tdm_dt.train.lsa[,names(tdm_dt.train.lsa)!='TOPIC']
tdm_dt.train.classes.lsa <- tdm_dt.train.lsa[,names(tdm_dt.train.lsa)=='TOPIC']

#DT
data.dt.lsa <- rpart(as.factor(TOPIC) ~ ., data=tdm_dt.train.lsa, method="class", parms=list(split='gini'), control=rpart.control(minsplit=1,minbucket=1, cp=0))
pred_value.dt.lsa <- predict(data.dt.lsa, newdata=tdm_dt.test1.features.lsa, type="class")
confusion(pred_value.dt.lsa, tdm_dt.test1.classes.lsa)
pred_value2.dt.lsa <- predict(data.dt.lsa, newdata=tdm_dt.test2.features.lsa, type="class")

#LSA for KNN
tdm_knn.stack.features <- tdm_knn.stack[,names(tdm_knn.stack) !='TOPIC']
tdm_knn.stack.feature.matrix <- t(tdm_knn.stack.features)
tdm_knn.stack.feature.matrix <- lw_bintf(tdm_knn.stack.feature.matrix) * gw_idf(tdm_knn.stack.feature.matrix)
tdm_knn.stack.feature.lsa <- lsa(tdm_knn.stack.feature.matrix,dims=dimcalc_share(0.3))
tdm_knn.stack.feature.lsa.textmatrix <- as.textmatrix(tdm_knn.stack.feature.lsa)
tdm_knn.stack.feature.lsa.df <- as.data.frame(t(tdm_knn.stack.feature.lsa.textmatrix[,]))
tdm_knn.stack.feature.lsa.df <- cbind(tdm_knn.stack.feature.lsa.df, TOPIC=tdm_knn.stack$TOPIC)

#Divide data for KNN
tdm_knn.test1.lsa <- tdm_knn.stack.feature.lsa.df[tdm.mergedTestTrainInd$test,]
tdm_knn.test1.features.lsa <- tdm_knn.test1.lsa[,names(tdm_knn.test1.lsa)!='TOPIC']
tdm_knn.test1.classes.lsa <- tdm_knn.test1.lsa[,names(tdm_knn.test1.lsa)=='TOPIC']
tdm_knn.test2.lsa <- tdm_knn.stack.feature.lsa.df[tdm.mergedTestTrainInd$test2,]
tdm_knn.test2.features.lsa <- tdm_knn.test2.lsa[,names(tdm_knn.test2.lsa)!='TOPIC']
tdm_knn.test2.classes.lsa <- tdm_knn.test2.lsa[,names(tdm_knn.test2.lsa)=='TOPIC']
tdm_knn.train.lsa <- tdm_knn.stack.feature.lsa.df[tdm.mergedTestTrainInd$train,]
tdm_knn.train.features.lsa <- tdm_knn.train.lsa[,names(tdm_knn.train.lsa)!='TOPIC']
tdm_knn.train.classes.lsa <- tdm_knn.train.lsa[,names(tdm_knn.train.lsa)=='TOPIC']

#KNN
data.knn.lsa <- knn(train=tdm_knn.train.features.lsa, test=tdm_knn.test1.features.lsa, cl=tdm_knn.train.classes.lsa, k=1)
confusion(object = data.knn.lsa, true=tdm_knn.test1.classes.lsa)
data.knn2.lsa <- knn(train=tdm_knn.train.features.lsa, test=tdm_knn.test2.features.lsa, cl=tdm_knn.train.classes.lsa, k=1)


#Ensemble Method
rst <- data.frame(SVM=pred_value.svm.lsa, DT = pred_value.dt.lsa, NB= pred_value.nb.lsa, KNN=data.knn.lsa, ANN=pred_value.ann, ACTUAL = as.factor(tdm.test1.classes))
rst2 <- data.frame(SVM=pred_value2.svm.lsa, DT = pred_value2.dt.lsa, NB= pred_value2.nb.lsa, KNN=data.knn2.lsa, ANN=pred_value2.ann, ACTUAL = as.factor(tdm.test2.classes))

data.dt.ensemble <- rpart(ACTUAL ~ ., data=rst, method="class", parms=list(split='gini'), control=rpart.control(minsplit=1,minbucket=1, cp=0))
pred_value.dt.ensemble <- predict(data.dt.ensemble, newdata=rst2[,-6], type="class")
confusion(pred_value.dt.ensemble, rst2[,6])
#pred_value.dt.ensemble <- predict(data.dt.ensemble, newdata=rst[,-6], type="class")
#confusion(pred_value.dt.ensemble, rst[,6])

data.svm.ensemble <- svm(rst[,-6], as.factor(rst[,6]), scale=F)
pred_value.svm.ensemble <- predict(data.svm.ensemble, newdata=rst2[,-6], type="class")
confusion(pred_value.svm.ensemble, rst2[,6])
#pred_value.svm.ensemble <- predict(data.svm.ensemble, newdata=rst[,-6], type="class")
#confusion(pred_value.svm.ensemble, rst[,6])

data.nb.ensemble <-naiveBayes(rst[,-6], as.factor(rst[,6])) 
pred_value.nb.ensemble <- predict(data.nb.ensemble, newdata=rst2[,-6], type="class")
confusion(pred_value.nb.ensemble, rst2[,6])
#pred_value.nb.ensemble <- predict(data.nb.ensemble, newdata=rst[,-6], type="class")
#confusion(pred_value.nb.ensemble, rst[,6])

data.rf.ensemble <- randomForest(x=rst[,-6], y=as.factor(rst[,6]), importance=TRUE, ntree=100)
pred_value.rf.ensemble <- predict(data.rf.ensemble, newdata=rst2[,-6], type="class")
confusion(pred_value.rf.ensemble, rst2[,6])

data.bag.ensemble <- bagging(as.factor(ACTUAL) ~ ., data=rst, coob=T)
pred_value.bag.ensemble = predict(data.bag.ensemble, newdata=rst2[,-6], type='class')
confusion(pred_value.bag.ensemble, rst2[,6])

#Accuracy of recommendation of topic range
#nrow(rst2[rst2$SVM == rst2$ACTUAL | rst2$DT == rst2$ACTUAL | rst2$KNN == rst2$ACTUAL | rst2$NB == rst2$ACTUAL | rst2$ANN==rst2$ACTUAL,])/nrow(rst2)
