library(mxnet)
splitTrain <- function(TrainTreated, sampleFrac = 2.4, folds = 5, nApproved = 13000, splitData = TRUE, ...) {
  ### first is optional down sampling for class imbalance
  index <- sample(nrow(TrainTreated), floor(nrow(TrainTreated) / folds), replace = FALSE)
  Test <- TrainTreated[index, ]
  Train <- TrainTreated[-index, ]
  sampleSize <- min(nApproved, sum(Train$completion_status == "APPROVED"))
  if(splitData) {
    Train <- bind_rows(list(
      Train %>%
        filter(completion_status == "COMPLETE") %>%
        sample_n(., size = sampleSize * sampleFrac),
      Train %>%
        filter(completion_status == "APPROVED") %>%
        sample_n(., size = sampleSize)
    ))
  }
  
  
  x_Train <- Train[,-which(names(Train) %in% c('completion_status', 'customer_application_id'))]
  y_Train <- Train$completion_status
  x_Test <- Test[,-which(names(Train) %in% c('completion_status', 'customer_application_id'))]
  y_Test <- Test$completion_status
  
  y_mxnet <- rep(0, length(y_Train)) # 0 is approved
  y_mxnet[y_Train == "COMPLETE"] <- 1
  x_mxnet <- data.matrix(x_Train)
  y_mxnet <- y_mxnet[complete.cases(x_mxnet)]
  x_mxnet <- x_mxnet[complete.cases(x_mxnet), ]
  
  y_mxnet_Test <- rep(0, length(y_Test)) # 0 is approved
  y_mxnet_Test[y_Test == "COMPLETE"] <- 1
  x_mxnet_Test <- data.matrix(x_Test)
  y_mxnet_Test <- y_mxnet_Test[complete.cases(x_mxnet_Test)]
  x_mxnet_Test <- x_mxnet_Test[complete.cases(x_mxnet_Test), ]
  out <- list(x_mxnet, y_mxnet, x_mxnet_Test, y_mxnet_Test)
  names(out) <- c("xTrain", "yTrain", "xTest", "yTest")
  out
}




getMXModel = function(xTrain, yTrain, xTest, yTest, dropout0 = 0, hidden1 = 12, 
                      dropout1 = 0.5, hidden2 = 10, dropout2 = 0.45, 
                      learning_rate = 0.1, num_round = 12, momentum_in = 0.70) {
  # model parameters
  data <- mx.symbol.Variable("data")
  dropout0 <- mx.symbol.Dropout(data, p = dropout0)
  fc1 <- mx.symbol.FullyConnected(data,  num_hidden=hidden1)
  act1 <- mx.symbol.Activation(fc1, act_type="relu")
  dropout1 <- mx.symbol.Dropout(act1, p = dropout1)
  fc2 <- mx.symbol.FullyConnected(act1, num_hidden=hidden2)
  act2 <- mx.symbol.Activation(fc2, act_type="relu")
  dropout2 <- mx.symbol.Dropout(act2, p = dropout2)
  fc3 <- mx.symbol.FullyConnected(act2, num_hidden=2)
  softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
  
  model <- mx.model.FeedForward.create(symbol=softmax,
                                       X=xTrain,
                                       y=yTrain,
                                       num.round=num_round,
                                       ctx=mx.cpu(),
                                       # need optimizer, eval.data, eval.metric
                                       eval.data=list(data=xTest, label=yTest),
                                       array.layout="rowmajor",
                                       learning.rate=learning_rate,
                                       momentum=momentum_in,
                                       array.batch.size = 128,
                                       eval.metric=mx.metric.accuracy,
                                       optimizer = 'sgd',
                                       initializer= mx.init.normal(1/sqrt(nrow(xTrain))))
  model
}


evaluateMXModel <- function(xTrain, yTrain, xTest, yTest, m, ...) {
  preds = predict(m, xTest, array.layout="rowmajor")
  pred.label <- max.col(t(preds)) - 1
  outAuc = ModelMetrics::auc(yTest, pred.label)
  predsll <- predict(m, xTest, array.layout="rowmajor")
  outLL <- ModelMetrics::logLoss(yTest, predsll[2,])
  outSens <- ModelMetrics::sensitivity(yTest, pred.label)
  outSpec <- ModelMetrics::specificity(yTest, pred.label)
  out <- list(outAuc, outLL, outSens, outSpec)
  names(out) <- c("AUC", "LogLoss", "Sensitivity", "Specificity")
  out
}

MXPipeline <- function(TrainTreated, ...) {
  trainingTestData <- splitTrain(TrainTreated, ...)
  MXModel <- getMXModel(trainingTestData[[1]], trainingTestData[[2]], 
                        trainingTestData[[3]], trainingTestData[[4]], ...)
  evaluateMXModel(trainingTestData[[1]], trainingTestData[[2]], 
                  trainingTestData[[3]], trainingTestData[[4]], MXModel)
}

MXPipelineCross <- function(...) {
  out <- list()
  for(i in 1:3) {
    out <- bind_rows(out, MXPipeline(...))
  }
  colMeans(out)
}


MXPipelineC <- function(x, ...) {
  function(...) MXPipelineCross(x, ...)
} 

MXPipelineTrain <- MXPipelineC(TrainTreated)

MXPipelineTrain()
