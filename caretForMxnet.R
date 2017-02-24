# a generic grid search function using mxnet multi-layer perceptron for binary classification
# example using sonar data


require(mlbench)
require(mxnet)
data(Sonar, package="mlbench")
Sonar[,61] = as.numeric(Sonar[,61])-1
data.x = data.matrix(Sonar[, 1:60])
data.y = Sonar[, 61]


# cv and tuning for mxnet mlp
# add option to return model for one row tune_grid (if user wants model for minimum logloss)
mx_binaryclass_cv = function(data.x, data.y, folds=5, repeats=3, tune_grid, 
                             activation_type="sigmoid", return_model=FALSE, ...) {
  stopifnot(length(data.y) == nrow(data.x))
  stopifnot(is.matrix(data.x) & is.vector(data.y))
  
  if(return_model == TRUE) {

    model <- mx.mlp(data.x, data.y, hidden_node=tune_grid[["hidden_nodes"]], 
                    out_node=2, out_activation="softmax", activation = activation_type,
                    num.round=tune_grid[["num_rounds"]], array.batch.size=15, 
                    learning.rate=tune_grid[["learning_rate"]], 
                    momentum=0.9, array.layout="rowmajor",
                    dropout=0, initializer=mx.init.Xavier(),
                    eval.metric=mx.metric.accuracy)
    return(model)
  } else {
  
  ll_cv_train = function(...) {
    index = sample(nrow(data.x), floor(nrow(data.x) * (1 - 1 / folds)), replace=FALSE)
    model <- mx.mlp(data.x[index,], data.y[index], hidden_node=tune_grid[["hidden_nodes"]], 
                    out_node=tune_grid[["out_nodes"]], out_activation="softmax", activation = activation_type,
                    num.round=tune_grid[["num_rounds"]], array.batch.size=15, 
                    learning.rate=tune_grid[["learning_rate"]], 
                    momentum=0.9, array.layout="rowmajor",
                    dropout=0, initializer=mx.init.Xavier(),
                    eval.metric=mx.metric.accuracy)
    preds = predict(model, data.x[-index,], array.layout="rowmajor")
    out = ModelMetrics::logLoss(data.y[-index], preds[2,])
    out
  }
  
  results = vapply(1:repeats, ll_cv_train, numeric(1))
  out = mean(results)
  print(paste("Average Log-loss: ", out))
  out
  }
}


tune_table = expand.grid(num_rounds = c(20, 40, 80), hidden_nodes = c(10, 20), out_nodes = c(2, 4),
                         learning_rate = c(0.05, 0.1), stringsAsFactors = FALSE)


tune_table$results = apply(tune_table, 1, 
                           FUN = function(x) mx_binaryclass_cv(data.x, data.y, folds=5, repeats = 3, 
                                                               tune_grid = x))
tune_table

# get model for lowest logloss
best_params = tune_table[which(tune_table$results == min(tune_table$results)), ]
best_mlp_model = mx_binaryclass_cv(data.x, data.y, tune_grid=best_params, return_model = TRUE)
graph.viz(best_mlp_model$symbol)
preds = predict(best_mlp_model, data.x)
pred.label = max.col(t(preds))-1
table(pred.label, data.y)
# it works



# compare with ranger
ranger.data.y = as.factor(data.y)
levels(ranger.data.y) = make.names(c("level_0", "level_1"), unique = TRUE)
ctrl_params <- trainControl(method='repeatedcv', number = 5, repeats = 3, verboseIter=TRUE, 
                            summaryFunction=mnLogLoss, classProbs = TRUE)
fit.ranger <- train(x=data.x, y=ranger.data.y, 
                    method='ranger', 
                    trControl=ctrl_params, 
                    tuneLength = 5)  

fit.ranger
