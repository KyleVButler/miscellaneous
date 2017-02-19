# a generic grid search function using mxnet for binary classification

require(mlbench)
require(mxnet)
data(Sonar, package="mlbench")
Sonar[,61] = as.numeric(Sonar[,61])-1
train.ind = c(1:50, 100:150)
data.x = data.matrix(Sonar[, 1:60])
data.y = Sonar[, 61]


# cv and tuning for mxnet mlp
mx_binaryclass_cv = function(data.x, data.y, folds=5, repeats = 3, tune_grid, ...) {
  stopifnot(length(data.y) == nrow(data.x))
  stopifnot(is.matrix(data.x) & is.vector(data.y))

  ll_cv_train = function(...) {
    index = sample(nrow(data.x), floor(nrow(data.x) * (1 - 1 / folds)), replace=FALSE)
    model <- mx.mlp(data.x[index,], data.y[index], hidden_node=tune_grid[["hidden_nodes"]], 
                    out_node=2, out_activation="softmax", activation = "sigmoid",
                    num.round=tune_grid[["num_rounds"]], array.batch.size=15, 
                    learning.rate=tune_grid[["learning_rate"]], 
                    momentum=0.9, array.layout="rowmajor",
                    dropout=0, initializer=mx.init.normal(1),
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


tune_table = expand.grid(num_rounds = c(10, 20, 40), hidden_nodes = c(10, 20), 
                         learning_rate = c(0.05, 0.1, 0.2))


tune_table$results = apply(tune_table, 1, 
                           FUN = function(x) mx_binaryclass_cv(data.x, data.y, folds=5, repeats = 3, 
                                                               tune_grid = x))
tune_table

