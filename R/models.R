# Model training with K-fold CV: each function returns OOF + test predictions
# Depends: R/config.R, R/utils.R; packages: data.table, randomForest, xgboost, nnet, e1071, caret

#' Random Forest, raw features. Returns list(oof, test_preds) with matrices.
train_rf_cv <- function(X_train, y_factor, X_test, folds) {
  oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
  colnames(oof) <- CLASSES
  test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
  colnames(test_preds) <- CLASSES
  for (fold in seq_len(N_FOLDS)) {
    train_idx <- setdiff(seq_len(nrow(X_train)), folds[[fold]])
    test_idx <- folds[[fold]]
    model <- randomForest::randomForest(
      X_train[train_idx, ], y_factor[train_idx],
      ntree = 500, mtry = sqrt(ncol(X_train)), nodesize = 1
    )
    oof[test_idx, ] <- normalize_preds(predict(model, X_train[test_idx, ], type = "prob"))
    test_preds <- test_preds + normalize_preds(predict(model, X_test, type = "prob")) / N_FOLDS
  }
  list(oof = normalize_preds(oof), test_preds = normalize_preds(test_preds))
}

#' XGBoost on raw features (stacking-style: 300 rounds, no early stop).
train_xgb_raw_cv <- function(X_train, y_numeric, X_test, folds) {
  oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
  colnames(oof) <- CLASSES
  test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
  colnames(test_preds) <- CLASSES
  params <- list(
    objective = "multi:softprob", num_class = N_CLASSES,
    max_depth = 8, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8,
    min_child_weight = 1, eval_metric = "mlogloss"
  )
  for (fold in seq_len(N_FOLDS)) {
    train_idx <- setdiff(seq_len(nrow(X_train)), folds[[fold]])
    test_idx <- folds[[fold]]
    dtrain <- xgboost::xgb.DMatrix(data = X_train[train_idx, ], label = y_numeric[train_idx])
    dtest <- xgboost::xgb.DMatrix(data = X_train[test_idx, ])
    dtest_full <- xgboost::xgb.DMatrix(data = X_test)
    model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = 300, verbose = 0)
    oof[test_idx, ] <- normalize_preds(matrix(predict(model, dtest), ncol = N_CLASSES, byrow = TRUE))
    test_preds <- test_preds + normalize_preds(matrix(predict(model, dtest_full), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
  }
  list(oof = normalize_preds(oof), test_preds = normalize_preds(test_preds))
}

#' XGBoost on log features.
train_xgb_log_cv <- function(log_X_train, y_numeric, log_X_test, folds) {
  oof <- matrix(0, nrow = nrow(log_X_train), ncol = N_CLASSES)
  colnames(oof) <- CLASSES
  test_preds <- matrix(0, nrow = nrow(log_X_test), ncol = N_CLASSES)
  colnames(test_preds) <- CLASSES
  params <- list(
    objective = "multi:softprob", num_class = N_CLASSES,
    max_depth = 8, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, eval_metric = "mlogloss"
  )
  for (fold in seq_len(N_FOLDS)) {
    train_idx <- setdiff(seq_len(nrow(log_X_train)), folds[[fold]])
    test_idx <- folds[[fold]]
    dtrain <- xgboost::xgb.DMatrix(data = log_X_train[train_idx, ], label = y_numeric[train_idx])
    dtest <- xgboost::xgb.DMatrix(data = log_X_train[test_idx, ])
    dtest_full <- xgboost::xgb.DMatrix(data = log_X_test)
    model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = 300, verbose = 0)
    oof[test_idx, ] <- normalize_preds(matrix(predict(model, dtest), ncol = N_CLASSES, byrow = TRUE))
    test_preds <- test_preds + normalize_preds(matrix(predict(model, dtest_full), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
  }
  list(oof = normalize_preds(oof), test_preds = normalize_preds(test_preds))
}

#' Neural network (nnet) on scaled log features.
train_nn_cv <- function(scaled_log_X_train, y_factor, scaled_log_X_test, folds) {
  oof <- matrix(0, nrow = nrow(scaled_log_X_train), ncol = N_CLASSES)
  colnames(oof) <- CLASSES
  test_preds <- matrix(0, nrow = nrow(scaled_log_X_test), ncol = N_CLASSES)
  colnames(test_preds) <- CLASSES
  for (fold in seq_len(N_FOLDS)) {
    train_idx <- setdiff(seq_len(nrow(scaled_log_X_train)), folds[[fold]])
    test_idx <- folds[[fold]]
    train_df <- as.data.frame(scaled_log_X_train[train_idx, ])
    train_df$target <- y_factor[train_idx]
    model <- nnet::nnet(target ~ ., data = train_df, size = 100, decay = 0.1, maxit = 300, trace = FALSE, MaxNWts = 20000)
    oof[test_idx, ] <- normalize_preds(predict(model, newdata = as.data.frame(scaled_log_X_train[test_idx, ]), type = "raw"))
    test_preds <- test_preds + normalize_preds(predict(model, newdata = as.data.frame(scaled_log_X_test), type = "raw")) / N_FOLDS
  }
  list(oof = normalize_preds(oof), test_preds = normalize_preds(test_preds))
}

#' Naive Bayes on log features.
train_nb_cv <- function(log_X_train, y_factor, log_X_test, folds) {
  oof <- matrix(0, nrow = nrow(log_X_train), ncol = N_CLASSES)
  colnames(oof) <- CLASSES
  test_preds <- matrix(0, nrow = nrow(log_X_test), ncol = N_CLASSES)
  colnames(test_preds) <- CLASSES
  for (fold in seq_len(N_FOLDS)) {
    train_idx <- setdiff(seq_len(nrow(log_X_train)), folds[[fold]])
    test_idx <- folds[[fold]]
    train_df <- as.data.frame(log_X_train[train_idx, ])
    train_df$target <- y_factor[train_idx]
    model <- e1071::naiveBayes(target ~ ., data = train_df)
    oof[test_idx, ] <- normalize_preds(predict(model, newdata = as.data.frame(log_X_train[test_idx, ]), type = "raw"))
    test_preds <- test_preds + normalize_preds(predict(model, newdata = as.data.frame(log_X_test), type = "raw")) / N_FOLDS
  }
  list(oof = normalize_preds(oof), test_preds = normalize_preds(test_preds))
}

#' Multinomial logistic regression on log features.
train_mlr_cv <- function(log_X_train, y_factor, log_X_test, folds) {
  oof <- matrix(0, nrow = nrow(log_X_train), ncol = N_CLASSES)
  colnames(oof) <- CLASSES
  test_preds <- matrix(0, nrow = nrow(log_X_test), ncol = N_CLASSES)
  colnames(test_preds) <- CLASSES
  for (fold in seq_len(N_FOLDS)) {
    train_idx <- setdiff(seq_len(nrow(log_X_train)), folds[[fold]])
    test_idx <- folds[[fold]]
    train_df <- as.data.frame(log_X_train[train_idx, ])
    train_df$target <- y_factor[train_idx]
    model <- nnet::multinom(target ~ ., data = train_df, MaxNWts = 15000, trace = FALSE)
    oof[test_idx, ] <- normalize_preds(predict(model, newdata = as.data.frame(log_X_train[test_idx, ]), type = "probs"))
    test_preds <- test_preds + normalize_preds(predict(model, newdata = as.data.frame(log_X_test), type = "probs")) / N_FOLDS
  }
  list(oof = normalize_preds(oof), test_preds = normalize_preds(test_preds))
}

#' Level 2 meta-model (XGBoost on L1 predictions + row stats).
train_meta_cv <- function(meta_train, y_numeric, meta_test, folds) {
  oof <- matrix(0, nrow = nrow(meta_train), ncol = N_CLASSES)
  colnames(oof) <- CLASSES
  test_preds <- matrix(0, nrow = nrow(meta_test), ncol = N_CLASSES)
  colnames(test_preds) <- CLASSES
  params <- list(
    objective = "multi:softprob", num_class = N_CLASSES,
    max_depth = 6, eta = 0.03, subsample = 0.8, colsample_bytree = 0.8,
    min_child_weight = 3, eval_metric = "mlogloss"
  )
  for (fold in seq_len(N_FOLDS)) {
    train_idx <- setdiff(seq_len(nrow(meta_train)), folds[[fold]])
    test_idx <- folds[[fold]]
    dtrain <- xgboost::xgb.DMatrix(data = meta_train[train_idx, ], label = y_numeric[train_idx])
    dtest <- xgboost::xgb.DMatrix(data = meta_train[test_idx, ])
    dtest_full <- xgboost::xgb.DMatrix(data = meta_test)
    model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0)
    oof[test_idx, ] <- normalize_preds(matrix(predict(model, dtest), ncol = N_CLASSES, byrow = TRUE))
    test_preds <- test_preds + normalize_preds(matrix(predict(model, dtest_full), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
  }
  list(oof = normalize_preds(oof), test_preds = normalize_preds(test_preds))
}

#' XGBoost with early stopping (single-model script style).
train_xgb_early_cv <- function(X_train, y_numeric, X_test, folds) {
  oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
  colnames(oof) <- CLASSES
  test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
  colnames(test_preds) <- CLASSES
  params <- list(
    objective = "multi:softprob", num_class = N_CLASSES, eval_metric = "mlogloss",
    max_depth = 5, eta = 0.05, min_child_weight = 1, subsample = 1, colsample_bytree = 0.3
  )
  for (fold in seq_len(N_FOLDS)) {
    train_idx <- setdiff(seq_len(nrow(X_train)), folds[[fold]])
    test_idx <- folds[[fold]]
    dtrain <- xgboost::xgb.DMatrix(data = X_train[train_idx, ], label = y_numeric[train_idx])
    dval <- xgboost::xgb.DMatrix(data = X_train[test_idx, ], label = y_numeric[test_idx])
    dtest_full <- xgboost::xgb.DMatrix(data = X_test)
    watchlist <- list(train = dtrain, eval = dval)
    set.seed(SEED)
    model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = 2500,
      watchlist = watchlist, early_stopping_rounds = 10, verbose = 0)
    oof[test_idx, ] <- normalize_preds(matrix(predict(model, dval), ncol = N_CLASSES, byrow = TRUE))
    test_preds <- test_preds + normalize_preds(matrix(predict(model, dtest_full), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
  }
  list(oof = normalize_preds(oof), test_preds = normalize_preds(test_preds))
}
