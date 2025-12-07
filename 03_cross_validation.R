# Cross-Validation Framework

source("config.R")

# Load required libraries
if (!require("caret")) install.packages("caret")
library(caret)

# Multi-class log loss function
multiclass_logloss <- function(y_true, y_pred) {
  # y_true: factor or character vector of true labels
  # y_pred: matrix of predicted probabilities (n_samples x n_classes)
  
  # Convert y_true to numeric indices
  if (is.factor(y_true)) {
    y_true_numeric <- as.numeric(y_true)
  } else {
    y_true_numeric <- match(y_true, CLASSES)
  }
  
  # Clip predictions to avoid log(0)
  y_pred <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
  
  # Calculate log loss
  n <- length(y_true_numeric)
  logloss <- 0
  
  for (i in 1:n) {
    logloss <- logloss - log(y_pred[i, y_true_numeric[i]])
  }
  
  return(logloss / n)
}

# Create stratified K-fold indices
create_folds <- function(y, n_folds = N_FOLDS_L1, seed = SEED) {
  set.seed(seed)
  
  # Use caret's createFolds for stratified splitting
  folds <- createFolds(y, k = n_folds, list = TRUE, returnTrain = FALSE)
  
  # Convert to train/test indices format
  fold_indices <- list()
  for (i in 1:n_folds) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:length(y), test_idx)
    fold_indices[[i]] <- list(train = train_idx, test = test_idx)
  }
  
  return(fold_indices)
}

# Cross-validation wrapper for model training
cv_predict <- function(X, y, model_func, predict_func, n_folds = N_FOLDS_L1, 
                       seed = SEED, ...) {
  # model_func: function that trains a model, takes (X_train, y_train, ...)
  # predict_func: function that predicts, takes (model, X_test, ...)
  
  set.seed(seed)
  
  folds <- create_folds(y, n_folds = n_folds, seed = seed)
  n_samples <- nrow(X)
  n_classes <- length(CLASSES)
  
  # Initialize out-of-fold predictions
  oof_predictions <- matrix(0, nrow = n_samples, ncol = n_classes)
  colnames(oof_predictions) <- CLASSES
  
  cat(sprintf("Starting %d-fold cross-validation...\n", n_folds))
  
  for (fold in 1:n_folds) {
    cat(sprintf("Fold %d/%d...\n", fold, n_folds))
    
    train_idx <- folds[[fold]]$train
    test_idx <- folds[[fold]]$test
    
    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test <- X[test_idx, , drop = FALSE]
    
    # Train model
    model <- model_func(X_train, y_train, ...)
    
    # Predict on test fold
    predictions <- predict_func(model, X_test, ...)
    
    # Store predictions
    oof_predictions[test_idx, ] <- predictions
  }
  
  cat("Cross-validation complete!\n")
  return(oof_predictions)
}

# Calculate CV score
calculate_cv_score <- function(y_true, oof_predictions) {
  logloss <- multiclass_logloss(y_true, oof_predictions)
  cat(sprintf("CV Log Loss: %.6f\n", logloss))
  return(logloss)
}

