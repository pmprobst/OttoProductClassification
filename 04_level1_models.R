# Level 1 Models - Base Learners

source("config.R")
source("03_cross_validation.R")

# Load required libraries
if (!require("randomForest")) install.packages("randomForest")
if (!require("xgboost")) install.packages("xgboost")
if (!require("extraTrees")) install.packages("extraTrees")
if (!require("e1071")) install.packages("e1071")
if (!require("nnet")) install.packages("nnet")
if (!require("class")) install.packages("class")
if (!require("keras")) install.packages("keras")

library(randomForest)
library(xgboost)
library(extraTrees)
library(e1071)
library(nnet)
library(class)

# Model 1: Random Forest
train_rf <- function(X_train, y_train, ntree = N_TREES_RF, mtry = sqrt(ncol(X_train)), 
                    seed = SEED) {
  set.seed(seed)
  y_factor <- factor(y_train, levels = CLASSES)
  model <- randomForest(X_train, y_factor, ntree = ntree, mtry = mtry)
  return(model)
}

predict_rf <- function(model, X_test) {
  predictions <- predict(model, X_test, type = "prob")
  # Ensure all classes are present
  pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(CLASSES))
  colnames(pred_matrix) <- CLASSES
  pred_matrix[, colnames(predictions)] <- predictions
  return(pred_matrix)
}

# Model 2: Logistic Regression (on log-transformed data)
train_logreg <- function(X_train, y_train, seed = SEED) {
  set.seed(seed)
  y_factor <- factor(y_train, levels = CLASSES)
  # Use multinom from nnet for multi-class logistic regression
  model <- multinom(y_factor ~ ., data = data.frame(X_train), 
                    MaxNWts = 10000, trace = FALSE)
  return(model)
}

predict_logreg <- function(model, X_test) {
  predictions <- predict(model, newdata = data.frame(X_test), type = "probs")
  if (is.vector(predictions)) {
    # If only one class predicted, convert to matrix
    pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(CLASSES))
    colnames(pred_matrix) <- CLASSES
    pred_matrix[, names(predictions)[1]] <- 1
    return(pred_matrix)
  }
  # Ensure all classes are present
  pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(CLASSES))
  colnames(pred_matrix) <- CLASSES
  pred_matrix[, colnames(predictions)] <- predictions
  return(pred_matrix)
}

# Model 3: Extra Trees
train_et <- function(X_train, y_train, ntree = N_TREES_ET, mtry = sqrt(ncol(X_train)),
                    seed = SEED) {
  set.seed(seed)
  y_factor <- factor(y_train, levels = CLASSES)
  model <- extraTrees(X_train, y_factor, ntree = ntree, mtry = mtry)
  return(model)
}

predict_et <- function(model, X_test) {
  predictions <- predict(model, X_test, probability = TRUE)
  # Ensure all classes are present
  pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(CLASSES))
  colnames(pred_matrix) <- CLASSES
  pred_matrix[, colnames(predictions)] <- predictions
  return(pred_matrix)
}

# Model 4: K-Nearest Neighbors
train_knn <- function(X_train, y_train, k = 5, seed = SEED) {
  set.seed(seed)
  # KNN doesn't need training, just store data
  return(list(X_train = X_train, y_train = y_train, k = k))
}

predict_knn <- function(model, X_test) {
  # Scale features for KNN
  X_train_scaled <- scale(model$X_train)
  X_test_scaled <- scale(X_test, 
                         center = attr(X_train_scaled, "scaled:center"),
                         scale = attr(X_train_scaled, "scaled:scale"))
  
  # Predict
  knn_pred <- knn(X_train_scaled, X_test_scaled, model$y_train, 
                  k = model$k, prob = TRUE)
  
  # Get probabilities
  pred_factor <- as.factor(knn_pred)
  pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(CLASSES))
  colnames(pred_matrix) <- CLASSES
  
  # Convert predictions to probability matrix
  for (i in 1:nrow(X_test)) {
    pred_matrix[i, as.character(knn_pred[i])] <- attr(knn_pred, "prob")[i]
    # Distribute remaining probability uniformly
    remaining <- (1 - attr(knn_pred, "prob")[i]) / (length(CLASSES) - 1)
    pred_matrix[i, ] <- pred_matrix[i, ] + remaining
    pred_matrix[i, as.character(knn_pred[i])] <- attr(knn_pred, "prob")[i]
  }
  
  return(pred_matrix)
}

# Model 5: Multinomial Naive Bayes
train_nb <- function(X_train, y_train, seed = SEED) {
  set.seed(seed)
  y_factor <- factor(y_train, levels = CLASSES)
  # Convert to data frame for naiveBayes
  train_df <- data.frame(X_train)
  model <- naiveBayes(y_factor ~ ., data = train_df)
  return(model)
}

predict_nb <- function(model, X_test) {
  predictions <- predict(model, newdata = data.frame(X_test), type = "raw")
  # Ensure all classes are present
  pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(CLASSES))
  colnames(pred_matrix) <- CLASSES
  pred_matrix[, colnames(predictions)] <- predictions
  return(pred_matrix)
}

# Model 6: XGBoost
train_xgb <- function(X_train, y_train, nrounds = N_TREES_XGB, max_depth = MAX_DEPTH,
                     eta = LEARNING_RATE, objective = "multi:softprob", 
                     num_class = N_CLASSES, seed = SEED) {
  set.seed(seed)
  
  # Convert target to numeric (0-indexed)
  y_numeric <- as.numeric(factor(y_train, levels = CLASSES)) - 1
  
  # Create DMatrix
  dtrain <- xgb.DMatrix(data = X_train, label = y_numeric)
  
  # Train model
  params <- list(
    objective = objective,
    num_class = num_class,
    max_depth = max_depth,
    eta = eta,
    eval_metric = "mlogloss"
  )
  
  model <- xgb.train(params = params, data = dtrain, nrounds = nrounds, 
                    verbose = 0)
  return(model)
}

predict_xgb <- function(model, X_test) {
  dtest <- xgb.DMatrix(data = X_test)
  predictions <- predict(model, dtest)
  pred_matrix <- matrix(predictions, nrow = nrow(X_test), ncol = length(CLASSES), 
                       byrow = TRUE)
  colnames(pred_matrix) <- CLASSES
  return(pred_matrix)
}

# Model 7: Neural Network (using nnet)
train_nnet <- function(X_train, y_train, size = 50, decay = 0.1, maxit = 200, 
                     seed = SEED) {
  set.seed(seed)
  y_factor <- factor(y_train, levels = CLASSES)
  
  # Scale features
  X_train_scaled <- scale(X_train)
  
  model <- nnet(y_factor ~ ., data = data.frame(X_train_scaled), 
               size = size, decay = decay, maxit = maxit, trace = FALSE,
               MaxNWts = 10000)
  
  # Store scaling parameters
  model$scaled_center <- attr(X_train_scaled, "scaled:center")
  model$scaled_scale <- attr(X_train_scaled, "scaled:scale")
  
  return(model)
}

predict_nnet <- function(model, X_test) {
  X_test_scaled <- scale(X_test, center = model$scaled_center, 
                        scale = model$scaled_scale)
  predictions <- predict(model, newdata = data.frame(X_test_scaled), type = "raw")
  
  # Ensure all classes are present
  pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(CLASSES))
  colnames(pred_matrix) <- CLASSES
  pred_matrix[, colnames(predictions)] <- predictions
  return(pred_matrix)
}

# Function to train all Level 1 models
train_level1_models <- function(X_train, y_train, features = NULL, 
                                n_folds = N_FOLDS_L1, seed = SEED) {
  set.seed(seed)
  
  cat("Training Level 1 models...\n\n")
  
  level1_predictions <- list()
  
  # Model 1: Random Forest on raw features
  cat("Model 1: Random Forest (raw features)...\n")
  oof_pred <- cv_predict(X_train, y_train, train_rf, predict_rf, 
                        n_folds = n_folds, seed = seed)
  level1_predictions$rf_raw <- oof_pred
  score <- calculate_cv_score(y_train, oof_pred)
  cat(sprintf("RF Raw CV Score: %.6f\n\n", score))
  
  # Model 2: Logistic Regression on log features
  if (!is.null(features) && !is.null(features$log_X_train)) {
    cat("Model 2: Logistic Regression (log features)...\n")
    oof_pred <- cv_predict(features$log_X_train, y_train, train_logreg, 
                          predict_logreg, n_folds = n_folds, seed = seed)
    level1_predictions$logreg_log <- oof_pred
    score <- calculate_cv_score(y_train, oof_pred)
    cat(sprintf("LogReg Log CV Score: %.6f\n\n", score))
  }
  
  # Model 3: Extra Trees on log features
  if (!is.null(features) && !is.null(features$log_X_train)) {
    cat("Model 3: Extra Trees (log features)...\n")
    oof_pred <- cv_predict(features$log_X_train, y_train, train_et, predict_et,
                          n_folds = n_folds, seed = seed)
    level1_predictions$et_log <- oof_pred
    score <- calculate_cv_score(y_train, oof_pred)
    cat(sprintf("ET Log CV Score: %.6f\n\n", score))
  }
  
  # Model 4: KNN on scaled log features
  if (!is.null(features) && !is.null(features$scaled_log_X_train)) {
    cat("Model 4: KNN (scaled log features)...\n")
    train_knn_wrapper <- function(X, y, k = 5, ...) train_knn(X, y, k = k)
    predict_knn_wrapper <- function(model, X, ...) predict_knn(model, X)
    
    oof_pred <- cv_predict(features$scaled_log_X_train, y_train, 
                          function(X, y) train_knn_wrapper(X, y, k = 5),
                          predict_knn_wrapper, n_folds = n_folds, seed = seed)
    level1_predictions$knn_scaled_log <- oof_pred
    score <- calculate_cv_score(y_train, oof_pred)
    cat(sprintf("KNN Scaled Log CV Score: %.6f\n\n", score))
  }
  
  # Model 5: Naive Bayes on log features
  if (!is.null(features) && !is.null(features$log_X_train)) {
    cat("Model 5: Naive Bayes (log features)...\n")
    oof_pred <- cv_predict(features$log_X_train, y_train, train_nb, predict_nb,
                          n_folds = n_folds, seed = seed)
    level1_predictions$nb_log <- oof_pred
    score <- calculate_cv_score(y_train, oof_pred)
    cat(sprintf("NB Log CV Score: %.6f\n\n", score))
  }
  
  # Model 6: XGBoost on raw features
  cat("Model 6: XGBoost (raw features)...\n")
  oof_pred <- cv_predict(X_train, y_train, train_xgb, predict_xgb,
                        n_folds = n_folds, seed = seed)
  level1_predictions$xgb_raw <- oof_pred
  score <- calculate_cv_score(y_train, oof_pred)
  cat(sprintf("XGB Raw CV Score: %.6f\n\n", score))
  
  # Model 7: Neural Network on scaled log features
  if (!is.null(features) && !is.null(features$scaled_log_X_train)) {
    cat("Model 7: Neural Network (scaled log features)...\n")
    oof_pred <- cv_predict(features$scaled_log_X_train, y_train, train_nnet, 
                          predict_nnet, n_folds = n_folds, seed = seed)
    level1_predictions$nnet_scaled_log <- oof_pred
    score <- calculate_cv_score(y_train, oof_pred)
    cat(sprintf("NN Scaled Log CV Score: %.6f\n\n", score))
  }
  
  # Additional KNN models with different k values
  if (!is.null(features) && !is.null(features$scaled_log_X_train)) {
    cat("Training additional KNN models with different k values...\n")
    for (k in c(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)) {
      cat(sprintf("KNN with k=%d...\n", k))
      oof_pred <- cv_predict(features$scaled_log_X_train, y_train,
                            function(X, y) train_knn(X, y, k = k),
                            predict_knn, n_folds = n_folds, seed = seed)
      level1_predictions[[paste0("knn_k", k)]] <- oof_pred
      score <- calculate_cv_score(y_train, oof_pred)
      cat(sprintf("KNN k=%d CV Score: %.6f\n", k, score))
    }
  }
  
  cat("\nLevel 1 model training complete!\n")
  return(level1_predictions)
}

