# Level 2 Models - Meta Learners

source("config.R")
source("03_cross_validation.R")
source("04_level1_models.R")

# Function to combine Level 1 predictions into meta-features
create_meta_features <- function(level1_predictions, additional_features = NULL) {
  # Stack all Level 1 predictions
  meta_features <- do.call(cbind, level1_predictions)
  
  # Add additional engineered features if provided
  if (!is.null(additional_features)) {
    if (is.matrix(additional_features)) {
      meta_features <- cbind(meta_features, additional_features)
    } else if (is.list(additional_features)) {
      # Flatten list of features
      for (feat in additional_features) {
        if (is.matrix(feat)) {
          meta_features <- cbind(meta_features, feat)
        }
      }
    }
  }
  
  return(meta_features)
}

# Level 2 Model 1: XGBoost
train_xgb_l2 <- function(X_train, y_train, nrounds = 200, max_depth = 4,
                         eta = 0.05, seed = SEED) {
  set.seed(seed)
  y_numeric <- as.numeric(factor(y_train, levels = CLASSES)) - 1
  dtrain <- xgb.DMatrix(data = X_train, label = y_numeric)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = max_depth,
    eta = eta,
    eval_metric = "mlogloss"
  )
  
  model <- xgb.train(params = params, data = dtrain, nrounds = nrounds, 
                    verbose = 0)
  return(model)
}

predict_xgb_l2 <- function(model, X_test) {
  dtest <- xgb.DMatrix(data = X_test)
  predictions <- predict(model, dtest)
  pred_matrix <- matrix(predictions, nrow = nrow(X_test), ncol = length(CLASSES),
                       byrow = TRUE)
  colnames(pred_matrix) <- CLASSES
  return(pred_matrix)
}

# Level 2 Model 2: Neural Network
train_nnet_l2 <- function(X_train, y_train, size = 100, decay = 0.1, maxit = 300,
                         seed = SEED) {
  set.seed(seed)
  y_factor <- factor(y_train, levels = CLASSES)
  
  # Scale features
  X_train_scaled <- scale(X_train)
  
  model <- nnet(y_factor ~ ., data = data.frame(X_train_scaled),
               size = size, decay = decay, maxit = maxit, trace = FALSE,
               MaxNWts = 20000)
  
  model$scaled_center <- attr(X_train_scaled, "scaled:center")
  model$scaled_scale <- attr(X_train_scaled, "scaled:scale")
  
  return(model)
}

predict_nnet_l2 <- function(model, X_test) {
  X_test_scaled <- scale(X_test, center = model$scaled_center,
                        scale = model$scaled_scale)
  predictions <- predict(model, newdata = data.frame(X_test_scaled), type = "raw")
  
  pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(CLASSES))
  colnames(pred_matrix) <- CLASSES
  pred_matrix[, colnames(predictions)] <- predictions
  return(pred_matrix)
}

# Level 2 Model 3: AdaBoost with Extra Trees
train_adaboost_et <- function(X_train, y_train, n_estimators = 100, 
                              max_depth = 3, seed = SEED) {
  set.seed(seed)
  y_factor <- factor(y_train, levels = CLASSES)
  
  # Use Extra Trees as base estimator with boosting-like approach
  # We'll use a bagged Extra Trees approach
  n_bags <- n_estimators
  models <- list()
  
  for (i in 1:n_bags) {
    # Sample with replacement
    sample_idx <- sample(nrow(X_train), nrow(X_train), replace = TRUE)
    X_sample <- X_train[sample_idx, , drop = FALSE]
    y_sample <- y_factor[sample_idx]
    
    # Train Extra Trees
    model <- extraTrees(X_sample, y_sample, ntree = 50, mtry = sqrt(ncol(X_train)))
    models[[i]] <- model
  }
  
  return(list(models = models, n_bags = n_bags))
}

predict_adaboost_et <- function(model, X_test) {
  n_bags <- model$n_bags
  all_predictions <- array(0, dim = c(nrow(X_test), length(CLASSES), n_bags))
  
  for (i in 1:n_bags) {
    pred <- predict(model$models[[i]], X_test, probability = TRUE)
    all_predictions[, , i] <- pred
  }
  
  # Average predictions
  pred_matrix <- apply(all_predictions, c(1, 2), mean)
  colnames(pred_matrix) <- CLASSES
  
  return(pred_matrix)
}

# Train all Level 2 models
train_level2_models <- function(meta_features, y_train, additional_features = NULL,
                                n_folds = N_FOLDS_L2, seed = SEED) {
  set.seed(seed)
  
  cat("Training Level 2 models...\n\n")
  
  # Create full meta-feature set
  X_meta <- create_meta_features(meta_features, additional_features)
  
  level2_predictions <- list()
  
  # Model 1: XGBoost
  cat("Level 2 Model 1: XGBoost...\n")
  oof_pred <- cv_predict(X_meta, y_train, train_xgb_l2, predict_xgb_l2,
                        n_folds = n_folds, seed = seed)
  level2_predictions$xgb <- oof_pred
  score <- calculate_cv_score(y_train, oof_pred)
  cat(sprintf("L2 XGBoost CV Score: %.6f\n\n", score))
  
  # Model 2: Neural Network
  cat("Level 2 Model 2: Neural Network...\n")
  oof_pred <- cv_predict(X_meta, y_train, train_nnet_l2, predict_nnet_l2,
                        n_folds = n_folds, seed = seed)
  level2_predictions$nnet <- oof_pred
  score <- calculate_cv_score(y_train, oof_pred)
  cat(sprintf("L2 Neural Network CV Score: %.6f\n\n", score))
  
  # Model 3: AdaBoost with Extra Trees
  cat("Level 2 Model 3: AdaBoost with Extra Trees...\n")
  oof_pred <- cv_predict(X_meta, y_train, train_adaboost_et, predict_adaboost_et,
                        n_folds = n_folds, seed = seed)
  level2_predictions$adaboost_et <- oof_pred
  score <- calculate_cv_score(y_train, oof_pred)
  cat(sprintf("L2 AdaBoost ET CV Score: %.6f\n\n", score))
  
  cat("Level 2 model training complete!\n")
  return(level2_predictions)
}

