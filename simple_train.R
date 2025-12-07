# Simplified Training Script - 2-Level Stacking
# Core principle: Train diverse models, use their predictions as features for a meta-model

source("simple_config.R")

# Load libraries
library(data.table)
library(randomForest)
library(xgboost)
library(nnet)
library(caret)

# Load data
cat("Loading data...\n")
train <- fread(TRAIN_FILE)
test <- fread(TEST_FILE)

# Extract features and target
feature_cols <- grep("^feat_", colnames(train), value = TRUE)
X_train <- as.matrix(train[, ..feature_cols])
y_train <- train$target
X_test <- as.matrix(test[, ..feature_cols])
test_id <- test$id

cat(sprintf("Training: %d samples, %d features\n", nrow(X_train), ncol(X_train)))
cat(sprintf("Test: %d samples\n", nrow(X_test)))

# Convert target to factor
y_factor <- factor(y_train, levels = CLASSES)

# Multi-class log loss function
logloss <- function(y_true, y_pred) {
  y_true_num <- as.numeric(factor(y_true, levels = CLASSES))
  y_pred <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
  -mean(log(y_pred[cbind(1:nrow(y_pred), y_true_num)]))
}

# Create CV folds
set.seed(SEED)
folds <- createFolds(y_train, k = N_FOLDS, list = TRUE)

# ============================================================================
# LEVEL 1: Train diverse base models and get out-of-fold predictions
# ============================================================================
cat("\n" , paste(rep("=", 60), collapse = ""), "\n")
cat("LEVEL 1: Training Base Models\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Storage for Level 1 predictions
level1_train_preds <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(level1_train_preds) <- CLASSES
level1_test_preds_list <- list()

# Model 1: Random Forest
cat("Model 1: Random Forest...\n")
rf_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(rf_oof) <- CLASSES
rf_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(rf_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  X_fold_train <- X_train[train_idx, ]
  y_fold_train <- y_factor[train_idx]
  X_fold_test <- X_train[test_idx, ]
  
  # Train RF
  rf_model <- randomForest(X_fold_train, y_fold_train, ntree = 200, mtry = sqrt(ncol(X_train)))
  
  # Predict on fold
  rf_oof[test_idx, ] <- predict(rf_model, X_fold_test, type = "prob")
  
  # Predict on test (average across folds)
  rf_test_preds <- rf_test_preds + predict(rf_model, X_test, type = "prob") / N_FOLDS
}
level1_train_preds <- level1_train_preds + rf_oof
level1_test_preds_list$rf <- rf_test_preds
cat(sprintf("  RF Log Loss: %.4f\n\n", logloss(y_train, rf_oof)))

# Model 2: XGBoost
cat("Model 2: XGBoost...\n")
xgb_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(xgb_oof) <- CLASSES
xgb_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(xgb_test_preds) <- CLASSES

y_numeric <- as.numeric(y_factor) - 1  # XGBoost needs 0-indexed

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  X_fold_train <- X_train[train_idx, ]
  y_fold_train <- y_numeric[train_idx]
  X_fold_test <- X_train[test_idx, ]
  
  # Train XGBoost
  dtrain <- xgb.DMatrix(data = X_fold_train, label = y_fold_train)
  dtest <- xgb.DMatrix(data = X_fold_test)
  dtest_full <- xgb.DMatrix(data = X_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = 6,
    eta = 0.1,
    eval_metric = "mlogloss"
  )
  
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
  
  # Predict
  xgb_oof[test_idx, ] <- matrix(predict(xgb_model, dtest), ncol = N_CLASSES, byrow = TRUE)
  xgb_test_preds <- xgb_test_preds + matrix(predict(xgb_model, dtest_full), ncol = N_CLASSES, byrow = TRUE) / N_FOLDS
}
level1_train_preds <- level1_train_preds + xgb_oof
level1_test_preds_list$xgb <- xgb_test_preds
cat(sprintf("  XGB Log Loss: %.4f\n\n", logloss(y_train, xgb_oof)))

# Model 3: Neural Network (on log-transformed features)
cat("Model 3: Neural Network (log features)...\n")
log_X_train <- log(X_train + 1)
log_X_test <- log(X_test + 1)

# Scale features
log_X_train_scaled <- scale(log_X_train)
log_X_test_scaled <- scale(log_X_test, 
                           center = attr(log_X_train_scaled, "scaled:center"),
                           scale = attr(log_X_train_scaled, "scaled:scale"))

nn_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(nn_oof) <- CLASSES
nn_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(nn_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  X_fold_train <- log_X_train_scaled[train_idx, ]
  y_fold_train <- y_factor[train_idx]
  X_fold_test <- log_X_train_scaled[test_idx, ]
  
  # Train NN
  nn_model <- nnet(y_fold_train ~ ., 
                   data = data.frame(X_fold_train, y_fold_train),
                   size = 50, decay = 0.1, maxit = 200, trace = FALSE, MaxNWts = 10000)
  
  # Predict
  nn_oof[test_idx, ] <- predict(nn_model, newdata = data.frame(X_fold_test), type = "raw")
  nn_test_preds <- nn_test_preds + predict(nn_model, newdata = data.frame(log_X_test_scaled), type = "raw") / N_FOLDS
}
level1_train_preds <- level1_train_preds + nn_oof
level1_test_preds_list$nn <- nn_test_preds
cat(sprintf("  NN Log Loss: %.4f\n\n", logloss(y_train, nn_oof)))

# Average Level 1 predictions
level1_train_preds <- level1_train_preds / 3
cat(sprintf("Level 1 Average Log Loss: %.4f\n", logloss(y_train, level1_train_preds)))

# ============================================================================
# LEVEL 2: Train meta-model on Level 1 predictions
# ============================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("LEVEL 2: Training Meta-Model (XGBoost on Level 1 predictions)\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Use Level 1 predictions as features for Level 2
meta_train <- level1_train_preds
meta_test <- (level1_test_preds_list$rf + level1_test_preds_list$xgb + level1_test_preds_list$nn) / 3

# Train Level 2 XGBoost with CV
level2_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(level2_oof) <- CLASSES
level2_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(level2_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  meta_fold_train <- meta_train[train_idx, ]
  y_fold_train <- y_numeric[train_idx]
  meta_fold_test <- meta_train[test_idx, ]
  
  dtrain <- xgb.DMatrix(data = meta_fold_train, label = y_fold_train)
  dtest <- xgb.DMatrix(data = meta_fold_test)
  dtest_full <- xgb.DMatrix(data = meta_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = 4,
    eta = 0.05,
    eval_metric = "mlogloss"
  )
  
  level2_model <- xgb.train(params = params, data = dtrain, nrounds = 150, verbose = 0)
  
  level2_oof[test_idx, ] <- matrix(predict(level2_model, dtest), ncol = N_CLASSES, byrow = TRUE)
  level2_test_preds <- level2_test_preds + matrix(predict(level2_model, dtest_full), ncol = N_CLASSES, byrow = TRUE) / N_FOLDS
}

final_score <- logloss(y_train, level2_oof)
cat(sprintf("\nFinal Stacked Model Log Loss: %.4f\n", final_score))

# ============================================================================
# Generate submission
# ============================================================================
cat("\nGenerating submission file...\n")
submission <- data.frame(id = test_id)
for (class in CLASSES) {
  submission[[class]] <- level2_test_preds[, class]
}

write.csv(submission, SUBMISSION_FILE, row.names = FALSE)
cat(sprintf("Submission saved to: %s\n", SUBMISSION_FILE))
cat("\nDone!\n")

