# Improved Training Script - Enhanced Stacking
# Target: Log Loss < 0.7

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR <- "data"
TRAIN_FILE <- file.path(DATA_DIR, "train.csv")
TEST_FILE <- file.path(DATA_DIR, "test.csv")
OUTPUT_DIR <- "output"
SUBMISSION_FILE <- file.path(OUTPUT_DIR, "submission.csv")

# Create output directory
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# Cross-validation
N_FOLDS <- 5
SEED <- 42

# Class names
CLASSES <- paste0("Class_", 1:9)
N_CLASSES <- 9

# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Load libraries
library(data.table)
library(randomForest)
library(xgboost)
library(nnet)
library(caret)
library(e1071)


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
y_numeric <- as.numeric(y_factor) - 1  # 0-indexed for XGBoost

# Multi-class log loss function
logloss <- function(y_true, y_pred) {
  # Normalize predictions to sum to 1
  y_pred <- y_pred / rowSums(y_pred)
  y_pred <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
  y_true_num <- as.numeric(factor(y_true, levels = CLASSES))
  -mean(log(y_pred[cbind(1:nrow(y_pred), y_true_num)]))
}

# Normalize predictions function
normalize_preds <- function(preds) {
  preds <- pmax(preds, 0)  # Ensure non-negative
  preds / rowSums(preds)  # Normalize to sum to 1
}

# Create CV folds
set.seed(SEED)
folds <- createFolds(y_train, k = N_FOLDS, list = TRUE)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
cat("\nCreating feature transformations...\n")

# Log transformation
log_X_train <- log(X_train + 1)
log_X_test <- log(X_test + 1)

# Sqrt transformation
sqrt_X_train <- sqrt(X_train + 3/8)
sqrt_X_test <- sqrt(X_test + 3/8)

# Scaled features
scaled_X_train <- scale(X_train)
scaled_X_test <- scale(X_test, 
                       center = attr(scaled_X_train, "scaled:center"),
                       scale = attr(scaled_X_train, "scaled:scale"))

# Scaled log features
scaled_log_X_train <- scale(log_X_train)
scaled_log_X_test <- scale(log_X_test,
                          center = attr(scaled_log_X_train, "scaled:center"),
                          scale = attr(scaled_log_X_train, "scaled:scale"))

# Row statistics
row_stats_train <- cbind(
  rowSums(X_train),
  rowSums(X_train > 0),
  rowSums(X_train == 0),
  rowMeans(X_train),
  apply(X_train, 1, max),
  apply(X_train, 1, min)
)
row_stats_test <- cbind(
  rowSums(X_test),
  rowSums(X_test > 0),
  rowSums(X_test == 0),
  rowMeans(X_test),
  apply(X_test, 1, max),
  apply(X_test, 1, min)
)

cat("Feature engineering complete!\n\n")

# ============================================================================
# LEVEL 1: Train diverse base models
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("LEVEL 1: Training Base Models\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

level1_train_preds_list <- list()
level1_test_preds_list <- list()

# Model 1: Random Forest (raw features)
cat("Model 1: Random Forest (raw)...\n")
rf_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(rf_oof) <- CLASSES
rf_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(rf_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  rf_model <- randomForest(X_train[train_idx, ], y_factor[train_idx], 
                          ntree = 500, mtry = sqrt(ncol(X_train)), nodesize = 1)
  
  rf_oof[test_idx, ] <- normalize_preds(predict(rf_model, X_train[test_idx, ], type = "prob"))
  rf_test_preds <- rf_test_preds + normalize_preds(predict(rf_model, X_test, type = "prob")) / N_FOLDS
}
level1_train_preds_list$rf <- normalize_preds(rf_oof)
level1_test_preds_list$rf <- normalize_preds(rf_test_preds)
cat(sprintf("  RF Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds_list$rf)))

# Model 2: XGBoost (raw features) - More rounds
cat("Model 2: XGBoost (raw, 300 rounds)...\n")
xgb_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(xgb_oof) <- CLASSES
xgb_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(xgb_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  dtrain <- xgb.DMatrix(data = X_train[train_idx, ], label = y_numeric[train_idx])
  dtest <- xgb.DMatrix(data = X_train[test_idx, ])
  dtest_full <- xgb.DMatrix(data = X_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = 8,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    eval_metric = "mlogloss"
  )
  
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 300, verbose = 0)
  
  xgb_oof[test_idx, ] <- normalize_preds(matrix(predict(xgb_model, dtest), ncol = N_CLASSES, byrow = TRUE))
  xgb_test_preds <- xgb_test_preds + normalize_preds(matrix(predict(xgb_model, dtest_full), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
}
level1_train_preds_list$xgb_raw <- normalize_preds(xgb_oof)
level1_test_preds_list$xgb_raw <- normalize_preds(xgb_test_preds)
cat(sprintf("  XGB Raw Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds_list$xgb_raw)))

# Model 3: XGBoost (log features)
cat("Model 3: XGBoost (log features, 300 rounds)...\n")
xgb_log_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(xgb_log_oof) <- CLASSES
xgb_log_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(xgb_log_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  dtrain <- xgb.DMatrix(data = log_X_train[train_idx, ], label = y_numeric[train_idx])
  dtest <- xgb.DMatrix(data = log_X_train[test_idx, ])
  dtest_full <- xgb.DMatrix(data = log_X_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = 8,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "mlogloss"
  )
  
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 300, verbose = 0)
  
  xgb_log_oof[test_idx, ] <- normalize_preds(matrix(predict(xgb_model, dtest), ncol = N_CLASSES, byrow = TRUE))
  xgb_log_test_preds <- xgb_log_test_preds + normalize_preds(matrix(predict(xgb_model, dtest_full), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
}
level1_train_preds_list$xgb_log <- normalize_preds(xgb_log_oof)
level1_test_preds_list$xgb_log <- normalize_preds(xgb_log_test_preds)
cat(sprintf("  XGB Log Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds_list$xgb_log)))

# Model 4: Neural Network (scaled log features)
cat("Model 4: Neural Network (scaled log)...\n")
nn_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(nn_oof) <- CLASSES
nn_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(nn_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  # Create data frame with proper column names
  train_df <- data.frame(scaled_log_X_train[train_idx, ])
  train_df$target <- y_factor[train_idx]
  
  nn_model <- nnet(target ~ ., 
                   data = train_df,
                   size = 100, decay = 0.1, maxit = 300, trace = FALSE, MaxNWts = 20000)
  
  nn_oof[test_idx, ] <- normalize_preds(predict(nn_model, newdata = data.frame(scaled_log_X_train[test_idx, ]), type = "raw"))
  nn_test_preds <- nn_test_preds + normalize_preds(predict(nn_model, newdata = data.frame(scaled_log_X_test), type = "raw")) / N_FOLDS
}
level1_train_preds_list$nn <- normalize_preds(nn_oof)
level1_test_preds_list$nn <- normalize_preds(nn_test_preds)
cat(sprintf("  NN Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds_list$nn)))

# Model 5: Naive Bayes (log features)
cat("Model 5: Naive Bayes (log features)...\n")
nb_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(nb_oof) <- CLASSES
nb_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(nb_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  # Create data frame with proper column names
  train_df <- data.frame(log_X_train[train_idx, ])
  train_df$target <- y_factor[train_idx]
  
  nb_model <- naiveBayes(target ~ ., data = train_df)
  
  nb_oof[test_idx, ] <- normalize_preds(predict(nb_model, newdata = data.frame(log_X_train[test_idx, ]), type = "raw"))
  nb_test_preds <- nb_test_preds + normalize_preds(predict(nb_model, newdata = data.frame(log_X_test), type = "raw")) / N_FOLDS
}
level1_train_preds_list$nb <- normalize_preds(nb_oof)
level1_test_preds_list$nb <- normalize_preds(nb_test_preds)
cat(sprintf("  NB Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds_list$nb)))

# Combine Level 1 predictions
cat("Combining Level 1 predictions...\n")
n_models_l1 <- length(level1_train_preds_list)
level1_train_combined <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
level1_test_combined <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(level1_train_combined) <- CLASSES
colnames(level1_test_combined) <- CLASSES

for (model_name in names(level1_train_preds_list)) {
  level1_train_combined <- level1_train_combined + level1_train_preds_list[[model_name]]
  level1_test_combined <- level1_test_combined + level1_test_preds_list[[model_name]]
}
level1_train_combined <- normalize_preds(level1_train_combined / n_models_l1)
level1_test_combined <- normalize_preds(level1_test_combined / n_models_l1)

cat(sprintf("Level 1 Average Log Loss: %.4f\n\n", logloss(y_train, level1_train_combined)))

# ============================================================================
# LEVEL 2: Train meta-model on Level 1 predictions + original features
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("LEVEL 2: Training Meta-Model\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Combine Level 1 predictions with row statistics as additional features
meta_train <- cbind(level1_train_combined, row_stats_train)
meta_test <- cbind(level1_test_combined, row_stats_test)

# Train Level 2 XGBoost with CV
level2_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(level2_oof) <- CLASSES
level2_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(level2_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  test_idx <- folds[[fold]]
  
  dtrain <- xgb.DMatrix(data = meta_train[train_idx, ], label = y_numeric[train_idx])
  dtest <- xgb.DMatrix(data = meta_train[test_idx, ])
  dtest_full <- xgb.DMatrix(data = meta_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = 6,
    eta = 0.03,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    eval_metric = "mlogloss"
  )
  
  level2_model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0)
  
  level2_oof[test_idx, ] <- normalize_preds(matrix(predict(level2_model, dtest), ncol = N_CLASSES, byrow = TRUE))
  level2_test_preds <- level2_test_preds + normalize_preds(matrix(predict(level2_model, dtest_full), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
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

