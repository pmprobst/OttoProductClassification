# Winner's Approach - Simplified Single File
# Based on winning solutions from Otto Product Classification competition
# Simplified for quick testing (~15-30 minutes)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR <- "data"
TRAIN_FILE <- file.path(DATA_DIR, "train.csv")
TEST_FILE <- file.path(DATA_DIR, "test.csv")
SAMPLE_SUB_FILE <- file.path(DATA_DIR, "sampleSubmission.csv")
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
# LOAD LIBRARIES
# ============================================================================
cat("Loading libraries...\n")
library(data.table)
library(xgboost)
library(randomForest)
library(nnet)
library(caret)

# ============================================================================
# LOAD DATA
# ============================================================================
cat("\nLoading data...\n")
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

# Convert target to factor and numeric
y_factor <- factor(y_train, levels = CLASSES)
y_numeric <- as.numeric(y_factor) - 1  # 0-indexed for XGBoost

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Multi-class log loss function
logloss <- function(y_true, y_pred) {
  y_pred <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)  # Clip to [1e-15, 1-1e-15]
  y_pred <- y_pred / rowSums(y_pred)  # Normalize
  y_true_num <- as.numeric(factor(y_true, levels = CLASSES))
  -mean(log(y_pred[cbind(1:nrow(y_pred), y_true_num)]))
}

# Normalize predictions to sum to 1
normalize_preds <- function(preds) {
  preds <- pmax(preds, 0)  # Ensure non-negative
  preds / rowSums(preds)  # Normalize
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("FEATURE ENGINEERING\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Log transformation (key feature used by winners)
cat("Creating log features...\n")
log_X_train <- log(X_train + 1)
log_X_test <- log(X_test + 1)

# Row statistics (important aggregations)
cat("Creating row statistics...\n")
row_stats_train <- cbind(
  rowSums(X_train),
  rowMeans(X_train),
  apply(X_train, 1, max),
  apply(X_train, 1, min),
  rowSums(X_train > 0)  # Non-zero count
)
row_stats_test <- cbind(
  rowSums(X_test),
  rowMeans(X_test),
  apply(X_test, 1, max),
  apply(X_test, 1, min),
  rowSums(X_test > 0)
)
colnames(row_stats_train) <- paste0("row_stat_", 1:ncol(row_stats_train))
colnames(row_stats_test) <- paste0("row_stat_", 1:ncol(row_stats_test))

cat("Feature engineering complete!\n\n")

# ============================================================================
# CREATE CV FOLDS
# ============================================================================
set.seed(SEED)
folds <- createFolds(y_train, k = N_FOLDS, list = TRUE)

# ============================================================================
# LEVEL 1: BASE MODELS
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("LEVEL 1: TRAINING BASE MODELS\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

level1_train_preds <- list()
level1_test_preds <- list()

# ----------------------------------------------------------------------------
# Model 1: XGBoost (raw features) - Primary model
# ----------------------------------------------------------------------------
cat("Model 1: XGBoost (raw features)...\n")
xgb1_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(xgb1_oof) <- CLASSES
xgb1_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(xgb1_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  val_idx <- folds[[fold]]
  
  dtrain <- xgb.DMatrix(data = X_train[train_idx, ], label = y_numeric[train_idx])
  dval <- xgb.DMatrix(data = X_train[val_idx, ])
  dtest <- xgb.DMatrix(data = X_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = 6,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    eval_metric = "mlogloss"
  )
  
  model <- xgb.train(params = params, data = dtrain, nrounds = 150, verbose = 0)
  
  xgb1_oof[val_idx, ] <- normalize_preds(
    matrix(predict(model, dval), ncol = N_CLASSES, byrow = TRUE)
  )
  xgb1_test_preds <- xgb1_test_preds + 
    normalize_preds(matrix(predict(model, dtest), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
}

level1_train_preds$xgb_raw <- normalize_preds(xgb1_oof)
level1_test_preds$xgb_raw <- normalize_preds(xgb1_test_preds)
cat(sprintf("  XGBoost Raw Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds$xgb_raw)))

# ----------------------------------------------------------------------------
# Model 2: XGBoost (log features) - Different view of data
# ----------------------------------------------------------------------------
cat("Model 2: XGBoost (log features)...\n")
xgb2_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(xgb2_oof) <- CLASSES
xgb2_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(xgb2_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  val_idx <- folds[[fold]]
  
  dtrain <- xgb.DMatrix(data = log_X_train[train_idx, ], label = y_numeric[train_idx])
  dval <- xgb.DMatrix(data = log_X_train[val_idx, ])
  dtest <- xgb.DMatrix(data = log_X_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = 6,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    eval_metric = "mlogloss"
  )
  
  model <- xgb.train(params = params, data = dtrain, nrounds = 150, verbose = 0)
  
  xgb2_oof[val_idx, ] <- normalize_preds(
    matrix(predict(model, dval), ncol = N_CLASSES, byrow = TRUE)
  )
  xgb2_test_preds <- xgb2_test_preds + 
    normalize_preds(matrix(predict(model, dtest), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
}

level1_train_preds$xgb_log <- normalize_preds(xgb2_oof)
level1_test_preds$xgb_log <- normalize_preds(xgb2_test_preds)
cat(sprintf("  XGBoost Log Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds$xgb_log)))

# ----------------------------------------------------------------------------
# Model 3: Random Forest (raw features) - Tree diversity
# ----------------------------------------------------------------------------
cat("Model 3: Random Forest (raw features)...\n")
rf_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(rf_oof) <- CLASSES
rf_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(rf_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  val_idx <- folds[[fold]]
  
  model <- randomForest(
    X_train[train_idx, ], 
    y_factor[train_idx], 
    ntree = 200, 
    mtry = sqrt(ncol(X_train)), 
    nodesize = 1
  )
  
  rf_oof[val_idx, ] <- normalize_preds(
    predict(model, X_train[val_idx, ], type = "prob")
  )
  rf_test_preds <- rf_test_preds + 
    normalize_preds(predict(model, X_test, type = "prob")) / N_FOLDS
}

level1_train_preds$rf <- normalize_preds(rf_oof)
level1_test_preds$rf <- normalize_preds(rf_test_preds)
cat(sprintf("  Random Forest Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds$rf)))

# ----------------------------------------------------------------------------
# Model 4: Neural Network (log features) - Non-linear patterns
# ----------------------------------------------------------------------------
cat("Model 4: Neural Network (log features)...\n")
nn_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(nn_oof) <- CLASSES
nn_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(nn_test_preds) <- CLASSES

# Scale log features for neural network
scaled_log_X_train <- scale(log_X_train)
scaled_log_X_test <- scale(log_X_test,
                          center = attr(scaled_log_X_train, "scaled:center"),
                          scale = attr(scaled_log_X_train, "scaled:scale"))

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  val_idx <- folds[[fold]]
  
  train_df <- data.frame(scaled_log_X_train[train_idx, ])
  train_df$target <- y_factor[train_idx]
  
  model <- nnet(
    target ~ ., 
    data = train_df,
    size = 50, 
    decay = 0.1, 
    maxit = 200, 
    trace = FALSE, 
    MaxNWts = 15000
  )
  
  nn_oof[val_idx, ] <- normalize_preds(
    predict(model, newdata = data.frame(scaled_log_X_train[val_idx, ]), type = "raw")
  )
  nn_test_preds <- nn_test_preds + 
    normalize_preds(predict(model, newdata = data.frame(scaled_log_X_test), type = "raw")) / N_FOLDS
}

level1_train_preds$nn <- normalize_preds(nn_oof)
level1_test_preds$nn <- normalize_preds(nn_test_preds)
cat(sprintf("  Neural Network Log Loss: %.4f\n\n", logloss(y_train, level1_train_preds$nn)))

# ----------------------------------------------------------------------------
# Combine Level 1 predictions
# ----------------------------------------------------------------------------
cat("Combining Level 1 predictions...\n")
n_models_l1 <- length(level1_train_preds)
level1_train_combined <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
level1_test_combined <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(level1_train_combined) <- CLASSES
colnames(level1_test_combined) <- CLASSES

for (model_name in names(level1_train_preds)) {
  level1_train_combined <- level1_train_combined + level1_train_preds[[model_name]]
  level1_test_combined <- level1_test_combined + level1_test_preds[[model_name]]
}
level1_train_combined <- normalize_preds(level1_train_combined / n_models_l1)
level1_test_combined <- normalize_preds(level1_test_combined / n_models_l1)

cat(sprintf("Level 1 Average Log Loss: %.4f\n\n", logloss(y_train, level1_train_combined)))

# ============================================================================
# LEVEL 2: META-MODEL (STACKING)
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("LEVEL 2: TRAINING META-MODEL (STACKING)\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Combine Level 1 predictions with row statistics as additional features
# This is a key technique from winners: use predictions + engineered features
cat("Creating meta-features (Level 1 predictions + row statistics)...\n")
meta_train <- cbind(level1_train_combined, row_stats_train)
meta_test <- cbind(level1_test_combined, row_stats_test)

cat(sprintf("Meta-features: %d dimensions\n", ncol(meta_train)))
cat(sprintf("  - Level 1 predictions: %d\n", ncol(level1_train_combined)))
cat(sprintf("  - Row statistics: %d\n\n", ncol(row_stats_train)))

# Train Level 2 XGBoost with CV
cat("Training Level 2 XGBoost...\n")
level2_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(level2_oof) <- CLASSES
level2_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(level2_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  val_idx <- folds[[fold]]
  
  dtrain <- xgb.DMatrix(data = meta_train[train_idx, ], label = y_numeric[train_idx])
  dval <- xgb.DMatrix(data = meta_train[val_idx, ])
  dtest <- xgb.DMatrix(data = meta_test)
  
  params <- list(
    objective = "multi:softprob",
    num_class = N_CLASSES,
    max_depth = 4,
    eta = 0.03,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    eval_metric = "mlogloss"
  )
  
  model <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
  
  level2_oof[val_idx, ] <- normalize_preds(
    matrix(predict(model, dval), ncol = N_CLASSES, byrow = TRUE)
  )
  level2_test_preds <- level2_test_preds + 
    normalize_preds(matrix(predict(model, dtest), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
}

final_score <- logloss(y_train, level2_oof)
cat(sprintf("\nFinal Stacked Model Log Loss: %.4f\n", final_score))

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("GENERATING SUBMISSION\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Ensure predictions are normalized
level2_test_preds_final <- normalize_preds(level2_test_preds)

# Load sample submission to get correct ID order
sample_sub <- fread(SAMPLE_SUB_FILE)

# Verify test IDs match
if (length(test_id) != nrow(sample_sub)) {
  cat("WARNING: Test ID count mismatch. Using sample submission IDs.\n")
  test_id <- sample_sub$id
} else if (!all(test_id == sample_sub$id)) {
  cat("WARNING: Test IDs don't match sample submission. Using sample submission IDs.\n")
  test_id <- sample_sub$id
}

# Create submission
submission <- data.frame(
  id = test_id,
  Class_1 = as.numeric(level2_test_preds_final[, "Class_1"]),
  Class_2 = as.numeric(level2_test_preds_final[, "Class_2"]),
  Class_3 = as.numeric(level2_test_preds_final[, "Class_3"]),
  Class_4 = as.numeric(level2_test_preds_final[, "Class_4"]),
  Class_5 = as.numeric(level2_test_preds_final[, "Class_5"]),
  Class_6 = as.numeric(level2_test_preds_final[, "Class_6"]),
  Class_7 = as.numeric(level2_test_preds_final[, "Class_7"]),
  Class_8 = as.numeric(level2_test_preds_final[, "Class_8"]),
  Class_9 = as.numeric(level2_test_preds_final[, "Class_9"])
)

# Final validation
row_sums <- rowSums(submission[, 2:10])
if (any(row_sums < 0.99) || any(row_sums > 1.01)) {
  cat("WARNING: Some predictions don't sum to 1!\n")
  cat(sprintf("  Min sum: %.6f, Max sum: %.6f\n", min(row_sums), max(row_sums)))
} else {
  cat("Validation: All predictions sum to 1 ✓\n")
}

if (any(submission[, 2:10] < 0) || any(submission[, 2:10] > 1)) {
  cat("WARNING: Some predictions outside [0, 1] range!\n")
} else {
  cat("Validation: All predictions in [0, 1] range ✓\n")
}

# Save submission
write.csv(submission, SUBMISSION_FILE, row.names = FALSE)
cat(sprintf("\nSubmission saved to: %s\n", SUBMISSION_FILE))

# ============================================================================
# SUMMARY
# ============================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SUMMARY\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")
cat("Level 1 Models:\n")
for (model_name in names(level1_train_preds)) {
  score <- logloss(y_train, level1_train_preds[[model_name]])
  cat(sprintf("  %s: %.4f\n", model_name, score))
}
cat(sprintf("\nLevel 1 Average: %.4f\n", logloss(y_train, level1_train_combined)))
cat(sprintf("Level 2 Final: %.4f\n", final_score))
cat("\nDone! ✓\n")

