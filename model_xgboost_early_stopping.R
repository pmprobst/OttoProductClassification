# XGBoost Classifier with Early Stopping
# Based on DataRobot-style parameters with early stopping

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR <- "data"
TRAIN_FILE <- file.path(DATA_DIR, "train.csv")
TEST_FILE <- file.path(DATA_DIR, "test.csv")
SAMPLE_SUB_FILE <- file.path(DATA_DIR, "sampleSubmission.csv")
OUTPUT_DIR <- "output"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# Cross-validation
N_FOLDS <- 5
SEED <- 1234  # From parameters

# Class names
CLASSES <- paste0("Class_", 1:9)
N_CLASSES <- 9

# ============================================================================
# LOAD LIBRARIES
# ============================================================================
cat("Loading libraries...\n")
library(data.table)
library(xgboost)
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

# Convert target to numeric (0-indexed for XGBoost)
y_factor <- factor(y_train, levels = CLASSES)
y_numeric <- as.numeric(y_factor) - 1

# ============================================================================
# MISSING VALUE IMPUTATION
# ============================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("MISSING VALUE IMPUTATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Parameters from specification
THRESHOLD <- 10
MIN_COUNT_NA <- 5
MISSING_VALUE <- 0.0

# Check for missing values
na_counts <- colSums(is.na(X_train))
na_cols <- which(na_counts > 0)

if (length(na_cols) > 0) {
  cat(sprintf("Found %d columns with missing values\n", length(na_cols)))
  
  for (col in na_cols) {
    na_count <- na_counts[col]
    if (na_count >= MIN_COUNT_NA) {
      # Impute with specified missing value
      X_train[is.na(X_train[, col]), col] <- MISSING_VALUE
      X_test[is.na(X_test[, col]), col] <- MISSING_VALUE
      cat(sprintf("  Column %d: %d missing values imputed with %.1f\n", 
                  col, na_count, MISSING_VALUE))
    }
  }
} else {
  cat("No missing values found in training data\n")
}

cat("Missing value imputation complete!\n\n")

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
# CREATE CV FOLDS
# ============================================================================
set.seed(SEED)
folds <- createFolds(y_train, k = N_FOLDS, list = TRUE)

# ============================================================================
# XGBOOST MODEL WITH EARLY STOPPING
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("XGBOOST CLASSIFIER WITH EARLY STOPPING\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# XGBoost parameters (from specification)
xgb_params <- list(
  objective = "multi:softprob",
  num_class = N_CLASSES,
  eval_metric = "mlogloss",
  
  # Learning parameters
  learning_rate = 0.05,
  n_estimators = 2500,  # Maximum, but early stopping will stop earlier
  
  # Tree structure
  max_depth = 5,
  min_child_weight = 1.0,
  min_split_loss = 0.01,  # gamma
  
  # Regularization
  reg_lambda = 1.0,  # L2 regularization
  reg_alpha = 0.0,   # L1 regularization
  
  # Sampling
  subsample = 1.0,
  colsample_bytree = 0.3,
  colsample_bylevel = 1.0,
  
  # Other parameters
  scale_pos_weight = 1.0,
  max_delta_step = 0.0,
  num_parallel_tree = 1,
  tree_method = "auto",
  max_bin = 256,
  
  # Missing value handling
  missing = MISSING_VALUE
)

cat("XGBoost Parameters:\n")
cat(sprintf("  Learning rate: %.3f\n", xgb_params$learning_rate))
cat(sprintf("  Max depth: %d\n", xgb_params$max_depth))
cat(sprintf("  Min child weight: %.1f\n", xgb_params$min_child_weight))
cat(sprintf("  Subsample: %.1f\n", xgb_params$subsample))
cat(sprintf("  Colsample bytree: %.1f\n", xgb_params$colsample_bytree))
cat(sprintf("  Reg lambda: %.1f\n", xgb_params$reg_lambda))
cat(sprintf("  Max estimators: %d\n", xgb_params$n_estimators))
cat(sprintf("  Early stopping rounds: %d\n", 10))  # interval from params
cat("\n")

# Early stopping parameters
EARLY_STOPPING_ROUNDS <- 10  # Based on interval: 10
EARLY_STOPPING_INIT <- 5     # From early_stopping_init

xgb_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(xgb_oof) <- CLASSES
xgb_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(xgb_test_preds) <- CLASSES

best_iterations <- numeric(N_FOLDS)

for (fold in 1:N_FOLDS) {
  cat(sprintf("Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  val_idx <- folds[[fold]]
  
  # Create DMatrix objects
  dtrain <- xgb.DMatrix(data = X_train[train_idx, ], label = y_numeric[train_idx])
  dval <- xgb.DMatrix(data = X_train[val_idx, ], label = y_numeric[val_idx])
  dtest <- xgb.DMatrix(data = X_test)
  
  # Watchlist for early stopping
  watchlist <- list(train = dtrain, eval = dval)
  
  # Train with early stopping
  set.seed(SEED)
  model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = xgb_params$n_estimators,
    watchlist = watchlist,
    early_stopping_rounds = EARLY_STOPPING_ROUNDS,
    verbose = 0,
    print_every_n = 50
  )
  
  # Get best iteration
  best_iter <- model$best_iteration
  best_iterations[fold] <- best_iter
  cat(sprintf("  Best iteration: %d (stopped early at %d)\n", 
              best_iter, model$best_iteration))
  cat(sprintf("  Best score: %.6f\n", model$best_score))
  
  # Predict on validation set
  xgb_oof[val_idx, ] <- normalize_preds(
    matrix(predict(model, dval), ncol = N_CLASSES, byrow = TRUE)
  )
  
  # Predict on test set
  xgb_test_preds <- xgb_test_preds + 
    normalize_preds(matrix(predict(model, dtest), ncol = N_CLASSES, byrow = TRUE)) / N_FOLDS
}

xgb_oof <- normalize_preds(xgb_oof)
xgb_test_preds <- normalize_preds(xgb_test_preds)
xgb_score <- logloss(y_train, xgb_oof)

cat("\n")
cat(sprintf("Average best iteration: %.1f\n", mean(best_iterations)))
cat(sprintf("Iteration range: %d - %d\n", min(best_iterations), max(best_iterations)))
cat(sprintf("\nXGBoost Log Loss: %.4f\n\n", xgb_score))

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("GENERATING SUBMISSION\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Load sample submission to verify ID order
sample_sub <- fread(SAMPLE_SUB_FILE)

# Ensure IDs match
if (length(test_id) != nrow(sample_sub)) {
  test_id <- sample_sub$id
} else if (!all(test_id == sample_sub$id)) {
  test_id <- sample_sub$id
}

submission <- data.frame(
  id = test_id,
  Class_1 = as.numeric(xgb_test_preds[, "Class_1"]),
  Class_2 = as.numeric(xgb_test_preds[, "Class_2"]),
  Class_3 = as.numeric(xgb_test_preds[, "Class_3"]),
  Class_4 = as.numeric(xgb_test_preds[, "Class_4"]),
  Class_5 = as.numeric(xgb_test_preds[, "Class_5"]),
  Class_6 = as.numeric(xgb_test_preds[, "Class_6"]),
  Class_7 = as.numeric(xgb_test_preds[, "Class_7"]),
  Class_8 = as.numeric(xgb_test_preds[, "Class_8"]),
  Class_9 = as.numeric(xgb_test_preds[, "Class_9"])
)

# Validate
row_sums <- rowSums(submission[, 2:10])
if (any(row_sums < 0.99) || any(row_sums > 1.01)) {
  cat("WARNING: Predictions don't sum to 1!\n")
  cat(sprintf("  Min sum: %.6f, Max sum: %.6f\n", min(row_sums), max(row_sums)))
} else {
  cat("✓ All predictions sum to 1\n")
}

if (any(submission[, 2:10] < 0) || any(submission[, 2:10] > 1)) {
  cat("WARNING: Some predictions outside [0, 1] range!\n")
} else {
  cat("✓ All predictions in [0, 1] range\n")
}

output_file <- file.path(OUTPUT_DIR, "submission_xgboost_early_stopping.csv")
write.csv(submission, output_file, row.names = FALSE)
cat(sprintf("\nSubmission saved to: %s\n", output_file))
cat(sprintf("Cross-Validation Log Loss: %.4f\n", xgb_score))
cat(sprintf("Average iterations used: %.1f (max: %d)\n", 
            mean(best_iterations), xgb_params$n_estimators))
cat("\nDone! ✓\n")

