# Three Simple Unboosted Models for Otto Product Classification
# Models: Random Forest, Multinomial Logistic Regression, Naive Bayes
# No gradient boosting - simple, interpretable models

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
SEED <- 42

# Class names
CLASSES <- paste0("Class_", 1:9)
N_CLASSES <- 9

# ============================================================================
# LOAD LIBRARIES
# ============================================================================
cat("Loading libraries...\n")
library(data.table)
library(randomForest)
library(nnet)      # For multinomial logistic regression
library(e1071)     # For Naive Bayes
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

# Convert target to factor
y_factor <- factor(y_train, levels = CLASSES)

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
# FEATURE ENGINEERING (Simple)
# ============================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("FEATURE ENGINEERING\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Log transformation (helps with skewed data)
cat("Creating log features...\n")
log_X_train <- log(X_train + 1)
log_X_test <- log(X_test + 1)

cat("Feature engineering complete!\n\n")

# ============================================================================
# CREATE CV FOLDS
# ============================================================================
set.seed(SEED)
folds <- createFolds(y_train, k = N_FOLDS, list = TRUE)

# ============================================================================
# MODEL 1: RANDOM FOREST
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("MODEL 1: RANDOM FOREST\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")
cat("Random Forest is a tree-based ensemble that builds multiple decision trees\n")
cat("on random subsets of data and features, then averages their predictions.\n\n")

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
    ntree = 300,              # Number of trees
    mtry = sqrt(ncol(X_train)),  # Features per split
    nodesize = 1,             # Minimum node size
    importance = FALSE
  )
  
  rf_oof[val_idx, ] <- normalize_preds(
    predict(model, X_train[val_idx, ], type = "prob")
  )
  rf_test_preds <- rf_test_preds + 
    normalize_preds(predict(model, X_test, type = "prob")) / N_FOLDS
}

rf_oof <- normalize_preds(rf_oof)
rf_test_preds <- normalize_preds(rf_test_preds)
rf_score <- logloss(y_train, rf_oof)
cat(sprintf("\nRandom Forest Log Loss: %.4f\n\n", rf_score))

# ============================================================================
# MODEL 2: MULTINOMIAL LOGISTIC REGRESSION
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("MODEL 2: MULTINOMIAL LOGISTIC REGRESSION\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")
cat("Multinomial logistic regression is a linear model that estimates\n")
cat("class probabilities using a softmax function over linear combinations of features.\n\n")

# Use log features for logistic regression (better for count data)
mlr_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(mlr_oof) <- CLASSES
mlr_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(mlr_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  val_idx <- folds[[fold]]
  
  # Create data frame
  train_df <- data.frame(log_X_train[train_idx, ])
  train_df$target <- y_factor[train_idx]
  
  # Train multinomial logistic regression
  model <- multinom(
    target ~ ., 
    data = train_df,
    MaxNWts = 15000,  # Maximum number of weights
    trace = FALSE
  )
  
  # Predict
  mlr_oof[val_idx, ] <- normalize_preds(
    predict(model, newdata = data.frame(log_X_train[val_idx, ]), type = "probs")
  )
  mlr_test_preds <- mlr_test_preds + 
    normalize_preds(predict(model, newdata = data.frame(log_X_test), type = "probs")) / N_FOLDS
}

mlr_oof <- normalize_preds(mlr_oof)
mlr_test_preds <- normalize_preds(mlr_test_preds)
mlr_score <- logloss(y_train, mlr_oof)
cat(sprintf("\nMultinomial Logistic Regression Log Loss: %.4f\n\n", mlr_score))

# ============================================================================
# MODEL 3: NAIVE BAYES
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("MODEL 3: NAIVE BAYES\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")
cat("Naive Bayes is a probabilistic classifier that assumes feature independence\n")
cat("and uses Bayes' theorem to estimate class probabilities.\n\n")

# Use log features for Naive Bayes
nb_oof <- matrix(0, nrow = nrow(X_train), ncol = N_CLASSES)
colnames(nb_oof) <- CLASSES
nb_test_preds <- matrix(0, nrow = nrow(X_test), ncol = N_CLASSES)
colnames(nb_test_preds) <- CLASSES

for (fold in 1:N_FOLDS) {
  cat(sprintf("  Fold %d/%d...\n", fold, N_FOLDS))
  train_idx <- setdiff(1:nrow(X_train), folds[[fold]])
  val_idx <- folds[[fold]]
  
  # Create data frame
  train_df <- data.frame(log_X_train[train_idx, ])
  train_df$target <- y_factor[train_idx]
  
  # Train Naive Bayes
  model <- naiveBayes(
    target ~ ., 
    data = train_df
  )
  
  # Predict
  nb_oof[val_idx, ] <- normalize_preds(
    predict(model, newdata = data.frame(log_X_train[val_idx, ]), type = "raw")
  )
  nb_test_preds <- nb_test_preds + 
    normalize_preds(predict(model, newdata = data.frame(log_X_test), type = "raw")) / N_FOLDS
}

nb_oof <- normalize_preds(nb_oof)
nb_test_preds <- normalize_preds(nb_test_preds)
nb_score <- logloss(y_train, nb_oof)
cat(sprintf("\nNaive Bayes Log Loss: %.4f\n\n", nb_score))

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("GENERATING SUBMISSIONS\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Load sample submission to verify ID order
sample_sub <- fread(SAMPLE_SUB_FILE)

# Function to create submission file
create_submission <- function(predictions, model_name, test_ids) {
  # Ensure IDs match
  if (length(test_ids) != nrow(sample_sub)) {
    test_ids <- sample_sub$id
  } else if (!all(test_ids == sample_sub$id)) {
    test_ids <- sample_sub$id
  }
  
  submission <- data.frame(
    id = test_ids,
    Class_1 = as.numeric(predictions[, "Class_1"]),
    Class_2 = as.numeric(predictions[, "Class_2"]),
    Class_3 = as.numeric(predictions[, "Class_3"]),
    Class_4 = as.numeric(predictions[, "Class_4"]),
    Class_5 = as.numeric(predictions[, "Class_5"]),
    Class_6 = as.numeric(predictions[, "Class_6"]),
    Class_7 = as.numeric(predictions[, "Class_7"]),
    Class_8 = as.numeric(predictions[, "Class_8"]),
    Class_9 = as.numeric(predictions[, "Class_9"])
  )
  
  # Validate
  row_sums <- rowSums(submission[, 2:10])
  if (any(row_sums < 0.99) || any(row_sums > 1.01)) {
    cat(sprintf("WARNING: %s predictions don't sum to 1!\n", model_name))
  }
  
  output_file <- file.path(OUTPUT_DIR, paste0("submission_", model_name, ".csv"))
  write.csv(submission, output_file, row.names = FALSE)
  cat(sprintf("  Saved: %s\n", output_file))
  
  return(submission)
}

# Create submissions for each model
cat("Creating submission files...\n")
create_submission(rf_test_preds, "random_forest", test_id)
create_submission(mlr_test_preds, "multinomial_logistic", test_id)
create_submission(nb_test_preds, "naive_bayes", test_id)

# ============================================================================
# SUMMARY
# ============================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SUMMARY\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")
cat("Model Performance (Cross-Validation Log Loss):\n")
cat(sprintf("  1. Random Forest:              %.4f\n", rf_score))
cat(sprintf("  2. Multinomial Logistic Reg:    %.4f\n", mlr_score))
cat(sprintf("  3. Naive Bayes:                 %.4f\n", nb_score))
cat("\nSubmission files created:\n")
cat("  - output/submission_random_forest.csv\n")
cat("  - output/submission_multinomial_logistic.csv\n")
cat("  - output/submission_naive_bayes.csv\n")
cat("\nDone! ✓\n")

