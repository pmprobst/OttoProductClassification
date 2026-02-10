# Naive Bayes Model for Otto Product Classification
# Probabilistic classifier that assumes feature independence and uses
# Bayes' theorem to estimate class probabilities.

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
# FEATURE ENGINEERING
# ============================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("FEATURE ENGINEERING\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Log transformation (helps with skewed data for Naive Bayes)
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
# NAIVE BAYES MODEL
# ============================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("NAIVE BAYES MODEL\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")
cat("Naive Bayes is a probabilistic classifier that assumes feature independence\n")
cat("and uses Bayes' theorem to estimate class probabilities.\n\n")

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
  Class_1 = as.numeric(nb_test_preds[, "Class_1"]),
  Class_2 = as.numeric(nb_test_preds[, "Class_2"]),
  Class_3 = as.numeric(nb_test_preds[, "Class_3"]),
  Class_4 = as.numeric(nb_test_preds[, "Class_4"]),
  Class_5 = as.numeric(nb_test_preds[, "Class_5"]),
  Class_6 = as.numeric(nb_test_preds[, "Class_6"]),
  Class_7 = as.numeric(nb_test_preds[, "Class_7"]),
  Class_8 = as.numeric(nb_test_preds[, "Class_8"]),
  Class_9 = as.numeric(nb_test_preds[, "Class_9"])
)

# Validate
row_sums <- rowSums(submission[, 2:10])
if (any(row_sums < 0.99) || any(row_sums > 1.01)) {
  cat("WARNING: Predictions don't sum to 1!\n")
} else {
  cat("✓ All predictions sum to 1\n")
}

output_file <- file.path(OUTPUT_DIR, "submission_naive_bayes.csv")
write.csv(submission, output_file, row.names = FALSE)
cat(sprintf("\nSubmission saved to: %s\n", output_file))
cat(sprintf("Cross-Validation Log Loss: %.4f\n", nb_score))
cat("\nDone! ✓\n")

